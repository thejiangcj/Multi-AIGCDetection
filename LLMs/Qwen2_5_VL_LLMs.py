from utils.qwen2_5_utils import process_vision_info
from utils.tools import Tools


class Qwen2_5_VL_LLMs:
    @staticmethod
    def get_messages(file_path,file_type):
        if file_type == "video":
            return Qwen2_5_VL_LLMs.get_messages_with_video(file_path)

    @staticmethod
    def get_messages_with_video(file_path, system_prompt, user_prompt):
        video_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "video",
                    "video": file_path,
                    "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2, 
                    'fps': 2.0  # The default value is 2.0, but for demonstration purposes, we set it to 3.0.
                }]
            },
        ]
        video_messages, video_kwargs = Qwen2_5_VL_LLMs.prepare_message_for_vllm(video_messages)
        return video_messages, video_kwargs

    @staticmethod
    def prepare_message_for_vllm(content_messages):
        """
        The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
        Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
        By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
        """
        vllm_messages, fps_list = [], []
        for message in content_messages:
            message_content_list = message["content"]
            if not isinstance(message_content_list, list):
                vllm_messages.append(message)
                continue

            new_content_list = []
            for part_message in message_content_list:
                if 'video' in part_message:
                    video_message = [{'content': [part_message]}]
                    image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
                    assert video_inputs is not None, "video_inputs should not be None"
                    video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                    fps_list.extend(video_kwargs.get('fps', []))

                    # encode image with base64
                    base64_frames = []
                    for frame in video_input:
                        img = Image.fromarray(frame)
                        output_buffer = BytesIO()
                        img.save(output_buffer, format="jpeg")
                        byte_data = output_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data).decode("utf-8")
                        base64_frames.append(base64_str)

                    part_message = {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                    }
                new_content_list.append(part_message)
            message["content"] = new_content_list
            vllm_messages.append(message)
        return vllm_messages, {'fps': fps_list}

        