import sys
import os
# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import base64
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from utils.qwen2_5_utils import process_vision_info


# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://172.17.65.43:30001/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)


video_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "text", "text": "请用表格总结一下视频中的商品特点"},
        {
            "type": "video",
            "video": "/raid/share/jiangchangjiang/aws/dataset/demamba/train_I2VGEN_XL/I2VGEN_XL_001.mp4",
            "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2, 
            'fps': 3.0  # The default value is 2.0, but for demonstration purposes, we set it to 3.0.
        }]
    },
]


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


video_messages, video_kwargs = prepare_message_for_vllm(video_messages)
chat_response = client.chat.completions.create(
    model="/public/data_share/model_hub/lab10_model/Qwen/Qwen2.5-VL-7B-Instruct",
    messages=video_messages,
    extra_body={
        "mm_processor_kwargs": video_kwargs
    }
)
print("Chat response:", chat_response)

# import base64
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from openai import OpenAI
# from utils.qwen2_5_utils import process_vision_info


# # Set OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://172.17.65.43:30001/v1"

# client = OpenAI(
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )


# video_messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": [
#         {"type": "text", "text": "请用表格总结一下视频中的商品特点"},
#         {
#             "type": "video",
#             "video": "https://duguang-labelling.oss-cn-shanghai.aliyuncs.com/qiansun/video_ocr/videos/50221078283.mp4",
#             "total_pixels": 20480 * 28 * 28, "min_pixels": 16 * 28 * 2, 
#             'fps': 3.0  # The default value is 2.0, but for demonstration purposes, we set it to 3.0.
#         }]
#     },
# ]


# def prepare_message_for_vllm(content_messages):
#     """
#     The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
#     Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_typ`e of the video explicitly set to `video/jpeg`.
#     By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
#     """
#     vllm_messages, fps_list = [], []
#     for message in content_messages:
#         message_content_list = message["content"]
#         if not isinstance(message_content_list, list):
#             vllm_messages.append(message)
#             continue

#         new_content_list = []
#         for part_message in message_content_list:
#             if 'video' in part_message:
#                 video_message = [{'content': [part_message]}]
#                 image_inputs, video_inputs, video_kwargs = process_vision_info(video_message, return_video_kwargs=True)
#                 assert video_inputs is not None, "video_inputs should not be None"
#                 video_input = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
#                 fps_list.extend(video_kwargs.get('fps', []))

#                 # encode image with base64
#                 base64_frames = []
#                 for frame in video_input:
#                     img = Image.fromarray(frame)
#                     output_buffer = BytesIO()
#                     img.save(output_buffer, format="jpeg")
#                     byte_data = output_buffer.getvalue()
#                     base64_str = base64.b64encode(byte_data).decode("utf-8")
#                     base64_frames.append(base64_str)

#                 part_message = {
#                     "type": "video_url",
#                     "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
#                 }
#             new_content_list.append(part_message)
#         message["content"] = new_content_list
#         vllm_messages.append(message)
#     return vllm_messages, {'fps': fps_list}


# video_messages, video_kwargs = prepare_message_for_vllm(video_messages)
# chat_response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-VL-7B-Instruct",
#     messages=video_messages,
#     extra_body={
#         "mm_processor_kwargs": video_kwargs
#     }
# )
# print("Chat response:", chat_response)