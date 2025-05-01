import os
import json

from loguru import logger

from moviepy.editor import VideoFileClip

## pip install moviepy imageio loguru

class ClassificationSFTDataScripts:
    def __init__(self):
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP')
        self.video_extensions = ('.mp4','.gif')
        self.extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.mp4')

        self.system_prompt = "You are a AIGC Detector. Determine whether the following {{Type}} is AI-generated (fake) or human-produced (real). Provide your judgment clearly enclosed in <conclusion>fake or real</conclusion>. The conclusion is enclosed within <conclusion> </conclusion> tags, <conclusion> real or fake </conclusion>. <conclusion> content must strictly be (real/fake)."
        self.user_prompt = "Is the following {{Type}} fake or real?"

    def convert_gif_to_mp4(self, gif_path):
        mp4_path = os.path.splitext(gif_path)[0] + '.mp4'
        try:
            clip = VideoFileClip(gif_path)
            clip.write_videofile(mp4_path, codec='libx264', audio=False)
            logger.info(f"转换成功: {gif_path} -> {mp4_path}")
        except Exception as e:
            logger.error(f"转换失败: {gif_path}, 错误信息: {e}")

    def pipeline(self, data_path, to_path):
        """
        data_path: str
            Path to the dataset. 图片或视频文件需要包含在fake/real 文件夹下，作为 label
        to_path: str. json
        """
        assert to_path.endswith(".json"), "to_path must be a json file"
        to_data = []
        numbers = 0
        gif_path = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(self.extensions):
                    file_path = os.path.join(root, file)
                        
                    abs_path = os.path.abspath(file_path)
                    if file_path.lower().endswith('.gif'):
                        gif_path.append(file_path)
                        file_name = ".".join(file.split(".")[:-1])
                        abs_path = os.path.join(os.path.dirname(abs_path),file_name + ".mp4")
                    relative_path = os.path.relpath(file_path, data_path)
                    label = "real" if "real" in relative_path.lower() else "fake"
                    file_type = ""
                    if file.lower().endswith(self.image_extensions):
                        file_type = "image"
                    elif file.lower().endswith(self.video_extensions):
                        file_type = "video"
                    else:
                        continue

                    user_prompt = self.user_prompt.replace("{{Type}}", file_type)
                    system_prompt = self.system_prompt.replace("{{Type}}", file_type)
                    data_dict = {
                        "messages": [
                            {
                                "content": f"<{file_type}>{user_prompt}",
                                "role": "user"
                            },
                            {
                                "content": f"<conclusion> {label} </conclusion>",
                                "role": "assistant"
                            }
                            ],
                            f"{file_type}s": [
                                abs_path
                            ],
                        "system": system_prompt
                    }
                    numbers += 1
                    to_data.append(data_dict)

                    if numbers % 1000 == 0:
                        with open(to_path, 'w', encoding='utf-8') as f:
                            json.dump(to_data, f, ensure_ascii=False, indent=4)
                        
        with open(to_path, 'w', encoding='utf-8') as f:
            json.dump(to_data, f, ensure_ascii=False, indent=4)
        logger.info(f"Total {numbers} files processed. Data saved to {to_path}")
        if len(gif_path) > 0:
            logger.info(f"GIF files found: {gif_path}")
            logger.info("convert GIF files to MP4 format manually.")
            _ = [self.convert_gif_to_mp4(i) for i in gif_path]
            logger.info("convert GIF files to MP4 format successfully.")






if __name__ == "__main__":
    # Example usage
    data_path = "path/to/your/dataset"
    to_path = "./sft.json"
    classification_sft_data_scripts = ClassificationSFTDataScripts()
    classification_sft_data_scripts.pipeline(data_path, to_path)