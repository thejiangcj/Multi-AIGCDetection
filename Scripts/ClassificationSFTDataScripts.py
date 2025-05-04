import os
import json
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from loguru import logger

from moviepy import VideoFileClip
from utils.tools import Tools
from multiprocessing import Pool
from functools import partial
## pip install moviepy imageio loguru

class ClassificationSFTDataScripts:
    def __init__(self):
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP')
        self.video_extensions = ('.mp4','.gif')
        self.extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.mp4')

        self.system_prompt = "You are a AIGC Detector. Determine whether the following {{Type}} is AI-generated (fake) or human-produced (real). Provide your judgment clearly enclosed in <conclusion>fake or real</conclusion>. The conclusion is enclosed within <conclusion> </conclusion> tags, <conclusion> real or fake </conclusion>. <conclusion> content must strictly be (real/fake)."
        self.user_prompt = "Is the following {{Type}} fake or real?"

    def get_data_dict(self, file_type, system_prompt, user_prompt, label, rel_path, data_type="sharegpt", **kwargs):
        if data_type=="sharegpt":
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
                        rel_path
                    ],
                "system": system_prompt
            }
        elif data_type=="llava":
            data_dict = {
                "id": Tools.generate_id(rel_path),
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{system_prompt}\n<{file_type}>\n{user_prompt}"
                    },
                    {
                        "from": "gpt",
                        "value": f"<conclusion> {label} </conclusion>"
                    }
                ],
                "data_source": kwargs["data_source"],
                f"{file_type}": rel_path
            }
        return data_dict
    def convert_gif_to_mp4(self, gif_path):
        mp4_path = os.path.splitext(gif_path)[0] + '.mp4'
        try:
            clip = VideoFileClip(gif_path)
            clip.write_videofile(mp4_path, codec='libx264', audio=False)
            logger.info(f"转换成功: {gif_path} -> {mp4_path}")
        except Exception as e:
            logger.error(f"转换失败: {gif_path}, 错误信息: {e}")

    def process_file(self, file_path, data_path, data_type):
        relative_path = os.path.relpath(file_path, os.path.dirname(data_path))
        label = "real" if "real" in relative_path.lower() else "fake"

        if file_path.lower().endswith(self.image_extensions):
            file_type = "image"
        elif file_path.lower().endswith(self.video_extensions):
            file_type = "video"
        else:
            return None  # unsupported file

        user_prompt = self.user_prompt.replace("{{Type}}", file_type)
        system_prompt = self.system_prompt.replace("{{Type}}", file_type)

        return self.get_data_dict(
            file_type, system_prompt, user_prompt,
            label, relative_path, data_type=data_type,
            data_source=os.path.basename(data_path)
        )

    def pipeline_parallel(self, data_path, to_path, data_type="llava"):
        assert to_path.endswith(".json"), "to_path must be a json file"
        
        # Step 1: 收集所有支持的文件路径
        all_files = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith(self.extensions):
                    all_files.append(os.path.join(root, file))

        logger.info(f"共找到 {len(all_files)} 个文件，开始并行处理...")

        # Step 2: 多进程处理
        with Pool(processes=20) as pool:
            func = partial(self.process_file, data_path=data_path, data_type=data_type)
            results = pool.map(func, all_files)

        # Step 3: 清理 None 并保存结果
        to_data = [r for r in results if r is not None]
        logger.info(f"成功处理 {len(to_data)} 个文件，写入 {to_path}")

        with open(to_path, 'w', encoding='utf-8') as f:
            json.dump(to_data, f, ensure_ascii=False, indent=2)

    def pipeline(self, data_path, to_path, data_type="llava"):
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
                    # if file_path.lower().endswith('.gif'):
                    #     gif_path.append(file_path)
                    #     file_name = ".".join(file.split(".")[:-1])
                    #     abs_path = os.path.join(os.path.dirname(abs_path),file_name + ".mp4")
                    relative_path = os.path.relpath(file_path, os.path.dirname(data_path))
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
                    data_dict = self.get_data_dict(file_type, system_prompt, user_prompt, label, relative_path, data_type=data_type, data_source=os.path.basename(data_path))
                    numbers += 1
                    to_data.append(data_dict)

                    if numbers < 50000 and numbers % 1000 == 0:
                        logger.info(f"Processed {numbers} files. Saving to {to_path}...")
                        with open(to_path, 'w', encoding='utf-8') as f:
                            json.dump(to_data, f, ensure_ascii=False, indent=2)
                        
        with open(to_path, 'w', encoding='utf-8') as f:
            json.dump(to_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Total {numbers} files processed. Data saved to {to_path}")

if __name__ == "__main__":
    # Example usage
    data_path = "/raid/share/jiangchangjiang/Multi-AIGCDetection/modelscope_data/classifcation/video/train/classification/video/train/pika" # 数据文件夹路径，路径下包含fake或real即可
    to_path = "/raid/share/jiangchangjiang/Multi-AIGCDetection/dataset/classification/video/train/pika-train-llava.json" # 保存的sft格式的文件路径
    classification_sft_data_scripts = ClassificationSFTDataScripts()
    classification_sft_data_scripts.pipeline_parallel(data_path, to_path, data_type="llava")