
from utils.tools import Tools
from utils import *

import os
import json
import random

class DataGenerationPipeline:
    def __init__(self):
        seed_question = Tools.load_json("./Seed/UserQuestionSeed.json")

    def pipeline(self, data_path, **kwargs):
        """
        遍历 data_path 下的 Fake 和 Real 文件夹，识别图片和视频文件，
        并将信息写入 JSONLINE 格式的中间数据文件。

        每一行的格式为：
        {
            "label": "Fake" 或 "Real",
            "path": "相对路径",
            "desc": "文件名和类型描述",
            "type": "video" 或 "image"
        }

        参数:
            data_path (str): 数据目录路径，包含 Fake 和 Real 子目录。
            output_jsonl (str): 输出 JSONL 文件路径。
        """
        # 支持的视频和图片扩展名
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv'}

        output_jsonl = os.path.join(data_path, "data.jsonl")
        output_sft = os.path.join(data_path, "sft.json")
        # 存储所有记录
        records = []
        sft_records = []

        # 遍历 Fake 和 Real 文件夹
        for label in ["fake", "real"]:
            label_dir = os.path.join(data_path, label)
            if not os.path.isdir(label_dir):
                continue
            
            for root, dirs, files in os.walk(label_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    rel_path = os.path.relpath(os.path.join(root, file), data_path)

                    if ext in image_exts:
                        file_type = "image"
                    elif ext in video_exts:
                        file_type = "video"
                    else:
                        continue  # 忽略不支持的格式
                    # 1. 构造 JSONL 格式记录
                    record = {
                        "label": label,
                        "path": rel_path.replace("\\", "/"),  # 保证路径兼容性
                        "desc": desc,
                        "type": file_type
                    }
                    records.append(record)

                    # 2. 构造 SFT 格式记录
                    if file_type == "image":
                        sft_record = {
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": "<image>"+self._random_choice_question().format(media=file_type)
                                },
                                {
                                    "from": "gpt",
                                    "value": desc
                                }
                            ],
                            "images": [rel_path]
                        }
                    elif file_type == "video":
                        sft_record = {
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": "<video>"+self._random_choice_question().format(media=file_type)
                                },
                                {
                                    "from": "gpt",
                                    "value": desc
                                }
                            ],
                            "videos": [rel_path]
                        }
                    sft_records.append(sft_record)

        # 写入 jsonl 文件
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # 写入 sft.json 文件
        with open(output_sft, 'w', encoding='utf-8') as f:
            json.dump(sft_records, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 完成处理，生成 {len(records)} 条记录")

    def _random_choice_question(self):
        return random.choice(self.seed_question)

dataGenerationPipeline = DataGenerationPipeline()

if __name__ == "__main__":
    data_path = "D:/Data/DeepFakeDetection/DeepFakeDetection"