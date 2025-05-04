
import json
import os
import time
import math
import hashlib
import requests
from openai import OpenAI
from utils.tools import Tools
from loguru import logger

import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu

from utils.qwen2_5_utils import *

class OpenaiPipeline:
    def __init__(self, config_path, prompt_path):
        config = Tools.load_json(config_path)
        prompt_config = Tools.load_json(prompt_path)
        self.client = OpenAI(base_url=config["Openai"]["url"], api_key=config["Openai"]["APIKey"])
        self.model_name = config["Openai"]["model_name"]
        self.system_prompt = "You are a AIGC Detector. Determine whether the following {{Type}} is AI-generated (fake) or human-produced (real). Provide your judgment clearly enclosed in <conclusion>fake or real</conclusion>. The conclusion is enclosed within <conclusion> </conclusion> tags, <conclusion> real or fake </conclusion>. <conclusion> content must strictly be (real/fake)."
        self.user_prompt = "Is the following {{Type}} fake or real?"

        self.task_map = {
            "cls": self.infer_with_cls,
            "plausibility": self.infer_with_plausibility
        }

        
    def infer(self, messages, **kwargs):
        if kwargs.get("extra_body"):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096,
                extra_body={
                    "mm_processor_kwargs": kwargs.get("extra_body")
                }
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=4096
            )
        
        return response.choices[0].message.content.strip()
    def infer_with_cls(self, file_path):
        Type = Tools.get_file_type(file_path)
        system_prompt = self.system_prompt.replace("{{Type}}", Type)
        user_prompt = self.user_prompt.replace("{{Type}}", Type)
        response = self.infer(file_path, system_prompt, user_prompt)
        return response

    def infer_with_plausibility(self, file_path):
        Type = Tools.get_file_type(file_path)


    def pipeline(self, data_path, to_path, task="cls"):
        results = []
        for root, dirs,files in os.walk(data_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file)[1].lower()
                if file_extension in Tools.extensions:
                    label = Tools.get_file_type(file_path)
                    assert label is not None
                    to_item = {
                        "file_path": file_path,
                        "label": label,
                        "task": task,
                        "model": self.model_name,
                        "raw_predict": self.task_map[task](file_path),
                        "predict": ""
                    }
                    to_item["predict"] = Tools.get_think_tag_directly(to_item["raw_predict"])
                    results.append(to_item)
            if len(results) % 50 == 0:
                with open(to_path, 'w', encoding='utf-8') as json_file:
                    json.dump(results, json_file, ensure_ascii=False, indent=2)

        with open(to_path, 'w', encoding='utf-8') as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=2)
        return results

    def get_video_frames(self,video_path, num_frames=128, cache_dir='.cache'):
        os.makedirs(cache_dir, exist_ok=True)

        video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
        if video_path.startswith('http://') or video_path.startswith('https://'):
            video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
            if not os.path.exists(video_file_path):
                self.download_video(video_path, video_file_path)
        else:
            video_file_path = video_path

        frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
        timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

        if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
            frames = np.load(frames_cache_file)
            timestamps = np.load(timestamps_cache_file)
            return video_file_path, frames, timestamps

        vr = VideoReader(video_file_path, ctx=cpu(0))
        total_frames = len(vr)

        indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

        np.save(frames_cache_file, frames)
        np.save(timestamps_cache_file, timestamps)
        
        return video_file_path, frames, timestamps

    def create_image_grid(self,images, num_columns=8):
        pil_images = [Image.fromarray(image) for image in images]
        num_rows = math.ceil(len(images) / num_columns)

        img_width, img_height = pil_images[0].size
        grid_width = num_columns * img_width
        grid_height = num_rows * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height))

        for idx, image in enumerate(pil_images):
            row_idx = idx // num_columns
            col_idx = idx % num_columns
            position = (col_idx * img_width, row_idx * img_height)
            grid_image.paste(image, position)

        return grid_image

    def download_video(self, url, dest_path):
        response = requests.get(url, stream=True)
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8096):
                f.write(chunk)
        print(f"Video downloaded to {dest_path}")

openaiLLMs = OpenaiPipeline("./Config/LLMsConfig.json", prompt_path="./Config/PromptsConfig.json")

if __name__ == "__main__":
    openaiLLMs.pipeline(
        data_path="./Data/DetectionData",
        to_path="./Results/OpenaiResults.json",
        task="cls"
    )