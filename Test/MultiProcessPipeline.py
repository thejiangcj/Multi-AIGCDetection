# !pip install -q -U google-genai pandas openpyxl loguru modelscope

import json
import hashlib
import re
import os
from loguru import logger
import sys
from google import genai
from google.genai import types

import time
import traceback

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from modelscope.hub.snapshot_download import snapshot_download
import subprocess,shutil
from multiprocessing import Manager
# from multiprocessing import Pool

## 免费api
api = "AIzaSyBcesr-WGrESSHZf1YlLUPbCodIeMi-XVg"


debug = True
workers = 3

class Tools:
    @staticmethod
    def to_file(data, to_path):
        with open(to_path, 'w', encoding='utf-8') as jsonl_file:
            for entry in data:
                jsonl_file.write(json.dumps(entry) + '\n')
        # 保存到 JSON 文件
        # dir_name = os.path.dirname(to_path)
        # basename = os.path.join(dir_name,os.path.splitext(os.path.basename(to_path))[0] + ".json")

        # with open(basename, 'w', encoding='utf-8') as json_file:
        #     json.dump(data_results, json_file, ensure_ascii=False, indent=2)

    @staticmethod
    def to_json(data_results, to_path):
        # 保存到 JSON 文件
        dir_name = os.path.dirname(to_path)
        basename = os.path.join(dir_name,os.path.splitext(os.path.basename(to_path))[0] + ".json")

        with open(basename, 'w', encoding='utf-8') as json_file:
            json.dump(data_results, json_file, ensure_ascii=False, indent=2)

    @staticmethod
    def load_prompt(prompt_path):
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    @staticmethod
    def load_json(json_path):
        with open(json_path, 'r', encoding='utf-8') as file:
            return json.load(
                file
            )
    @staticmethod
    def generate_id(text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @staticmethod
    def load_jsonl(jsonl_path):
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            return [json.loads(line) for line in file]
    @staticmethod
    def has_think_tags(text):
        """检查是否存在<pros>标签"""
        return '<think>' in text and '</think>' in text

    @staticmethod
    def has_conclusion_tags(text):
        """检查是否存在<conclusion>标签"""
        return '<conclusion>' in text and '</conclusion>' in text

    @staticmethod
    def extract_think_content(text):
        """提取think标签内容（带缓存）"""
        first_think_pos = text.find('<think>')
        last_think_pos = text.rfind('</think>')

        if first_think_pos != -1 and last_think_pos != -1:
            # 提取匹配到的内容
            start = first_think_pos + len('<think>')
            end = last_think_pos
            return text[start:end].strip()
        elif first_think_pos != -1 and last_think_pos == -1 and text.rfind('<conclusion>') !=-1:
          start = first_think_pos + len('<think>')
          end = text.rfind('<conclusion>')
          return text[start:end].strip()
        return ''

    @staticmethod
    def extract_conclusion_content(text):
        """提取conclusion标签内容（带缓存）"""
        first_think_pos = text.find('<think>')
        last_think_pos = text.rfind('</think>')

        if first_think_pos != -1 and last_think_pos != -1:
            # 提取匹配到的内容
            start = first_think_pos + len('<think>')
            end = last_think_pos + len("</think>")
            match = re.search(r'<conclusion>(.*?)</conclusion>', text[end:], re.DOTALL)
            return match.group(1).strip() if match else ''
        elif first_think_pos != -1 and last_think_pos == -1 and text.rfind('<conclusion>') !=-1:

          start = first_think_pos + len('<think>')
          end = text.rfind('<conclusion>')
          match = re.search(r'<conclusion>(.*?)</conclusion>', text[end:], re.DOTALL)
          return match.group(1).strip() if match else ''
        return ''

    @staticmethod
    def conclusion_has_single_word(text):
        """检查conclusion内容是否只有一个单词"""
        content = Tools.extract_conclusion_content(text)
        return True if "fake" in content or "real" in content else False

    @staticmethod
    def label_vs(text, label):
        """检查conclusion内容是否只有一个单词"""
        # text = Tools.extract_conclusion_content(text)
        to_label = 0
        if text.strip().lower() == label.strip().lower():
            to_label = 1
        elif f"is {label}" in text.strip().lower():
            to_label = 1
        elif text.strip().startswith(label) or text.strip().endswith(label):
            to_label = 1
        elif f"as {label}" in text.strip().lower() or f"as '{label}'" in text.strip().lower():
            to_label = 1
        elif "synthetic generation" in text and label=="fake":
            to_label = 1
        elif "real capture" in text and label=="real":
            to_label = 1

        return to_label
    @staticmethod
    def deduplicate_by_id(data):
        seen_ids = set()
        deduped = []
        for item in data:
            item_id = item["rawInfer"]["id"]
            if item_id not in seen_ids:
                seen_ids.add(item_id)
                deduped.append(item)
        return deduped

class GeminiPipeline:
    def __init__(self):
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"
        self.search_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', 'mp4']

    def is_large_20(self, file_path):
        """
        是否大于 20MB
        """
        # 获取文件的大小（单位：字节）
        file_size = os.path.getsize(file_path)

        # 将 20MB 转换为字节
        size_limit = 20 * 1024 * 1024  # 20MB = 20 * 1024 * 1024 bytes

        # 判断文件大小是否超过 20MB
        return file_size >= size_limit

    def infer_with_base64(self, file_path, file_type, **kwargs):
        client = genai.Client(api_key=api)
        with open(file_path, 'rb') as f:
            filebytes = f.read()
        response = client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(
                    data=filebytes,
                    mime_type=file_type,
                ),
                kwargs["user_prompt"]
            ],
            config=types.GenerateContentConfig(
                system_instruction=kwargs["system_prompt"]
            )
        )
        return response

    def infer_with_upload(self, file_path, file_type, **kwargs):
        client = genai.Client(api_key=api)
        uploaded_file = client.files.upload(file=file_path)
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text=kwargs["system_prompt"]),
            ],
        )
        while uploaded_file.state.name == "PROCESSING":
            print("processing video...")
            time.sleep(5)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        result = self.client.models.generate_content(
            model=self.model_name,
            contents=[uploaded_file, kwargs["user_prompt"]],
            config=generate_content_config
        )
        # if result.text is None:
        #     logger.debug(f"DP is: {result}")
        return result.text

    def infer_with_file(self, file_path):
        ## 判断图片是否大于20MB

        prompt_config = {
            "image": "/content/drive/MyDrive/Project/AIGC-Detection/Prompts/ImageInferWithLabelPrompt.txt",
            "video": "/content/drive/MyDrive/Project/AIGC-Detection/Prompts/VideoInferWithLabelPrompt.txt"
        }

        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        video_extensions = ['.mp4']

        system_prompt_map = {
            "image": Tools.load_prompt(prompt_config["image"]),
            "video": Tools.load_prompt(prompt_config["video"])
        }


        result=""
        file_type = None
        system_prompt,user_prompt = "",""
        uid = Tools.generate_id(file_path)

        Label = "real" if "real" in file_path else "fake"


        if self.is_image_file(file_path):
            file_type = "image/jpeg"
            Type = "image"
            system_prompt,user_prompt = system_prompt_map[Type], f"This {Type} is {Label}, explain the reason."
        elif self.is_video_file(file_path):
            file_type = "video/mp4"
            Type = "video"
            system_prompt,user_prompt = system_prompt_map[Type],f"This {Type} is {Label}, explain the reason."
        else:
            logger.error(f"文件类型不确定：{file_path}")
            return result.text

        ## 保存为字典
        meta_dict = {
            "id": uid,
            "label": Label,
            "type": Type,
            "file_path": file_path,
            "raw_dataset": "",
            "raw_system_prompt": system_prompt,
            "raw_user_prompt": user_prompt,
            "category": "",
            "error": 0,
            "raw_predict": "",
            "think": "",
            "conclusion": "",
            "match": 0
        }

        times = 0
        while True:
            if self.is_large_20(file_path):
                result = self.infer_with_upload(
                    file_path,
                    file_type,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            else:
                result = self.infer_with_base64(
                    file_path,
                    file_type,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt
                )
            times += 1
            if result is None and times > 5:
                logger.error(f"文件{file_path}重复运行大于 5 次，不再重试")
                break
            elif result is None:
                logger.error(f"文件{file_path}正在重复运行{times}/5次。")
                time.sleep(5)

        think = Tools.extract_think_content(result)
        conclusion = Tools.extract_conclusion_content(result)
        matchs = Tools.label_vs(conclusion, Label)
        if "429 RESOURCE_EXHAUSTED" in str(result):
            logger.error(str(result))
            sys.exit(1)  # 非零退出码表示异常终止
        meta_dict["raw_predict"] = result
        meta_dict["think"] = think
        meta_dict["conclusion"] = conclusion
        meta_dict["match"] = matchs
        data_dict = None
        if result is None:
            logger.error(f"DP is: {result}")
            meta_dict["error"] = 1
            return None
        else:
            data_dict = {
                "messages": [
                    {
                        "content": f"<{Label}>{user_prompt}",
                        "role": "user"
                    },
                    {
                        "content": "",
                        "role": "assistant"
                    }
                    ],
                    f"{Type}s": [
                        file_path
                    ],
                "system": system_prompt,
                "rawInfer": meta_dict
            }
            return data_dict



    def infer_with_file_arvhive(self, file_path, **kwargs):
        uploaded_file = self.client.files.upload(file=file_path)
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text=kwargs["system_prompt"]),
            ],
        )
        while uploaded_file.state.name == "PROCESSING":
            print("processing video...")
            time.sleep(5)
            uploaded_file = self.client.files.get(name=uploaded_file.name)

        result = self.client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-01-21",
            contents=[uploaded_file, kwargs["user_prompt"]],
            config=generate_content_config
        )
        if result.text is None:
            logger.debug(f"DP is: {result}")
        return result.text

    def _search_raw_dataset(self, file_path):
        for i in self.raw_datasets:
            if i in file_path:
                return i
        return "FakeThread"

    def process_path(self,file_path):
        if not os.path.isabs(file_path):
            absolute_path = os.path.abspath(file_path)

            return absolute_path
        else:
            return file_path

    def is_image_file(self,filename):
        """
        判断一个文件是否是图片格式（根据文件后缀名）
        支持的图片格式包括：.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in image_extensions

    def _llm_infer(self, llm_engine, file_path, **kwargs):
        result = ""

        if llm_engine == "gemini":
            result = self.infer_with_file(file_path, system_prompt=kwargs["system_prompt"], user_prompt=kwargs["user_prompt"])
        return result

    def is_video_file(self,filename):
        """
        判断一个文件是否是图片格式（根据文件后缀名）
        支持的图片格式包括：.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp
        """
        image_extensions = {'.mp4'}
        ext = os.path.splitext(filename)[1].lower()
        return ext in image_extensions

    def infer_with_file_with_lock(self, file_path, share_list, lock):

        result = self.infer_with_file(file_path)
        if result is not None:
            with lock:
                share_list.append(result)
                if len(share_list) % 10 == 0:
                    logger.info(f"已经处理了{len(share_list)}.")
                    Tools.to_json(share_list, to_path)
        sys.exit(1)

    def pipeline(self, unzip_path, to_path, **kwargs):
        """
        流水线开始
        """
        if os.path.exists(to_path):
            to_data = Tools.load_json(to_path)
            to_data = [i for i in to_data if i["rawInfer"]["error"] != 1]
            to_data = deduplicate_by_id(to_data)
            uid_map = {i["id"]: i for i in to_data if i["rawInfer"]["error"] != 1}
            uids = uid_map.keys()
        else:
            uids = []
            to_data = []
            uid_map={}
        logger.info(f"{unzip_path}已经处理了{len(to_data)}")
        all_files = []
        for root, _, files in os.walk(unzip_path):
            for file in files:
                file_path = os.path.join(root, file)
                uid = Tools.generate_id(file_path)
                if os.path.splitext(file)[-1] in self.search_extensions:
                    if uid in uids:
                        continue
                    all_files.append(file_path)
        logger.info(f"还没有处理的文件为{len(all_files)}，即将开始处理。")

        ## 多进程运行
        manager = Manager()
        shared_list = manager.list(to_data)
        lock = manager.Lock()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_file = {executor.submit(self.infer_with_file_with_lock, file_path, shared_list, lock):file_path for file_path in all_files}

            for future in as_completed(future_to_file):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"处理 {future_to_file[future]} 出错: {e}")
                    sys.exit(1)
if __name__ == "__main__":
    file_root = "interpretability/image/train" ## modelscope的路径，保持不变
    to_root = "/content/drive/MyDrive/Project/AIGC-Detection" ## 保存 json 路径（相对路径）
    files = ['FakeClue.tar.gz', 'self_craw_part01.tar.gz', 'self_craw_part02.tar.gz', 'self_craw_part03.tar.gz', 'self_craw_part04.tar.gz', 'self_craw_part05.tar.gz', 'self_craw_part06.tar.gz', 'self_craw_part07.tar.gz', 'self_craw_part08.tar.gz', 'self_craw_part09.tar.gz', 'self_craw_part10.tar.gz', 'self_craw_part11.tar.gz', 'WildFake.tar.gz', 'self_craw_part12.tar.gz']
    
    
    files = [os.path.join(file_root,i) for i in files]

    for file_path in files:
        geminiPipeline=GeminiPipeline()
        file_path_name = os.path.basename(file_path).split(".")[0] # 不带后缀的文件路径

        unzip_path = os.path.join(file_root,file_path_name)
        to_path = os.path.join(to_root, file_root, file_path_name+".json")
        os.makedirs(os.path.join(to_root, file_root), exist_ok=True)

        if os.path.exists(file_path):
            pass
        else:
            logger.info(f"正在从 modelscope 下载文件：{file_path}")
            data_dir = snapshot_download('thejiangcj/FakeThread',local_dir=".",allow_patterns=file_path,repo_type="dataset")
        subprocess.run(['tar', '-zxvf', file_path, '-C', file_root], check=True)

        geminiPipeline.pipeline(
            unzip_path, # 原始图片路径
            to_path # 保存的jsonl文件路径
        )

        shutil.rmtree(unzip_path)
        os.remove(unzip_path)