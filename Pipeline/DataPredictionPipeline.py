
from utils.tools import Tools
from utils import *
from LLMs.GeminiLLMs import gemini

import os
import json
import random

class DataPredictionnPipeline:
    def __init__(self):
        self.search_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', 'mp4']
        self.labels = ["real", "fake"]
        self.raw_datasets = ["FakeClue", "GenVideo"]
        self.category = {
            "doc": "doc",
            "ff++": "DeepFake",
            "genimage": "genimage",
            "satellite": "satellite",
            "chameleon": "chameleon"
        }

    def generate(self, to_path, raw_data_path="/Users/arnodjiang/Desktop/AIGCDetection/data/fakecle/fakethread", llm_engine="gemini", **kwargs):
        assert to_path.endswith(".jsonl")

        if os.path.exists(to_path):
            to_data = Tools.load_jsonl(to_path)
            uids = [i["id"] for i in to_data]
        else:
            uids = []
            to_data = []

        for root, dirs, files in os.walk(raw_data_path):
            for file in files:
                file_path = os.path.join(root, file)
                if os.path.splitext(file)[-1] in self.search_extensions:
                    rel_path = os.path.relpath(file_path, raw_data_path)
                    rwa_dataset = self._search_raw_dataset(file_path)
                    category = self._search_category(file_path)
                    uid = Tools.generate_id(rel_path)
                    if uid in uids:
                        continue
                    to_dict = {
                        "id": uid,
                        "label": "real" if "real" in file_path else "fake",
                        "file_path": rel_path,
                        "raw_dataset": rwa_dataset,
                        "category": category,
                        "raw_predict": "",
                        "think": "",
                        "conclusion": "",
                        "match": 0
                    }
                    result = self._llm_infer(llm_engine=llm_engine, file_path=file_path)
                    to_dict["raw_predict"] = result
                    think = Tools.extract_think_content(result)
                    conclusion = Tools.extract_conclusion_content(result)
                    match = Tools.label_vs(conclusion, to_dict["label"])

                    to_dict["think"] = think
                    to_dict["conclusion"] = conclusion
                    to_dict["match"] = match

                    to_data.append(to_dict)
                    Tools.to_file(to_data, to_path)




    def _search_raw_dataset(self, file_path):
        for i in self.raw_datasets:
            if i in file_path:
                return i
        return "FakeThread"

    def _search_category(self, file_path):
        for i in self.category.keys():
            if i in file_path:
                return self.category[i]
        return "Other"
    def _llm_infer(self, llm_engine, file_path, **kwargs):
        result = ""
        if llm_engine == "gemini":
            result = gemini.infer_with_file(file_path)
        return result
    
    def _validate_response_pipeline(self, response_text):
        """多条件验证管道"""
        conditions = [
            Tools.has_think_tags,
            Tools.has_conclusion_tags,
            Tools.conclusion_has_single_word
        ]

        for condition in conditions:
            if not condition(response_text):
                return False
        return True

dataPredictionPipeline = DataPredictionnPipeline()