from utils import tools
from LLMs.GeminiLLMs import gemini

import os
import json
import random

class DistilPipeline:
    def __init__(self):
        self.llm = gemini()
        self.llm.load_model()
    
    def pipeline(self, )

distilPipeline = DistilPipeline(tools.Tools.load_json("./Config/LLMsConfig.json"))