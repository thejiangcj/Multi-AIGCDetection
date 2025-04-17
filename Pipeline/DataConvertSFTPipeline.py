
class DataConvertSFTPipeline:
    def __init__(self):
        pass
    def pipeline(self, pipeline_path):
        jsonl_path = os.path.join(pipeline_path, "data.jsonl")

dataConvertSFTPipeline = DataConvertSFTPipeline()