import os
import sys

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from LLMs.OpenaiLLMs import openaiPipeline

if __name__ == "__main__":
    file_path = "/raid/share/jiangchangjiang/aws/dataset/demamba/train_I2VGEN_XL/I2VGEN_XL_001.mp4"
    print(openaiPipeline.infer_with_cls(file_path))