"""
Date: 20250402090601
Description: This script is a project launch interface that starts different tasks depending on the parameter configuration.
Launch scirpts:
  - python3.11 main.py -t DistilPipeline -tp /Users/arnodjiang/Desktop/AIGCDetection/data/fakecle/fakethread/image/gemini.jsonl -rdp /Users/arnodjiang/Desktop/AIGCDetection/data/fakecle/fakethread -le gemini
"""

import time
import os
import argparse

from Pipeline.DistilPipeline import distilPipeline



_task_map = {
   "DistilPipeline": distilPipeline
}

def init_args():
    parser = argparse.ArgumentParser(description='项目启动接口', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--task', '-t', help='任务参数，包括: 数据生成DataGeneration，测试 Test')
    parser.add_argument('--to_path', '-tp', help='流水文件路径')
    parser.add_argument('--raw_data_path', '-rdp', help='流水文件路径')
    parser.add_argument('--llm_engine', '-le', help='大模型，Gemini')
    args = parser.parse_args()
    return args

def main(args):
   if args.task not in _task_map:
      raise ValueError(f"不支持的任务: {args.task}")

   _task_map[args.task](to_path=args.to_path, raw_data_path=args.raw_data_path, llm_engine=args.llm_engine)
   

if __name__ == "__main__":
   args = init_args()
   main(args)
  