"""
Date: 20250402090601
Description: This script is a project launch interface that starts different tasks depending on the parameter configuration.
Launch scirpts:
  - python main.py
"""

import time
import os
import argparse

from Pipeline.DataGenerationPipeline import dataGenerationPipeline

_task_map = {
    "DataGeneration": dataGenerationPipeline.pipeline, # 输入原始数据路径，在目标目录生成 jsonl
}

def init_args():
    parser = argparse.ArgumentParser(description='项目启动接口', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--task', '-t', help='任务参数，包括: 数据生成DataGeneration，测试 Test')
    parser.add_argument('--pipeline', '-p', help='流水文件路径')
    args = parser.parse_args()
    return args

def main(args):
   if args.task not in _task_map:
      raise ValueError(f"不支持的任务: {args.task}")

   _task_map[args.task](args.pipeline)
   

if __name__ == "__main__":
   args = init_args()
   main(args)
  