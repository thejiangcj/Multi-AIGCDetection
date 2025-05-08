import os
import shutil
import random
from multiprocessing import Pool, cpu_count
from functools import partial

SRC_ROOT = "/raid/share/jiangchangjiang/Multi-AIGCDetection/modelscope_data/classifcation/video/train/classification/video/train/Youku_1M_10s"
DST_ROOT = "/raid/share/jiangchangjiang/Multi-AIGCDetection/tmp_58"
SAMPLE_NUM = 30000

def find_mp4_files(root_dir):
    mp4_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mp4'):
                full_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(full_path, root_dir)
                mp4_files.append((full_path, rel_path))
    return mp4_files

def copy_file(src_dst_tuple):
    src_path, rel_path = src_dst_tuple
    dst_path = os.path.join(DST_ROOT, rel_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        shutil.copy2(src_path, dst_path)
    except Exception as e:
        print(f"Failed to copy {src_path}: {e}")

def main():
    print("Scanning files...")
    all_files = find_mp4_files(SRC_ROOT)
    print(f"Total .mp4 files found: {len(all_files)}")

    if len(all_files) < SAMPLE_NUM:
        raise ValueError(f"Only found {len(all_files)} files, which is fewer than the required {SAMPLE_NUM}.")

    sampled_files = random.sample(all_files, SAMPLE_NUM)
    print(f"Copying {SAMPLE_NUM} files...")

    with Pool(processes=cpu_count()) as pool:
        pool.map(copy_file, sampled_files)

    print("Copy complete.")

if __name__ == "__main__":
    main()