import json

# 原文件路径
json_path = "/raid/share/jiangchangjiang/Multi-AIGCDetection/dataset/classification/image/train/WildFake-train-llava.json"

# 加载 JSON 数据
with open(json_path, "r") as f:
    data = json.load(f)

# 前缀路径
prefix = "classification/image/train/"

# 修改每个 dict 的 image 字段
for item in data:
    if "image" in item and not item["image"].startswith(prefix):
        item["image"] = prefix + item["image"]

# 保存修改后的数据（可选：覆盖原文件或另存为新文件）
with open(json_path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("修改完成 ✅")