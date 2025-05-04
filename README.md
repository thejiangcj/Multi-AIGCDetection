
## 启动脚本

运行 Gemini 蒸馏数据：`python ./Test/GeminiPipeline.py`，注意修改内部的：

- prompt_config：两个 Prompt 的相对路径
- api = "": Gemoni 的 API
- gemini.pipeline()中传输的各个参数，包括保存 jsonl 的结果文件地址（重要），模型名不用改变。

## 数据分布

- interpretability
  - video
    - test
      - 数据集名 XX
      - 数据集名 XX
    - train
      - 数据集名 XX
      - 数据集名 XX
  - image
    - test
      - 数据集名 XX
      - 数据集名 XX
    - train
      - 数据集名 XX
      - 数据集名 XX

数量分布：

- video train:

| 数据集名称        | 数量   | 格式 | 标签 |
|------------------|--------|------|------|
| Kinetics-400     | 25000  | mp4  | real |
| SEINE            | 10000  | mp4  | fake |
| pika             | 4123   | mp4  | fake |
| OpenSora         | 10000  | mp4  | fake |
| ZeroScope        | 10000  | mp4  | fake |
| Latte            | 10000  | mp4  | fake |
| DynamicCrafter   | 10000  | mp4  | fake |
| Youku_1M_10s     | 30000  | mp4  | real |

- video test:

| 数据集名称          | 数量  | 标签 |
|----------------------|-------|------|
| MSR-VTT              | 2500  | real |
| Kinetics-400-val     | 2500  | real |
| Crafter              | 300   | fake |
| HotShot              | 300   | fake |
| Show_1               | 300   | fake |
| MorphStudio          | 300   | fake |
| Sora                 | 56    | fake |
| Gen2                 | 300   | fake |
| WildScrape           | 300   | fake |
| Lavie                | 300   | fake |
| ModelScope           | 300   | fake |
| MoonValley           | 300   | fake |
| self_craw            | 1580  | fake |
| loki                 | 767   | fake |

- image train:

| 数据集名称     | 数量   | 标签 |
|----------------|--------|------|
| FakeClue       | 35947  | real |
| FakeClue       | 20000  | fake |
| self_craw      | 30000  | fake |
| WildFake       | 26250  | real |

- image test:

| 数据集名称         | 数量 | 标签 |
|--------------------|------|------|
| FakeClue           | 1808 | real |
| FakeClue           | 1000 | fake |
| loki               | 3201 | fake |
| loki               | 889  | fake |
| loki               | 2    | fake |
| MMFakeBench_test   | 3000 | real |

## 数据生成（SFT）

./Scripts/ClassificationSFTDataScripts.py

## 模型推理

模型推理需要用 sglang 外挂模型，包括微调的模型，外挂的模型在 Config/LLMsConfig.json 中配置。

```python
./LLMs/OpenaiLLMs.py
```

查看`./LLMs/OpenaiLLMs.py`文件下的测试数据，其中主要任务为 pipeline 方法，参数为：

- data_path: 包括分类好的数据集文件夹（fake/real）文件夹下
- to_path: 保存的 json 文件路径
- task: cls 或者 plausibility

## 评估

### 自动化评估

```python
./Evaluate/JsonEvaluate.py
```