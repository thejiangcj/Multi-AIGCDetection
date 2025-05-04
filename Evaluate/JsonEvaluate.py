from loguru import logger
import evaluate
import json

class JsonEvaluate:
    def __init__(self):
        # 初始化评估指标
        self.accuracy_metric = evaluate.load("accuracy")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")
        self.f1_metric = evaluate.load("f1")

    def evaluate(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        required_keys = {'label', 'predict'}
        valid_values = {'fake', 'real'}

        for entry in data:
            if not required_keys.issubset(entry.keys()):
                raise ValueError(f"Each entry must contain {required_keys}")
            if entry['label'] not in valid_values or entry['predict'] not in valid_values:
                raise ValueError(f"Invalid label or prediction value: {entry}")

        # 将标签和预测值转换为二进制值  
        y_true, y_pred = [], []

        for item in data:
            if item['label'] == item['predict']:
                y_true.append(1)
                y_pred.append(1)
            else:
                y_true.append(0)
                y_pred.append(0)

        accuracy = self.accuracy_metric.compute(
            references=y_true,
            predictions=y_pred
        )["accuracy"]

        precision = self.precision_metric.compute(
            references=y_true,
            predictions=y_pred,
            average="macro"
        )["precision"]

        recall = self.recall_metric.compute(
            references=y_true,
            predictions=y_pred,
            average="macro"
        )["recall"]

        f1 = self.f1_metric.compute(
            references=y_true,
            predictions=y_pred,
            average="macro"
        )["f1"]

        # 打印结果（按百分比显示）
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"Macro Precision: {precision * 100:.2f}%")
        logger.info(f"Macro Recall: {recall * 100:.2f}%")
        logger.info(f"Macro F1: {f1 * 100:.2f}%")

jsonEvaluate = JsonEvaluate()

if __name__ == "__main__":
    json_path = "/path/to/your/eval.json"
    jsonEvaluate.evaluate(json_path)