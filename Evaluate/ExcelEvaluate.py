from loguru import logger
import evaluate
import pandas as pd

class ExcelEvaluate:
    def __init__(self):
        # 初始化评估指标
        self.accuracy_metric = evaluate.load("accuracy")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")
        self.f1_metric = evaluate.load("f1")
    def evaluate(self, excel_path):
        df = pd.read_excel(excel_path)
        required_columns = {'label', 'predict'}
        # 验证标签值是否为Fake或Real
        valid_values = {'Fake', 'Real'}
        for col in ['label', 'predict']:
            invalid_entries = df[~df[col].isin(valid_values)]
        
        # 提取真实标签和预测标签
        y_true = df['label']
        y_pred = df['predict']
        y_true = [1 if i == "Fake" else 0 for i in y_true]
        y_pred = [1 if i == "Fake" else 0 for i in y_pred]
        # 计算 Accuracy
        accuracy = self.accuracy_metric.compute(
            references=y_true, 
            predictions=y_pred
        )["accuracy"]

        # 计算宏平均（Macro）指标
        precision = self.precision_metric.compute(
            references=y_true,
            predictions=y_pred,
            average="macro",
            # labels=["Fake", "Real"]
        )["precision"]

        recall = self.recall_metric.compute(
            references=y_true,
            predictions=y_pred,
            average="macro",
            # labels=["Fake", "Real"]
        )["recall"]

        f1 = self.f1_metric.compute(
            references=y_true,
            predictions=y_pred,
            average="macro",
            # labels=["Fake", "Real"]
        )["f1"]

        
        # 打印结果（按百分比显示）
        logger.info(f"Accuracy: {accuracy * 100:.2f}%")
        logger.info(f"Macro Precision: {precision * 100:.2f}%")
        logger.info(f"Macro Recall: {recall * 100:.2f}%")
        logger.info(f"Macro F1: {f1 * 100:.2f}%")

excelEvaluate = ExcelEvaluate()

if __name__=="__main__":
    excelEvaluate.evaluate("/Users/arnodorian/Desktop/aigc_detection/Data/test100Data/cot_excel.xlsx")