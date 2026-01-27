class F1Calculator:
    def __init__(self, save_file=None):
        """
        初始化 F1 计算器
        :param save_file: 可选，保存预测和参考结果的文件路径
        """
        self.save_file = save_file
        self.tp_list = []
        self.fp_list = []
        self.fn_list = []

    def update(self, predictions, references):
        """
        更新预测和参考数据，同时计算 TP, FP, FN
        :param predictions: list of str, 预测结果
        :param references: list of str, 金标准结果
        """
        processed_predictions = []
        processed_references = []

        for pred, gold in zip(predictions, references):
            processed_predictions.append("" if "there is no entity" in pred else pred)
            processed_references.append("" if "there is no entity" in gold else gold)

        # 拆分字符串，按 '##' 分割，并去掉空字符串
        split_predictions = [[item.strip() for item in entry.split('##') if item.strip()] 
                             for entry in processed_predictions]
        split_references = [[item.strip() for item in entry.split('##') if item.strip()] 
                            for entry in processed_references]

        predictions_set = [list(set(sublist)) for sublist in split_predictions]
        references_set = [list(set(sublist)) for sublist in split_references]

        # 保存到文件（可选）
        if self.save_file:
            with open(self.save_file, 'a', encoding='utf-8') as f:
                for pred, ref in zip(predictions_set, references_set):
                    f.write(f'prediction: {pred}\n')
                    f.write(f'reference: {ref}\n')
                    f.write('---\n')

        # 计算 TP, FP, FN 并累积
        for g, p in zip(references_set, predictions_set):
            set_g, set_p = set(g), set(p)
            self.tp_list.append(len(set_g & set_p))
            self.fp_list.append(len(set_p - set_g))
            self.fn_list.append(len(set_g - set_p))

    def compute_f1(self):
        """
        计算当前累积的 TP, FP, FN 对应的 Precision, Recall, F1
        :return: true_positives, false_positives, false_negatives, precision, recall, f1
        """
        true_positives = sum(self.tp_list)
        false_positives = sum(self.fp_list)
        false_negatives = sum(self.fn_list)

        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return true_positives, false_positives, false_negatives, f1
