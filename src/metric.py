class F1Calculator:
    def __init__(self, save_file=None):

        self.save_file = save_file
        self.sum_tp = 0
        self.sum_fp = 0
        self.sum_fn = 0

    def update(self, predictions, references):##更新TP,FP,FN
        processed_predictions = []
        processed_references = []

        for pred, gold in zip(predictions, references):##遍历batch中的每一条数据
            processed_predictions.append("" if "there is no" in pred else pred)
            processed_references.append("" if "there is no" in gold else gold)

        # 拆分字符串，按 '##' 分割，并去掉空字符串
        split_predictions = [[item.strip() for item in entry.split('##') if item.strip()] 
                             for entry in processed_predictions]
        split_references = [[item.strip() for item in entry.split('##') if item.strip()] 
                            for entry in processed_references]

        predictions_set = [list(set(sublist)) for sublist in split_predictions]
        references_set = [list(set(sublist)) for sublist in split_references]

        # 保存到文件
        if self.save_file:
            with open(self.save_file, 'a', encoding='utf-8') as f:
                for pred, ref in zip(predictions_set, references_set):
                    f.write(f'prediction: {pred}\n')
                    f.write(f'reference: {ref}\n')
                    f.write('---\n')

        # 计算 TP, FP, FN 并累积
        for g, p in zip(references_set, predictions_set):
            g_set = set(g)
            p_set = set(p)
            self.sum_tp += len(g_set & p_set)
            self.sum_fp += len(p_set - g_set)
            self.sum_fn += len(g_set - p_set)

    def compute_f1(self):
        """
        计算当前累积的 TP, FP, FN 对应的 Precision, Recall, F1
        :return: true_positives, false_positives, false_negatives, f1
        """
        true_positives = self.sum_tp
        false_positives = self.sum_fp
        false_negatives = self.sum_fn

        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return true_positives, false_positives, false_negatives, f1



class F1Calculator_for_code:
    def __init__(self, save_file=None):

        self.save_file = save_file
        self.sum_tp = 0
        self.sum_fp = 0
        self.sum_fn = 0
    
    def update(self, predictions, references):##更新TP,FP,FN
        processed_predictions = []
        processed_references = []

        for gold in references:
            # processed_predictions.append("" if "there is no " in pred else pred)
            processed_references.append("" if "there is no" in gold else gold)
        ##去除最后的"}"符号
        predictions1 = []
        if isinstance(predictions, list) and len(predictions) > 0 and isinstance(predictions[0], list):
            
            for row in predictions:
                # 每个 row 可能是一个 list，例如 ["GENE(X)", "}"]
                new_row = [s.replace('\n}', '').strip() for s in row if isinstance(s, str)]
                predictions1.append(new_row)
        else:        
            predictions1 = [p.replace('\n}', '') for p in predictions]
        for sublist in predictions1:
            # 1. 保证 sublist 一定是列表，如果是字符串 → 单独包装成列表
            if isinstance(sublist, str):
                sublist = [sublist]
            cleaned_sublist = [line.strip() for item in sublist for line in item.split("\n") if line.strip()]##将内容按照换行符号分开
            

            # 2. 如果列表中任何项包含 "there is no"
            if any("there is no" in item for item in cleaned_sublist):
                processed_predictions.append([])  # 置空
            else:
                processed_predictions.append(cleaned_sublist)  # 保留原列表
        split_references = [[item.strip() for item in entry.split('\n') if item.strip() and item.strip() != '}'] 
                            for entry  in processed_references]
        split_predictions1 = processed_predictions
        predictions_set = [list(set(sublist)) for sublist in split_predictions1]
        references_set = [list(set(sublist)) for sublist in split_references]

        # 保存到文件
        if self.save_file:
            with open(self.save_file, 'a', encoding='utf-8') as f:
                for pred, ref in zip(predictions_set, references_set):
                    f.write(f'prediction: {pred}\n')
                    f.write(f'reference: {ref}\n')
                    f.write('---\n')

        # 计算 TP, FP, FN 并累积
        for g, p in zip(references_set, predictions_set):
            g_set = set(g)
            p_set = set(p)
            self.sum_tp += len(g_set & p_set)
            self.sum_fp += len(p_set - g_set)
            self.sum_fn += len(g_set - p_set)
        
    def compute_f1(self):
        """
        计算当前累积的 TP, FP, FN 对应的 Precision, Recall, F1
        :return: true_positives, false_positives, false_negatives, f1
        """
        true_positives = self.sum_tp
        false_positives = self.sum_fp
        false_negatives = self.sum_fn

        precision = true_positives / (true_positives + false_positives) \
            if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) \
            if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return true_positives, false_positives, false_negatives, f1