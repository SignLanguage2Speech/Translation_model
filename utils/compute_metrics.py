import numpy as np
import evaluate

class ComputeMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def compute(self, EvalPrediction):
        preds = np.array(self.tokenizer.batch_decode(EvalPrediction.predictions, skip_special_tokens=True))
        np_label_ids = np.array(EvalPrediction.label_ids)
        np_label_ids[np_label_ids == -100] = 1
        targets = self.tokenizer.batch_decode(np_label_ids, skip_special_tokens=True)
        metrics = {}
        bleu = evaluate.load("bleu")
        rouge = evaluate.load('rouge')
        for n_gram in [1,2,3,4]:
            metrics[f"BLEU_{n_gram}"] = bleu.compute(
                                            predictions = preds, 
                                            references = [[target] for target in targets],
                                            max_order = n_gram).get("bleu")
        metrics["ROUGE"] = rouge.compute(
                                        predictions = preds, 
                                        references = [[target] for target in targets]).get("rouge1")
        return metrics