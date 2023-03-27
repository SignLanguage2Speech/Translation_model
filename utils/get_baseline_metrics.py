import numpy as np
import evaluate


def get_baseline_metrics(test_dataset):
    preds = np.array([test_dataset.get_gloss(i) for i in range(len(test_dataset))])
    targets = np.array([test_dataset.get_label(i) for i in range(len(test_dataset))])

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