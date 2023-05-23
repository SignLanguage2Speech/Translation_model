import numpy as np
import evaluate
import torch
import torch.nn as nn


def compute_metrics(model,tokenizer, dataloaderTest, loss_preds_fc, epoch, CFG):

    metrics = {}
    bleu = evaluate.load("bleu")
    metrics[f"BLEU_1"] = 0
    metrics[f"BLEU_2"] = 0
    metrics[f"BLEU_3"] = 0
    metrics[f"BLEU_4"] = 0
    metrics[f"LOSS"] = 0
    rouge = evaluate.load('rouge')
    metrics["ROUGE"] = 0

    preds = []
    targets = []

    example_index = np.random.randint(len(dataloaderTest))
    for j, (inputs, labels) in enumerate(dataloaderTest):

        out = model(**inputs.to(CFG.device)).logits

        loss = loss_preds_fc(
            nn.functional.log_softmax(out,dim=-1), 
            inputs["labels"]) / inputs["input_ids"].size(0)

        metrics[f"LOSS"] += loss.detach().cpu().numpy()

        output_dict = model.generate(
            input_ids = inputs["input_ids"], 
            attention_mask = inputs["attention_mask"], 
            decoder_start_token_id = tokenizer.lang_code_to_id["de_DE"],
            num_beams=CFG.num_beams, 
            length_penalty=CFG.length_penalty, 
            max_length=100, 
            repetition_penalty = CFG.repetition_penalty,
            early_stopping = CFG.early_stopping,
            do_sample=False,
            return_dict_in_generate=True)
        raw_preds = tokenizer.batch_decode(output_dict['sequences'], skip_special_tokens=True)

        for i in range(len(raw_preds)):
            targets.append(labels[i])
            if j == example_index:
                metrics[f"EXAMPLE"] = f"pred: {raw_preds[i]}, target: {labels[i]}"
            if raw_preds[i]:
                preds.append(raw_preds[i])
            else:
                preds.append("@")

    metrics[f"BLEU_1"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 1).get("bleu")
    metrics[f"BLEU_2"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 2).get("bleu")
    metrics[f"BLEU_3"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 3).get("bleu")
    metrics[f"BLEU_4"] += bleu.compute(predictions = preds, references = [[target] for target in targets], max_order = 4).get("bleu")
    metrics[f"ROUGE"] += rouge.compute(predictions = preds, references = [[target] for target in targets]).get("rouge1")
    metrics[f"LOSS"] /= len(dataloaderTest)
    
    return metrics