from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_cosine_with_hard_restarts_schedule_with_warmup
from utils.compute_metrics import ComputeMetrics
import torch

def get_trainer(model, tokenizer, CFG, train_dataset, eval_dataset):
    compute_metrics = ComputeMetrics(tokenizer)
    return Seq2SeqTrainer(
        model = model, 
        args = Seq2SeqTrainingArguments(
            CFG.save_directory,
            num_train_epochs = CFG.num_train_epochs,
            evaluation_strategy = CFG.evaluation_strategy,
            per_device_train_batch_size = CFG.per_device_train_batch_size,
            per_device_eval_batch_size = CFG.per_device_eval_batch_size,
            learning_rate = CFG.learning_rate,
            weight_decay = CFG.weight_decay,
            save_strategy = CFG.save_strategy,
            save_total_limit = CFG.save_total_limit,
            seed = CFG.seed,
            predict_with_generate = True,
            label_smoothing_factor = CFG.label_smoothing_factor,
            optim = "adamw_torch",
            lr_scheduler_type = "cosine",
            logging_strategy = CFG.logging_strategy,
            warmup_ratio = CFG.warmup_ratio,
            ),
        train_dataset = train_dataset, 
        eval_dataset = eval_dataset,
        tokenizer = tokenizer,
        data_collator = DataCollatorForSeq2Seq(
            tokenizer = tokenizer, 
            model = model,
            return_tensors = "pt"),
        compute_metrics = compute_metrics.compute,
        )