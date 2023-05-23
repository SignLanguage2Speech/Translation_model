import torch
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import time
import numpy as np
import random
from utils.loss import CustomLoss
from utils.metrics import compute_metrics
from utils.save_checkpoint import save_checkpoint


def train(model, tokenizer, dataloaderTrain, dataloaderVal, CFG):

    loss_preds_fc = CustomLoss(
        ignore_index = tokenizer.pad_token_id, 
        label_smoothing = CFG.label_smoothing_factor)
    ctc_loss_fc = torch.nn.CTCLoss(
        blank=0, 
        zero_infinity=True, 
        reduction='sum').to(CFG.device)
    optimizer = optim.AdamW(
        params = model.parameters(), 
        lr=CFG.learning_rate,
        betas = CFG.betas,
        weight_decay = CFG.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max = CFG.num_train_epochs)

    epoch_losses = {}
    losses = {}
    epoch_metrics = {}
    epoch_times = {}


    if CFG.verbose:
        print("\n" + "-"*20 + "Starting Training" + "-"*20)
    
    for epoch in range(CFG.num_train_epochs):
        losses[epoch] = []

        for i, (inputs, labels) in enumerate(dataloaderTrain):

            out = model(**inputs.to(CFG.device)).logits

            loss = loss_preds_fc(
                nn.functional.log_softmax(out,dim=-1), 
                inputs["labels"]) / inputs["input_ids"].size(0)

            model.zero_grad()
            loss.backward()
            optimizer.step()
            losses[epoch].append(loss.detach().cpu().numpy())

            if CFG.verbose_batches and i % 1 == 0:
                print(f"{i}/{len(dataloaderTrain)}", end="\r", flush=True)
        
        with torch.no_grad():
            model.eval()
            epoch_losses[epoch] = sum(losses[epoch])/len(dataloaderTrain)
            epoch_metrics[epoch] = compute_metrics(model, tokenizer, dataloaderVal, loss_preds_fc, epoch, CFG)
            model.train()
        
        if CFG.verbose:
            print("\n" + "-"*50)
            print(f"EPOCH: {epoch}")
            print(f"AVG. LOSS: {epoch_losses[epoch]}")
            print(f"EPOCH METRICS: {epoch_metrics[epoch]}")
            print("-"*50)

        scheduler.step()

        ## save model ### 
        if CFG.save_state and epoch % 2 == 0:
            save_path = CFG.save_directory +  "Gloss2Text_Epoch" + str(epoch+1) + "_loss_" + str(epoch_losses[epoch]) +  "_B4_" + str(epoch_metrics[epoch]["BLEU_4"])
            save_checkpoint(save_path, model, optimizer, scheduler, epoch, epoch_losses, epoch_metrics[epoch]["BLEU_4"])

    return losses, epoch_losses, epoch_metrics, epoch_times