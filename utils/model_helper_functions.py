from transformers import MBartForConditionalGeneration, MBartTokenizer
import torch

def get_model_from_checkpoint(checkpoint_path: str):
    return MBartForConditionalGeneration.from_pretrained(checkpoint_path)

def prune_model_embeddings(model: MBartForConditionalGeneration, tokenizer: MBartTokenizer):
    vocab_ids = sorted(list(tokenizer.get_vocab().values()))
    new_model = model
    changed_weights = {'final_logits_bias' : model.final_logits_bias[:, vocab_ids],
                        'model.shared.weight' : model.model.shared.weight.detach().numpy()[vocab_ids,:], 
                        'model.encoder.embed_tokens.weight' : model.model.encoder.embed_tokens.weight.detach().numpy()[vocab_ids, :], 
                        'model.decoder.embed_tokens.weight' : model.model.decoder.embed_tokens.weight.detach().numpy()[vocab_ids, :], 
                        'lm_head.weight' : model.lm_head.weight.detach().numpy()[vocab_ids, :]}

    for param in model.state_dict().keys():
        if param in changed_weights.keys():
            continue
        new_model.state_dict()[param].copy_(model.state_dict()[param])

    
    # change final_logits
    new_model.final_logits_bias = changed_weights['final_logits_bias']
    # change lm head
    new_model.lm_head.weight.data = torch.Tensor(changed_weights['lm_head.weight'])
    # change input embeddings
    embeddingMatrix = torch.nn.Embedding.from_pretrained(torch.Tensor(changed_weights['model.shared.weight']), 
                                                        freeze=False, 
                                                        padding_idx = tokenizer.pad_token_id)
    new_model.set_input_embeddings(embeddingMatrix)
    new_model.tie_weights()
    
    return new_model