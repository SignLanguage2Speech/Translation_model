from transformers import MBartForConditionalGeneration, MBartTokenizer, GenerationConfig

def get_model_and_tokenizer(CFG):
    tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained(CFG.tokenizer_path,  src_lang=CFG.trg_lang, tgt_lang=CFG.trg_lang)
    model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(CFG.model_checkpoint)
    model.generation_config.repetition_penalty = CFG.repetition_penalty
    model.generation_config.early_stopping = CFG.early_stopping
    model.generation_config.num_beams = CFG.num_beams
    model.generation_config.length_penalty = CFG.length_penalty
    model.generation_config.decoder_start_token_id = tokenizer.lang_code_to_id[CFG.trg_lang]
    model.config.vocab_size = len(tokenizer.get_vocab())
    model.config.dropout = CFG.dropout
    model.config.attention_dropout = CFG.attention_dropout
    model.config.classifier_dropout = CFG.classifier_dropout
    
    return model, tokenizer