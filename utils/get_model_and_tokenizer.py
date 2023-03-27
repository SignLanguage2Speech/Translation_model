from transformers import MBartForConditionalGeneration, MBartTokenizer, GenerationConfig

def get_model_and_tokenizer(CFG):
    tokenizer: MBartTokenizer = MBartTokenizer.from_pretrained(CFG.tokenizer_path,  src_lang=CFG.scr_lang, tgt_lang=CFG.scr_lang)
    model: MBartForConditionalGeneration = MBartForConditionalGeneration.from_pretrained(CFG.model_checkpoint)
    model.config.early_stopping = CFG.early_stopping
    model.config.num_beams = CFG.num_beams
    model.config.repetition_penalty = CFG.repetition_penalty
    model.config.length_penalty = CFG.length_penalty
    model.config.decoder_start_token_id = tokenizer.lang_code_to_id[CFG.scr_lang]
    model.config.vocab_size = len(tokenizer.get_vocab())
    model.config.dropout = CFG.dropout
    model.config.classifier_dropout = CFG.classifier_dropout

    ### DONT USE!!! ###
    # model.config.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    # model.config.eos_token_id = tokenizer.convert_tokens_to_ids("<s/>")
    # model.config.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
    # model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
    
    return model, tokenizer