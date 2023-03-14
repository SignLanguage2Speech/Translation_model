from transformers import MBartTokenizer
from pathlib import Path
from sentencepiece import sentencepiece_model_pb2 as spm


def get_new_vocab(tokenizer, data: list[str]):
    new_vocab = set()
    new_vocab_ids = None

    # converts sentences to words
    for sent in data:
        tokenized = tokenizer(sent, add_special_tokens=False, return_attention_mask=True)
        for e in tokenized['input_ids']:
            new_vocab.add(tokenizer.convert_ids_to_tokens(e))
    
    # add special tokens to the new vocab
    new_vocab.update(tokenizer.all_special_tokens)
    new_vocab.update(tokenizer.additional_special_tokens)

    # add input ids from new vocab
    new_vocab_ids = sorted(tokenizer.convert_tokens_to_ids(new_vocab))
    return new_vocab, new_vocab_ids



def get_new_tokenizer(tokenizer: MBartTokenizer, tokenizer_path: str, data: list[str]):

    new_vocab, _ = get_new_vocab(tokenizer, data)
    
    # save old tokenizer configs
    tokenizer.save_pretrained(tokenizer_path)

    # load vocab into spm
    m = spm.ModelProto()
    m.ParseFromString(open('{tokenizer_path}/sentencepiece.bpe.model', 'rb').read())
    N = len(m.pieces)
    # check words that are in the new vocab
    for i in range(N):
        e = m.pieces[i]
        if e.piece in new_vocab:
            m.pieces.append(e)

    # delete oov words
    del m.pieces[:N]

    # backup old model
    Path('{tokenizer_path}/sentencepiece.bpe.model').rename('{tokenizer_path}/old_sentencepiece.bpe.model')
    # save new model
    with open('{tokenizer_path}/sentencepiece.bpe.model', 'wb') as f:
        f.write(m.SerializeToString())

    new_tokenizer = load_tokinizer(tokenizer_path)
    return new_tokenizer


def load_tokinizer(tokenizer_path: str):
    return MBartTokenizer.from_pretrained(tokenizer_path)

