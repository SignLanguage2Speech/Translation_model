from utils.tokenizer_helper_functions import load_tokinizer, get_new_tokenizer
from utils.model_helper_functions import get_model_from_checkpoint, prune_model_embeddings


class CFG:
    def __init__(self):
        self.tokenizer_path = "mBART_tokenizer"
        self.new_tokenizer = True
        self.model_checkpoint = "facebook/mbart-large-cc25"
        self.prune_embeddings = True
        

cfg = CFG()

### Tokenizer ###
tokenizer = load_tokinizer(cfg.tokenizer_path)
if cfg.new_tokenizer:
    tokenizer = get_new_tokenizer(tokenizer, cfg.tokenizer_path, ...)

### Model ###
model = get_model_from_checkpoint("facebook/mbart-large-cc25")
if cfg.prune_embeddings:
    model = prune_model_embeddings(model, tokenizer)

