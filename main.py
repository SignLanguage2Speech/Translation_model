import torch
from utils.PhoenixDatasetForMBART import PhoenixDatasetForMBART, collator
from utils.get_trainer import get_trainer
from utils.get_model_and_tokenizer import get_model_and_tokenizer
from utils.model_pruning import reduce_to_vocab
from utils.get_baseline_metrics import get_baseline_metrics
from torch.utils.data import DataLoader
from utils.trainer import train

class cfg:
  def __init__(self):
    ### type of run ###
    self.should_train = True
    self.should_eval = True
    self.should_test = True
    self.should_pre_eval = False
    self.save_state = False
    self.model_checkpoint =  "facebook/mbart-large-cc25" #'/work3/s200925/mBART/checkpoints/checkpoint-70960'#"facebook/mbart-large-cc25" #'/work3/s200925/mBART/checkpoints/checkpoint-70960' ###  "facebook/mbart-large-cc25" ### '/work3/s200925/mBART/checkpoints_hpc/checkpoint-26640' ### 
    self.tokenizer_path = "facebook/mbart-large-cc25" #'/work3/s200925/mBART/checkpoints/checkpoint-70960'# "facebook/mbart-large-cc25" #'/work3/s200925/mBART/checkpoints/checkpoint-70960'

    ### paths ###
    self.train_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.train.corpus.csv'
    self.test_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.test.corpus.csv'
    self.eval_path = '/work3/s204138/bach-data/PHOENIX/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv'
    self.vocab_data = 'german_data/autocomplete.txt'

    ### device ###
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ### model params ###
    self.scr_lang = "gl_DE"
    self.trg_lang = "de_DE"
    self.early_stopping = True
    self.num_beams = 4
    self.repetition_penalty = 1.0
    self.length_penalty = 1.0

    ### train params ###
    self.verbose_batches = True
    self.verbose = True
    self.num_train_epochs = 80
    self.per_device_train_batch_size = 8
    self.per_device_eval_batch_size = 1
    self.num_workers = 0
    self.learning_rate = 0.00001
    self.weight_decay = 0.001
    self.betas = (0.9, 0.998)
    self.label_smoothing_factor = 0.2
    self.logging_strategy = "epoch"
    self.classifier_dropout = 0.1
    self.dropout = 0.3
    self.attention_dropout = 0.1
    self.warmup_ratio = 0

    ### evaluation params ###
    self.evaluation_strategy = "epoch"

    ### save params ###
    self.save_directory = "/work3/s200925/mBART/checkpoints_3/"
    self.save_strategy = "epoch"
    self.save_total_limit = 30

    ### random seed ###
    self.seed = 1



def main():

    CFG = cfg()

    train_dataset = PhoenixDatasetForMBART(CFG.train_path)
    test_dataset = PhoenixDatasetForMBART(CFG.test_path)
    eval_dataset = PhoenixDatasetForMBART(CFG.eval_path)

    multilingual_model, multilingual_tokenizer = get_model_and_tokenizer(CFG)
    model, tokenizer = reduce_to_vocab(multilingual_model, multilingual_tokenizer, train_dataset, CFG)
    model.to(CFG.device)

    dataloader_train = DataLoader(
      train_dataset, 
      collate_fn = lambda data: collator(data, tokenizer), 
      batch_size=CFG.per_device_train_batch_size, 
      shuffle=True, num_workers=CFG.num_workers)

    dataloader_eval = DataLoader(
      eval_dataset, 
      collate_fn = lambda data: collator(data, tokenizer), 
      batch_size=CFG.per_device_eval_batch_size, 
      shuffle=False, num_workers=CFG.num_workers)

    train(model, tokenizer, dataloader_train, dataloader_eval, CFG)

    # trainer = get_trainer(model, tokenizer, CFG, train_dataset, eval_dataset)

    # if CFG.should_pre_eval:
    #   print("\nPre-evaluation on Eval:\n")
    #   print(trainer.evaluate())

    # if CFG.should_train:
    #   print("\nStrating Training:\n")
    #   trainer.train()

    # if CFG.should_eval:
    #   print("\nEvaluation on Eval:\n")
    #   print(trainer.evaluate())

    # if CFG.should_test:
    #   print("\nEvaluation on Test:\n")
    #   print("\nBaseline:")
    #   print(get_baseline_metrics(test_dataset))
    #   print("\nModel:")
    #   print(trainer.evaluate(test_dataset))

    # if CFG.save_state:
    #   print("\n Saving Trainer State")
    #   trainer.save_state()

if __name__ == '__main__':
  main()