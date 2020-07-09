import os, gc, random
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, f1_score
from collections import Counter
from IPython.display import clear_output
import torch
from transformers import (
    AutoTokenizer, RobertaTokenizerFast, 
    BertTokenizerFast, ElectraTokenizerFast
)

def seed_everything(seed):
  print(f'Set seed to {seed}.')
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available(): 
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_blackbone(n):
  return n.startswith('model')    

def balance_training(df):
  threshold = 150

  dep = df[df.label==0].copy()
  alc = df[df.label==1].copy()
  sui = df[df.label==2].copy()
  dru = df[df.label==3].copy()
  ndep = dep.sample(n=threshold, random_state=seed)

  ndf = pd.concat([sui, dru, ndep, alc], axis=0)
  ndf = ndf.sample(frac=1, random_state=seed).reset_index(drop=True)

  print('Rebalance ', Counter(ndf.label.values))

  return ndf.copy()
  
def evaluation(ytrue, y_pred, labels=[0,1,2,3]):
  log = log_loss(ytrue, y_pred, labels=labels)
  f1 = f1_score(ytrue, y_pred.argmax(1), average='weighted')
  log_score = max(0, (1-log))
  log_f1 = (f1 + log_score)*0.5

  score = np.array([ 
    log,
    f1,
    log_f1
  ])

  return score

def getTokenizer(model_name):
  if 'roberta' in model_name:
    return RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=False)
  elif model_name.startswith('bert'):
    return BertTokenizerFast.from_pretrained(model_name, add_prefix_space=False)
  elif 'bart' in model_name:
    return RobertaTokenizerFast.from_pretrained('roberta-large', add_prefix_space=False) #check https://github.com/huggingface/transformers/blob/68e19f1c228c92d5d800533f558faff24b57127a/src/transformers/tokenization_bart.py#L27
  elif 'electra' in model_name:
    return ElectraTokenizerFast.from_pretrained(model_name, add_prefix_space=False)
  else: return AutoTokenizer.from_pretrained(model_name, add_prefix_space=False)

class EarlyStopping:
  def __init__(self, patience=3, mode='max'):
    self.step = 0
    self.patience = patience
    self.mode = mode
    self.stop = False
    self.logF1 = 0

  def update(self, logF1):
    if self.logF1 > logF1:
      self.step += 1
    else: 
      self.step = 0
      self.logF1 = logF1
    
    if self.step == self.patience: 
      self.stop = True


class Printer:
  def __init__(self, fold=0):
    self.print = []
    self.fold = fold

  def pprint(self, kwargs):
    str_log = ""
    for key in kwargs.keys():
      str_log += "{}: {} - ".format(key, kwargs[key])
    self.print.append(str_log.strip(" -"))
    self.show()

  def update(self, epoch, losses, score):
    str_log = "Epoch: {} - Loss: {:.5f} - ValLoss: {:.5f} - Log: {:.5f} - F1: {:.5f} - LogF1: {:.5f}".format(epoch, losses['loss'][epoch], losses['val_loss'][epoch], *score)
    self.print.append(str_log)

  def show(self):
    clear_output()

    print("_"*100, "\nFold ", self.fold)
    for p in self.print:
      print("\t" + "_" * 100)
      print("\t"+'| '+ p)

  def update_and_show(self, epoch, losses, score):
    self.update(epoch, losses, score)
    self.show()


class WorkplaceManager:
  def __init__(self, seed, dirs, exts, n_fols=10):
    self.seed = seed
    self.dirs = dirs
    self.exts = exts
    self.n_folds = 10

    self._set_workplace()

  @staticmethod
  def create_dir(dir):
    os.makedirs(dir, exist_ok=True)
  
  def _create_dirs(self):
    print('Created {}'.format(' '.join(self.dirs)))
    for d in self.dirs:
      self.create_dir(d)
  
  def _clear_dirs(self):
    print('Deleted {}'.format(' '.join(self.dirs)))
    self.clear([f'{d}*' for d in self.dirs])

  def _clear_files(self):
    print('Deleted {}'.format(' '.join(self.exts)))
    self.clear([f'*{ext}' for ext in self.exts])

  def clear(self, objs_name):
    os.system('rm -r {}'.format(' '.join(objs_name)))

  def _set_workplace(self):
    seed_everything(self.seed)
    if os.path.exists('models') and len(os.listdir('models/')) == self.n_folds:
      self._clear_dirs()
      self._clear_files()    
    self._create_dirs()

class CrossValLogger:
  def __init__(self, df, n_folds=10, oof_cv = 'cv_score.pkl', path='evals/roberta-base/'):
    assert df.fold.nunique()==n_folds, "Unconsistency betwwen df.n_folds and n_folds"

    self.df = df.copy()
    self.path = path
    self.n_folds = n_folds
    self.oof_cv = oof_cv
    self.score1, self.score2 = None, None

  def _retrieve_eval_preds(self):
    ph = self.path+'fold_{}_best_eval.npy'
    shape = ( self.df.shape[0], self.df.label.nunique() )
    preds = np.empty(shape, dtype=np.float32)
    for i in self.df.fold.unique():
      index = self.df[self.df.fold==i].index.values
      fold_pred = np.load(ph.format(i))
      preds[index] = fold_pred[:, :]
    return preds

  def _load_oof_cv_score(self):
    score = 0
    with open(self.oof_cv, 'rb') as f:
      score = pickle.load(f)
      f.close()
    return score

  def show_results(self, return_score=False):
    if self.score1 is None:
      eval_preds = self._retrieve_eval_preds()
      self.score1 = self._load_oof_cv_score() / self.n_folds #oof_cv_score
      self.score2 = evaluation(self.df.label.values, eval_preds)[0] #ovr_score

    print('OOF_CV_SCORE: {:.5f} | OVR_SCORE: {:.5f}'.format(self.score1, self.score2))
    
    if return_score: return self.score1, self.score2
