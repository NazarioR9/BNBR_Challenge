import os, sys, gc
import pickle
import subprocess
import numpy as np
import pandas as pd
from typing import List
from functools import lru_cache
from argparse import Namespace
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from transformers import AutoConfig, AutoModel, BartConfig, AdamW, get_linear_schedule_with_warmup

from .activation import Mish
from .utils import evaluation, is_blackbone, Printer, WorkplaceManager
from .data import MentalHealthDataset, FastTokCollateFn

def get_model(global_config, **kwargs):
  if 'bart' in global_config.model_name:
    return BARTyMHM(global_config, **kwargs)
  else:
    return BERTyMHM(global_config, **kwargs)


class BaseMHM(nn.Module):
  def __init__(self, global_config, n_classes=4, **kwargs):
    super(BaseMHM, self).__init__()

    self.global_config = global_config
    try:
    	self.model = AutoModel.from_pretrained(global_config.model_path)
    except AttributeError:
    	self.model = AutoModel.from_pretrained(global_config.model_name)

    self.low_dropout = nn.Dropout(global_config.low_dropout)
    self.high_dropout = nn.Dropout(global_config.high_dropout)
    self.activation = global_config.mhm_activation


  def _init_weights(self, layer):
    layer.weight.data.normal_(mean=0.0, std=0.02)
    if layer.bias is not None:
      layer.bias.data.zero_()

  def _reinit(self, L=4, n=11):
    '''
    This is an humble implementation of "Revisiting Few-sample BERT Fine-tuning" (https://arxiv.org/abs/2006.05987)
    L is the max number of layer (at the head) to re-initialize
    n is the number of encoder layer in the model
    '''

    params = {
      'pooler': ['pooler.dense'],
      'encoder': [
        'encoder.layer.{}.attention.self.query',
        'encoder.layer.{}.attention.self.key',
        'encoder.layer.{}.attention.self.value',
        'encoder.layer.{}.attention.output.dense',
        'encoder.layer.{}.attention.output.LayerNorm',
        'encoder.layer.{}.intermediate.dense',
        'encoder.layer.{}.output.dense',
        'encoder.layer.{}.output.LayerNorm',
      ],
    }

    final_params_name = params['pooler'] + [ln.format(n-l) for l in range(L) for ln in params['encoder']]

    for layer_name in final_params_name:
      layer = self.model
      for ln in layer_name.split('.'):
        layer = getattr(layer, ln)
      self._init_weights(layer)

  def freeze(self):
    for child in self.model.children():
      for param in child.parameters():
        param.requires_grad = False

  def unfreeze(self):
    for child in self.model.children():
      for param in child.parameters():
        param.requires_grad = True



class BERTyMHM(BaseMHM):
  def __init__(self, global_config, n_classes=4, **kwargs):
    super(BERTyMHM, self).__init__(global_config, n_classes, **kwargs)

    config = AutoConfig.from_pretrained(global_config.model_name)
    self.config = config

    self.l0 = nn.Linear(config.hidden_size, config.hidden_size)
    self.classifier = nn.Linear(config.hidden_size, n_classes)

    self._init_weights(self.l0)
    self._init_weights(self.classifier)

    if global_config.reinit: self._reinit(global_config.L, global_config.n)

  
  def forward(self, inputs):
    input_ids, attention_mask = list(inputs)[:2]

    outputs = self.model(input_ids, attention_mask=attention_mask)
    x = outputs[0][:, 0, :]

    x = self.l0( self.low_dropout(x))
    x = self.activation(x)
    x = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(x)) for _ in range(self.global_config.multi_samp_nb)],
                dim=0,
            ),
            dim=0,
        )
    return x


class BARTyMHM(BaseMHM):
  def __init__(self, global_config, n_classes=4, **kwargs):
    super(BARTyMHM, self).__init__(global_config, n_classes, **kwargs)

    config = BartConfig.from_pretrained(global_config.model_name)
    self.config = config

    self.l0 = nn.Linear(config.hidden_size, config.hidden_size)
    self.classifier = nn.Linear(config.hidden_size, n_classes)

    self._init_weights(self.l0)
    self._init_weights(self.classifier)

    if global_config.reinit: self._reinit(global_config.L, global_config.n)

  
  def forward(self, inputs):
    input_ids, attention_mask = list(inputs)[:2]

    outputs = self.model(input_ids, attention_mask=attention_mask)
    x = outputs[0]  # last hidden state
    eos_mask = input_ids.eq(self.config.eos_token_id)

    if len(torch.unique(eos_mask.sum(1))) > 1:
        raise ValueError("All examples must have the same number of <eos> tokens.")

    x = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
    x = self.l0( self.low_dropout(x))
    x = self.activation(x)
    x = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(x)) for _ in range(self.global_config.multi_samp_nb)],
                dim=0,
            ),
            dim=0,
        )
    return x

class LightTrainingModule(nn.Module):
    def __init__(self, global_config):
        super().__init__()
	
        self.model = get_model(global_config)
        self.loss = global_config.loss
        self.loss_name = global_config.loss_name
        self.activation = global_config.activation
        self.global_config = global_config
        self.losses = {'loss': [], 'val_loss': []}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model.to(self.device)


    def step(self, batch, step_name="train", epoch=-1):
        X, y = batch
        X, y = (x.to(self.device) for x in X), y.to(self.device)
        y_probs = self.forward(X)

        if self.loss_name == 'ce':
          loss = self.loss(y_probs, y.argmax(1).long(), epoch)
          try:
          	y_probs = self.activation(y_probs, 1) #softmax
          except:
          	y_probs = self.activation(y_probs) #sigmoid
        else:
          try:
          	y_probs = self.activation(y_probs, 1) #softmax
          except:
          	y_probs = self.activation(y_probs) #sigmoid
          loss = self.loss(y_probs, y, epoch)

        loss_key = f"{step_name}_loss"

        return { ("loss" if step_name == "train" else loss_key): loss.cpu()}, y_probs.cpu()

    def forward(self, X, *args):
        return self.model(X, *args)

    def training_step(self, batch, batch_idx, epoch):
        return self.step(batch, "train", epoch)
    
    def validation_step(self, batch, batch_idx, epoch):
        return self.step(batch, "val", epoch)

    def training_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.losses['loss'].append(loss.item())

        return {"train_loss": loss}

    def validation_epoch_end(self, outputs: List[dict]):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.losses['val_loss'].append(loss.item())

        return {"val_loss": loss}
        
    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")
    
    def train_dataloader(self):
        return self.create_data_loader(self.global_config.train_df, shuffle=True)

    def val_dataloader(self):
        return self.create_data_loader(self.global_config.val_df)

    def test_dataloader(self):
        return self.create_data_loader(self.global_config.test_df, 'test')
                
    def create_data_loader(self, df: pd.DataFrame, task='train', shuffle=False):
        return DataLoader(
                    MentalHealthDataset(df, task),
                    batch_size=self.global_config.batch_size if task=='train' else int(0.25*self.global_config.batch_size),
                    shuffle=shuffle,
                    collate_fn=FastTokCollateFn(self.global_config.model_name, self.global_config.max_tokens)
        )
        
    @lru_cache()
    def total_steps(self):
        return len(self.train_dataloader()) // self.global_config.accumulate_grad_batches * self.global_config.epochs

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        param_optimizer = self.model.named_parameters()
        optimizer_grouped_parameters = [
             {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
             {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.global_config.lr)
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=self.global_config.warmup_steps,
                    num_training_steps=self.total_steps(),
        )
        if self.global_config.swa: optimizer = SWA(optimizer, self.global_config.swa_start, self.global_config.swa_freq, self.global_config.swa_lr)
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

class Trainer:
  def __init__(self, global_config, **kwargs):

    if global_config.task=='train':
      self.best_eval = []
      self.scores = []
      self.best_log = np.inf
      self.best_metric = 0
      self.global_config = global_config
      self.fold = global_config.fold
      self.printer = Printer(global_config.fold)
      self.module = LightTrainingModule(global_config)
      self.opts, scheds = self.module.configure_optimizers()
      self.scheduler = scheds[0]['scheduler']
      self.train_dl = self.module.train_dataloader()
      self.val_dl = self.module.val_dataloader()
    else:
      self.probs = None
      self.module = kwargs['module']
      self.test_dl = self.module.test_dataloader()

  def train(self, epoch):
    self.module.train()
    self.module.zero_grad()
    outputs = []

    for i, batch in enumerate(tqdm(self.train_dl, desc='Training')):
      output, _ = self.module.training_step(batch, i, epoch)
      outputs.append(output)

      output['loss'].backward()

      if (i+1) % self.module.global_config.accumulate_grad_batches == 0:
        if self.global_config.clip_grad: nn.utils.clip_grad_norm_(self.module.model.parameters(), self.global_config.max_grad_norm)
        self.opts[0].step()
        if self.global_config.scheduler: self.scheduler.step()
      self.module.zero_grad()
    
    self.module.training_epoch_end(outputs)

  def evaluate(self, epoch):
    self.module.eval()

    with torch.no_grad():
      score = np.zeros((3,))
      outputs = []
      eval_probs = []

      for i, batch in enumerate(tqdm(self.val_dl, desc='Eval')):
        output, y_probs = self.module.validation_step(batch, i, epoch)
        y_probs = y_probs.detach().cpu().numpy()      
        score += self.get_score(batch, y_probs)
        eval_probs.append(y_probs.reshape(-1, 4))
        outputs.append(output)
      score = score/len(self.val_dl)
      self.scores.append(score)
      self.module.validation_epoch_end(outputs)
      self._check_evaluation_score(score[-1], score[0], eval_probs)
    
  def predict(self):
    if self.probs is None:
      self.module.eval()
      self.probs = []

      with torch.no_grad():
        for i, batch in enumerate(self.test_dl):
          _, y_probs = self.module.test_step(batch, i)
          self.probs += y_probs.detach().cpu().numpy().tolist()
    else:
      print('[WARNINGS] Already predicted. Use "trainer.get_preds()" to obtain the preds.')

  def fit_one_epoch(self, epoch):
    self.train(epoch)
    if self.global_config.swa and (self.global_config.epochs-1) == epoch:
    	self.opts[0].swap_swa_sgd()
    self.evaluate(epoch) 
    self.printer.update_and_show(epoch, self.module.losses, self.scores[epoch])


  def get_score(self, batch, y_probs):
    return evaluation(batch[-1].cpu().numpy().argmax(1), y_probs)

  def evaluate_post_processing(self):
    szs = len(self.val_dl)
    score, s_drugs, s_alc = 0, 0, 0
    with torch.no_grad():
      for i, batch in enumerate(tqdm(self.val_dl, desc='Evaluate Post-processing: ')):
        s_drugs += self.get_score(batch, self._pp_drugs(self.best_eval[i]))[0]
        s_alc += self.get_score(batch, self._pp_alcohol(self.best_eval[i]))[0]
        score += self.get_score(batch, self.post_process(self.best_eval[i]))[0]
      score = score / szs
      s_drugs = s_drugs / szs
      s_alc = s_alc / szs
    self.printer.pprint({"[PP DRUGS]  [BEST SCORE]": self.best_log, '[POST PROCESS SCORE]': s_drugs, "[VERDICT]": s_drugs<=self.best_log})
    self.printer.pprint({"[PP ALCOH]  [BEST SCORE]": self.best_log, '[POST PROCESS SCORE]': s_alc, "[VERDICT]": s_alc<=self.best_log})
    self.printer.pprint({"[BEST SCORE]": self.best_log, '[POST PROCESS SCORE]': score, "[VERDICT]": score<=self.best_log})

  def get_preds(self):
    return self.probs

  @staticmethod
  def post_process(pred):
    # pred[pred[:, :] > 0.9] = 1.0
    pp_pred = Trainer._pp_drugs(pred)
    pp_pred = Trainer._pp_alcohol(pp_pred)
    pp_pred[pp_pred[:, :] < 0.009] = 0.0

    return pp_pred

  @staticmethod
  def _pp_drugs(preds):
    pred = np.copy(preds)
    pred[pred.argmax(1)==3, 3] = 1.0
    return pred
  
  @staticmethod
  def _pp_alcohol(preds):
    pred = np.copy(preds)
    pred[pred[:, 1] > 0.9] = 1.0
    return pred


  def _save_weights(self, half_precision=False, path='models/'):
    print('Saving weights ...')
    if half_precision: self.module.half() #for fast inference
    torch.save(self.module.state_dict(), f'{path}model_{self.fold}.bin')
    gc.collect()

  def _check_evaluation_score(self, metric, log_score, best_eval=None):
    if metric > self.best_metric:
      self.best_metric = metric
      self.best_log = log_score
      self.best_eval = best_eval
      self._save_weights()

  def save_best_eval(self, path='evals/{}/fold_{}'):
    if self.global_config.task=='train':
      np.save(path.format(self.global_config.model_name, self.global_config.fold)+'_best_eval.npy', np.vstack(self.best_eval))
