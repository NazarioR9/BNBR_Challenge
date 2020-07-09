import torch
from torch.utils.data import Dataset
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from keras.utils import to_categorical
from .utils import getTokenizer

class MentalHealthDataset(Dataset):
  def __init__(self, df, task='train'):
    super(MentalHealthDataset, self).__init__()
    self.text_col = 'text'
    self.target_col = 'label'
    self.length_col = 'length'
    self.df = df.reset_index(drop=True)
    self.task = task
    
  def __len__(self):
    return self.df.shape[0]

  def __getitem__(self, idx):
    text = self.df.loc[idx, self.text_col]
    y = self.df.loc[idx, self.target_col] if self.task=='train' else -1
    length = self.df.loc[idx, self.length_col]

    return text, length, to_categorical(y, 4)
    

class FastTokCollateFn:
    def __init__(self, model_name, max_tokens=50):
        self.tokenizer = getTokenizer(model_name)
        self.max_tokens = max_tokens

    def __call__(self, batch):
        labels = torch.tensor([x[-1] for x in batch])
        #lengths = [x[1] for x in batch]
        #max_pad = min(max(lengths), self.max_tokens)
        max_pad = self.max_tokens

        encoded = self.tokenizer.batch_encode_plus([x[0] for x in batch], return_attention_mask=True, truncation=True, padding=True, max_length=max_pad)
        sequences_padded = torch.tensor(encoded.input_ids)
        attention_masks_padded = torch.tensor(encoded.attention_mask)
        
        return (sequences_padded, attention_masks_padded), labels
