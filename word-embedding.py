import numpy as np
import pandas as pd

import torch
import transformers as ppb

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

def getFeatures(batch_1):

  tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True, truncation=True, max_length=512)))

  max_len = 0
  for i in tokenized.values:
      if len(i) > max_len:
          max_len = len(i)

  padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])


  attention_mask = np.where(padded != 0, 1, 0)
  attention_mask.shape


  input_ids = torch.tensor(padded)  
  attention_mask = torch.tensor(attention_mask)

  with torch.no_grad():
      last_hidden_states = model(input_ids, attention_mask=attention_mask)

  features = last_hidden_states[0][:,0,:].numpy()

  return features

df = pd.read_csv('./data/training-set.csv', delimiter=',')
df = df[['selftext', 'is_suicide']]
df = df.rename(columns={'selftext': 0, 'is_suicide': 1})

bert_features = getFeatures(df)
np.savetxt("bert-training-features.csv", bert_features, delimiter=',')

df = pd.read_csv('./data/testing-set.csv', delimiter=',')
df = df[['selftext', 'is_suicide']]
df = df.rename(columns={'selftext': 0, 'is_suicide': 1})

bert_features = getFeatures(df)
np.savetxt("bert-testing-features.csv", bert_features, delimiter=',')

df = pd.read_csv('./data/combined-set.csv', delimiter=',')
df = df[['selftext', 'is_suicide']]
df = df.rename(columns={'selftext': 0, 'is_suicide': 1})

bert_features = getFeatures(df)
np.savetxt("bert-combined-features.csv", bert_features, delimiter=',')
