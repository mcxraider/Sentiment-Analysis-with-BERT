import numpy as np
import datetime
import pandas as pd
import torch
from torch import cuda
import transformers
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from transformers import DistilBertModel, DistilBertTokenizer
import re

# Load test data
test_split = pd.read_csv('/content/drive/MyDrive/test_split.csv')
test_split = test_split[['Content','rating']]

# Convert to binary classification
def good_bad(row):
    if row < 5:
        return 0
    else:
        return 1
test_split['rating'] = test_split['rating'].apply(good_bad)


# DistillBERT with custom classification head finetuning
VALID_BATCH_SIZE = 1
MAX_LEN = 512
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
    ''' Custom dataset class defined to create '''

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.Content = dataframe.Content.to_numpy()
        self.targets = dataframe.rating.to_numpy()
        self.max_len = max_len

    # __len__ and __getitem__ methods to create map-style dataset to be interfaced by torch DataLoader method
    def __len__(self):
        return len(self.Content)

    def __getitem__(self, index):
        # Data preprocessing code to convert to lower-cased, remove trailing whitespace, html tags and urls
        Content = str(self.Content[index]).lower()
        Content = re.sub(r'<[^>]+>', '', Content)
        Content = re.sub(r'https://\S+|www\.\S+', '', Content)
        Content = re.sub(r'br\s', '', Content)
        Content = " ".join(Content.split())

        rating = self.targets[index]

        # Tokenisation of text
        inputs = self.tokenizer.encode_plus(
            Content,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            pad_to_max_length=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(rating, dtype=torch.int)
        }

class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    # Note: DistilBERT outputs a tuple where the first element at index 0
    # represents the hidden-state at the output of the model's last layer.
    # It is a tensor of shape (batch_size, sequence_length, hidden_size=768)
    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

# Preparing testing data and dataloader
testing_set = CustomDataset(test_split, tokenizer, MAX_LEN)
test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': False,
                'sampler': SequentialSampler(test_split),
                'num_workers': 0
                }
testing_loader = DataLoader(testing_set, **test_params)

# Instantiate model
model = DistillBERTClass()
model.to(device)

model.load_state_dict(torch.load(r'/content/drive/MyDrive/IT1244 final project/pytorch_distilbert_moviereview'))

def predict(model, testing_loader):
    model.eval()
    preds=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            outputs = model(ids, mask)
            big_val, big_idx = torch.max(outputs.data, dim=1)
            preds.append(int(big_idx[0]))

    return preds

print(predict(model, testing_loader))
