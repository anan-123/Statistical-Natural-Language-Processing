# Dataset class for handling sentiment analysis data
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
from utils import Indexer
class SentimentDatasetRANDOM_DAN(Dataset):
    def __init__(self, infile):  # Change _init_ to __init__
        self.indexer = Indexer()
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        self.word_embed_m1 = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        self.word_embed_m2 = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        
        # RAN DAN implementation
        len_pad = max(len(e.words) for e in self.examples)
        self.embeddings = []
        # moving to index since for we need only index and not the embeddings for RANDAN. 
        for ex in self.examples:
            indx = []
            for s in ex.words:
                idx = self.word_embed_m1.word_indexer.index_of(s)
                if idx==-1:
                    indx.append(1)
                else:
                    indx.append(idx)
            if len(indx)>len_pad:
                indx=indx[:len_pad]
            elif len(indx)<len_pad:
                indx = indx + [0 for i in range(len_pad-len(indx))]
            self.embeddings.append(indx)
        
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):  # Change _len_ to __len__
        return len(self.examples)

    def __getitem__(self, idx):  # Change _getitem_ to __getitem__
        return self.embeddings[idx], self.labels[idx]

# Two-layer fully connected neural network
class NN2RANDOM_DAN(nn.Module):
    def __init__(self, embed_size, hidden_size):  
        super(NN2RANDOM_DAN, self).__init__()
        word_embed_m1  = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        word_embed_m2 = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        self.e = nn.Embedding(len(word_embed_m1.word_indexer), embed_size)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size,2)
        

    def forward(self, x):
        x = x.long()
        x = self.e(x)
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Three-layer fully connected neural network
class NN3RANDOM_DAN(nn.Module):
    def __init__(self, embed_size, hidden_size):  
        super(NN3RANDOM_DAN, self).__init__()
        word_embed_m1  = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        word_embed_m2 = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        self.e = nn.Embedding(len(word_embed_m1.word_indexer), embed_size)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden_size, hidden_size)  
        self.fc3 = nn.Linear(hidden_size, 2)
        

    def forward(self,x):
        x = x.long()
        a = self.e(x)
        b = a.mean(dim=1)
        x = F.relu(self.fc1(b))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
