# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset
from sentiment_data import read_word_embeddings
import numpy as np

# Dataset class for handling sentiment analysis data
class SentimentDatasetDAN(Dataset):
    def __init__(self, infile):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        self.word_embed_m1 = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        self.word_embed_m2 = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        
        # DAN implementation
        self.embeddings = []
        for s in self.sentences:
            word = s.split()
            embed_vec = [self.word_embed_m2.get_embedding(w) for w in word]
            self.embeddings.append(np.mean(embed_vec,axis=0))
           
        self.embeddings = np.array(self.embeddings)
        
        # Convert embeddings and labels to PyTorch tensors
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]


# Two-layer fully connected neural network
class NN2DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

    
# Three-layer fully connected neural network
class NN3DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.log_softmax(x)

