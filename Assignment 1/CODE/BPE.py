
# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sentiment_data import read_sentiment_examples, read_word_embeddings
import numpy as np
import re
from collections import defaultdict, Counter
from utils import Indexer
from torch.nn.utils.rnn import pad_sequence
i = 0
# get good pair at index.
def get_the_good_pair(w, bp):
    final_ans = "" 
    for i in range(len(w)):
        if i < len(w) - 1 and w[i] == bp[0] and w[i + 1] == bp[1]:
            final_ans += ''.join(bp)
            i += 1  
        else:
            final_ans += w[i] 

    return final_ans


def bpe_encoding(texts,n):
    # find the frequency dict to use
    freq = {}
    for t in texts:
        for w in t.split():
            if w not in freq:
                freq[w] = 1
            else:
                freq[w] += 1
    merged_freq = []
    for i in range(n):
        # finding the pairs
        pairs = {}
        bp = None
        hfreq = -1
        for w, f in freq.items():
            str_p = w.split()
            for i in range(len(str_p) - 1):
                if (str_p[i], str_p[i + 1]) not in pairs: # if its already present update the freq 
                    pairs[(str_p[i], str_p[i + 1])] = 1
                else:
                    pairs[(str_p[i], str_p[i + 1])] += 1
        # if no such pair exit loop
        if not pairs:
            break
        # if there are pairs find the highest freq
        for p, f in pairs.items():
            if f > hfreq:
                hfreq = f
                bp = p

        if bp is not None:
            merged_freq.append(bp)
        # get the new frequency 
        new_freq = {}
        for w, f in freq.items():
            final_ans = get_the_good_pair(w,bp)
            if final_ans not in new_freq:
                new_freq[final_ans] = f
            else:
                new_freq[final_ans] += f
        freq = new_freq
    # return freq after merging.
    return merged_freq
    
    
class BPE(Dataset):
    def __init__(self, infile, vectorizer=None, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        self.word_embed_m1 = read_word_embeddings("data/glove.6B.50d-relativized.txt")
        self.word_embed_m2 = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        # part 2
       # self.word_embed_m3 = torch.nn.Embedding(self.word_embed_m1.get_embedding_length(),50)
     

        self.labels = [ex.label for ex in self.examples]
       #  print(" sentences = ",self.sentences)
        self.ed = []
        #length of pad sequence
        len_pad = max(len(e.words) for e in self.examples)
        for ex in self.examples:
            # freq size/ merge limitation
            n = 1500
            print(n)
            merged_freq = bpe_encoding([' '.join(ex.words)],n)
            final_txt = []
            for t in [' '.join(ex.words)]:
                for w in t.split():
                    for p in merged_freq:
                        w = w.replace(''.join(p),p[0]+' '+p[1])
                    final_txt.extend(w.split())

            indx=[]
            for s in final_txt:
                idx = self.word_embed_m1.word_indexer.index_of(s)
                if idx==-1:
                    indx.append(1)
                else:
                    indx.append(idx)
            embed_vec = indx
            if len(embed_vec)>len_pad:
                embed_vec=embed_vec[:len_pad]
            elif len(embed_vec)<len_pad:
                embed_vec = embed_vec + [0 for i in range(len_pad-len(embed_vec))]
            
            self.ed.append(embed_vec)

      
        # Convert embeddings and labels to PyTorch tensors
        self.ed = torch.tensor(self.ed, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.ed[idx], self.labels[idx]


# Two-layer fully connected neural network
class NN2BPE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NN2BPE, self).__init__()
        
        self.word_embed_m2 = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        self.a = nn.Embedding(len(self.word_embed_m2.word_indexer), 300)
        self.fc1 = nn.Linear(300, hidden_size)  # Adjusted input size to match embedding output
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = x.long() 
        e = self.a(x)  
        x = F.relu(self.fc1(e.mean(dim=1)))  
        x = self.dropout(x)  
        x = self.fc2(x)  
        x = self.log_softmax(x)  
        return x


    


class NN3BPE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.word_embed_m2 = read_word_embeddings("data/glove.6B.300d-relativized.txt") # comment
        self.a = torch.nn.Embedding(len(self.word_embed_m2.word_indexer),300) # freq size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x=x.long() # commen t
        e = self.a(x) #comment
        x = F.relu(self.fc1(torch.mean(e,dim=1))) # give x as input
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.log_softmax(x)