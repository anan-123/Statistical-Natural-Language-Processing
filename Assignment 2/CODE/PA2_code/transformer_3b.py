# add all  your Encoder and Decoder code here
import torch.nn as nn
import torch
import torch.nn.functional as F
d_model = 64 # embedding dimension
sequence_length = 32
from tokenizer import SimpleTokenizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers
import math
import numpy as np

class SparseAttentionHead(nn.Module):
    def __init__(self, head_size, d_model):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.window_size = 5  # the local window size for sparse attention

    def forward(self, x, mask=None):
        B, T, C = x.shape
        k = self.key(x)  
        q = self.query(x)  
        v = self.value(x)      
        flag =0
        if mask is not None:
            flag =1 
        attention_scores = q @ k.transpose(-2, -1) * (C ** -0.5) + self.get_local_window_mask(T, device=x.device)
        if flag == 1:
            attention_scores = attention_scores + mask  
       
        op = F.softmax(attention_scores, dim=-1) @ v  
        return op, F.softmax(attention_scores, dim=-1)

    def get_local_window_mask(self, sequence_length, device):
        mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1)
        for i in range(sequence_length):
            sz = self.window_size//2
            mask[i, : max(0,i-sz)] = float('-inf')
            mask[i, min(sequence_length, i+sz+1):] = float('-inf')
        return mask.to(device)  


class multiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size, d_model):
        super().__init__()
        self.heads = nn.ModuleList([SparseAttentionHead(head_size, d_model) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, d_model)

    def forward(self, x, mask=None):
        outputs = []
        attention_maps = []
        for head in self.heads:
            output, attention = head(x,mask)
            outputs.append(output)
            attention_maps.append(attention)
        
        out = self.proj(torch.cat(outputs, dim=-1))
        return out, attention_maps
      

class Block(nn.Module):
    def __init__(self, d_model, num_heads, hidden_size):
        super().__init__()
        head_size = d_model // num_heads
        self.sa = multiheadAttention(num_heads, head_size, d_model)
        self.ffwd = FeedForward(d_model, hidden_size)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None): 
        x_ln1 = self.ln1(x)
        sa_out, attention_maps = self.sa(x_ln1, mask)  
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, attention_maps

class FeedForwardClassifier(nn.Module):
    def __init__(self,d_model,hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,3)
        )
    def forward(self,x):
        return self.net(x)  

 

class Encoder_3b(nn.Module):
    def __init__(self, vocab_size, d_model, sequence_length, num_heads, num_layers, hidden_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(sequence_length, d_model)
       
        self.blocks = nn.ModuleList([Block(d_model, num_heads, hidden_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = FeedForwardClassifier(d_model, hidden_size)
        
    def forward(self, idx, classify=False, targets=None):
        B, T = idx.shape
        if torch.any(idx >= self.token_embedding_table.num_embeddings):
            raise ValueError(f"Input indices are out of vocabulary range. Max index: {idx.max().item()}, Vocabulary size: {self.token_embedding_table.num_embeddings}")
        
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))

        x = tok_emb + pos_emb
        
        all_attention_maps = []
        for block in self.blocks:
            x, attention_maps = block(x)
            all_attention_maps.extend(attention_maps)
            
        x = self.ln_f(x)
        
        if classify:
            x = x.mean(dim=1)
            logits = self.classifier(x)
            if targets is not None:
                loss = F.cross_entropy(logits, targets)
                return logits, loss, all_attention_maps
            return logits, all_attention_maps
        
        return x, all_attention_maps


class FeedForward(nn.Module):
    def __init__(self,d_model,hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,d_model)
        )
    def forward(self,x):
        return self.net(x)
class Decoder_3b(nn.Module):
    def __init__(self, vocab_size, d_model, sequence_length, num_heads, num_layers, hidden_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(sequence_length, d_model)
        self.blocks = nn.ModuleList([Block(d_model, num_heads, hidden_size) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.final_projection = nn.Linear(d_model, vocab_size)

    def forward(self, idx,targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        mask = self._generate_square_subsequent_mask(T).to(idx.device)
        
        all_attention_maps = []
        for block in self.blocks:
            x, attention_maps = block(x, mask)  
            all_attention_maps.extend(attention_maps)
        
        
        x = self.ln_f(x)
        logits = self.final_projection(x)
        if targets is None:
            return logits, torch.stack(attention_maps, dim=0)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

