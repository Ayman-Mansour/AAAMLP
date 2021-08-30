# src/lstm.py

import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, embedding_matrix):
        """
        :param reviews: numpy array with vecrtors for all words
        """
        num_words = embedding_matrix.shape[0]
        
        embd_dim = embedding_matrix.shape[1]
        
        self.embedding = nn.Embedding(
            num_embeddings=num_words,
            embedding_dim=embd_dim
        )
        self.embedding.weight = nn.Parameter(
            torch.tensor(
                embedding_matrix,
                dtype=torch.float32
            )
        )
        
        self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(
            embd_dim,
            128,
            bidirectional=True,
            batch_first=True
        )
        
        self.out = nn.Linear(512, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        
        x, _ = self.lstm(x)
        
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        
        out = torch.cat((avg_pool, max_pool), 1)
        out = self.out(out)
        
        return out
        