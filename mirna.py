import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

plt.style.use('fivethirtyeight')

base_number_dict = {"A": 1, "G": 2, "C": 3, "U": 4, "T": 4}


def get_tokens(seq):
    tokens = []
    for base in seq:
        if base in base_number_dict:
            tokens.append(base_number_dict[base])
        else:
            tokens.append(0)
    return torch.LongTensor(tokens)


class PairedDataset(Dataset):
    def __init__(self, seqs1, seqs2, labels):
        self.seqs1 = seqs1
        self.seqs2 = seqs2
        self.labels = labels

    def __len__(self):
        return len(self.seqs1)

    def __getitem__(self, i):
        x1 = get_tokens(self.seqs1[i])
        x2 = get_tokens(self.seqs2[i])
        return x1, x2, torch.as_tensor(self.labels[i]).float()


def collate_paired_sequences(args):
    x1 = [a[0] for a in args]
    x2 = [a[1] for a in args]
    y = [a[2] for a in args]
    x1 = pad_sequence(x1, batch_first=True)
    x2 = pad_sequence(x2, batch_first=True)
    return x1, x2, torch.stack(y, 0)


def plot_losses(losses, val_losses):
    fig = plt.figure(figsize=(10, 4))
    plt.plot(losses, label='Training Loss', c='b')
    plt.plot(val_losses, label='Validation Loss', c='r')
    plt.yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    return fig


# Model

class Featuring(nn.Module):
    def __init__(self, embed_dim, d_model, feature_dim, n_heads, n_layers):
        super().__init__()

        self.feature_dim = feature_dim

        self.embedding = nn.Embedding(5, embed_dim, padding_idx=0)
        self.proj1 = nn.Linear(embed_dim, d_model)
        layer = TransformerLayer(n_heads=n_heads, d_model=d_model,
                                 ff_units=2 * feature_dim, dropout=0.2)
        self.encoder = TransformerEncoder(layer, n_layers=n_layers)
        self.proj2 = nn.Linear(d_model, feature_dim)

        self.conv1 = nn.Conv1d(embed_dim, feature_dim, kernel_size=3,
                               padding=1)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.conv2 = nn.Conv1d(2 * feature_dim, feature_dim, kernel_size=3,
                               padding=1)
        self.conv3 = nn.Conv1d(2 * feature_dim, feature_dim, kernel_size=3,
                               padding=1)

    def forward(self, x):
        # x (b, l)
        x = self.embedding(x)  # b, l, embed_dim
        x = self.proj1(x)  # b, l, d_model
        x = self.encoder(x)  # b, l, d_model
        x = self.proj2(x)  # b, l, feature_dim
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 1)

        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc5(x)

        x = x.squeeze(-1)
        x = F.sigmoid(x)

        return x


class InteractionModel(nn.Module):
    def __init__(self, featuring, classifier, n_heads, n_layers):
        super().__init__()

        self.featuring = featuring
        self.classifier = classifier
        layer = TransformerLayer(n_heads=n_heads, d_model=self.featuring.feature_dim,
                                 ff_units=2*self.featuring.feature_dim, dropout=0.5)
        self.encoder = TransformerEncoder(layer, n_layers=n_layers)

    def forward(self, x1, x2):
        x1 = self.featuring(x1)  # b, m, feature_dim
        x2 = self.featuring(x2)  # b, n, feature_dim
        x = torch.cat((x1, x2), dim=1)  # b, m+n, feature_dim
        x = self.encoder(x)  # b, m+n, feature_dim
        x = torch.mean(x, dim=1)  # b, feature_dim
        x = self.classifier(x)
        return x


# Transformer

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        angular_speed = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * angular_speed)  # even dimensions
        pe[:, 1::2] = torch.cos(position * angular_speed)  # odd dimensions
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x is N, L, D
        # pe is 1, maxlen, D
        scaled_x = x * np.sqrt(self.d_model)
        encoded = scaled_x + self.pe[:, :x.size(1), :]
        return encoded


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = int(d_model / n_heads)
        self.linear_query = nn.Linear(d_model, d_model)
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.alphas = None

    def make_chunks(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        # N, L, D -> N, L, n_heads * d_k
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        # N, n_heads, L, d_k
        x = x.transpose(1, 2)
        return x

    def init_keys(self, key):
        # N, n_heads, L, d_k
        self.proj_key = self.make_chunks(self.linear_key(key))
        self.proj_value = self.make_chunks(self.linear_value(key))

    def score_function(self, query):
        # scaled dot product
        # N, n_heads, L, d_k x # N, n_heads, d_k, L -> N, n_heads, L, L
        proj_query = self.make_chunks(self.linear_query(query))
        dot_products = torch.matmul(proj_query,
                                    self.proj_key.transpose(-2, -1))
        scores = dot_products / np.sqrt(self.d_k)
        return scores

    def attn(self, query, mask=None):
        # Query is batch-first: N, L, D
        # Score function will generate scores for each head
        scores = self.score_function(query)  # N, n_heads, L, L
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        alphas = F.softmax(scores, dim=-1)  # N, n_heads, L, L
        alphas = self.dropout(alphas)
        self.alphas = alphas.detach()

        # N, n_heads, L, L x N, n_heads, L, d_k -> N, n_heads, L, d_k
        context = torch.matmul(alphas, self.proj_value)
        return context

    def output_function(self, contexts):
        # N, L, D
        out = self.linear_out(contexts)  # N, L, D
        return out

    def forward(self, query, mask=None):
        if mask is not None:
            # N, 1, L, L - every head uses the same mask
            mask = mask.unsqueeze(1)

        # N, n_heads, L, d_k
        context = self.attn(query, mask=mask)
        # N, L, n_heads, d_k
        context = context.transpose(1, 2).contiguous()
        # N, L, n_heads * d_k = N, L, d_model
        context = context.view(query.size(0), -1, self.d_model)
        # N, L, d_model
        out = self.output_function(context)
        return out


class SubLayerWrapper(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, sublayer, is_self_attn=False, **kwargs):
        norm_x = self.norm(x)
        if is_self_attn:
            sublayer.init_keys(norm_x)
        out = x + self.drop(sublayer(norm_x, **kwargs))
        return out


class TransformerLayer(nn.Module):
    def __init__(self, n_heads, d_model, ff_units, dropout=0.2):
        super().__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.ff_units = ff_units
        self.attn_heads = MultiHeadedAttention(n_heads, d_model,
                                               dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_units, d_model),
        )
        self.sublayers = nn.ModuleList(
            [SubLayerWrapper(d_model, dropout) for _ in range(2)])

    def forward(self, query, mask=None):
        # SubLayer 0 - Self-Attention
        att = self.sublayers[0](query,
                                sublayer=self.attn_heads,
                                is_self_attn=True,
                                mask=mask)
        # SubLayer 1 - FFN
        out = self.sublayers[1](att, sublayer=self.ffn)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, layer, n_layers=1, max_len=10000):
        super().__init__()

        self.d_model = layer.d_model
        self.pe = PositionalEncoding(max_len, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.layers = nn.ModuleList([copy.deepcopy(layer)
                                     for _ in range(n_layers)])

    def forward(self, query, mask=None):
        # Positional Encoding
        x = self.pe(query)
        for layer in self.layers:
            x = layer(x, mask)
        # Norm
        return self.norm(x)
