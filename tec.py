import einops
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
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


class ContactCNN(nn.Module):
    def __init__(self, projection_dim):
        super().__init__()

        ks = 9

        self.conv1 = nn.Conv2d(2 * projection_dim, projection_dim, ks,
                               padding=ks // 2)
        self.batch_norm1 = nn.BatchNorm2d(projection_dim)
        self.activation1 = nn.ReLU()

        self.conv2 = nn.Conv2d(projection_dim, projection_dim // 2, ks,
                               padding=ks // 2)
        self.batch_norm2 = nn.BatchNorm2d(projection_dim // 2)
        self.activation2 = nn.ReLU()

        self.conv3 = nn.Conv2d(projection_dim // 2, projection_dim // 4, ks,
                               padding=ks // 2)
        self.batch_norm3 = nn.BatchNorm2d(projection_dim // 4)
        self.activation3 = nn.ReLU()

        self.conv4 = nn.Conv2d(projection_dim // 4, 1, ks, padding=ks // 2)
        self.batch_norm4 = nn.BatchNorm2d(1)
        self.activation4 = nn.Sigmoid()
        self.clip()

    def clip(self):
        self.conv2.weight.data[:] = 0.5 * (
                self.conv2.weight + self.conv2.weight.transpose(2, 3))

    def forward(self, x1, x2):
        # x1 (b, m, d), x2 (b, n, d)
        x1 = x1.transpose(1, 2)  # b, d, m
        x2 = x2.transpose(1, 2)  # b, d, n

        dif = torch.abs(x1.unsqueeze(3) - x2.unsqueeze(2))
        mul = x1.unsqueeze(3) * x2.unsqueeze(2)
        cat = torch.cat([dif, mul], 1)

        x = self.conv1(cat)
        x = self.activation1(x)
        x = self.batch_norm1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)

        x = self.conv3(x)
        x = self.activation3(x)
        x = self.batch_norm3(x)

        x = self.conv4(x)
        x = self.activation4(x)
        x = self.batch_norm4(x)

        return x


class PositionalEncoder(torch.nn.Module):
    def __init__(self, dim_model, max_wavelength=10000):
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin
        scale = max_wavelength / (2 * np.pi)

        sin_term = scale ** (torch.arange(0, n_sin).float() / (n_sin - 1))
        cos_term = scale ** (torch.arange(0, n_cos).float() / (n_cos - 1))
        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        # x (b, l, d_model)
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X  # b, l, d_model


class EmbeddingTransform(nn.Module):
    def __init__(self, nin, nout, dropout=0.2, nhead=1, num_layers=1,
                 activation=nn.ReLU()):
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.dropout_p = dropout
        self.embedding = nn.Embedding(5, nin, padding_idx=0)
        self.position_embedding = PositionalEncoder(nin)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=nin,
            nhead=nhead,
            dim_feedforward=nin * 2,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )
        self.transform = nn.Linear(nin, nout)
        self.activation = activation
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        # x (b, l)
        # mask = ~x.sum(dim=1).bool()
        x = self.embedding(x)  # b, l, d_model
        x = self.position_embedding(x)
        # x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.transformer_encoder(x)
        x = self.drop(self.activation(self.transform(x)))
        return x


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:

    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`

    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        if train:
            self.k = nn.Parameter(torch.FloatTensor([float(k)]))
        else:
            self.k = k

    def forward(self, x):
        x = torch.clamp(1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0,
                        max=1).squeeze()
        return x


class ModelInteraction(nn.Module):
    def __init__(self, embedding_transform, contact):
        super().__init__()
        self.activation = LogisticActivation(x0=0.5, k=20, train=True)
        self.embedding_transform = embedding_transform
        self.contact = contact
        self.gamma = nn.Parameter(torch.FloatTensor([0]))
        self.clip()

    def clip(self):
        self.contact.clip()
        self.gamma.data.clamp_(min=0)

    def forward(self, x1, x2):
        x1 = self.embedding_transform(x1)  # b, m, d
        x2 = self.embedding_transform(x2)  # b, n, d
        x = self.contact(x1, x2)

        mu = torch.mean(x, dim=(1, 2, 3)).repeat(
            x.shape[2] * x.shape[3], 1).T.reshape(x.shape[0],
                                                        x.shape[1],
                                                        x.shape[2],
                                                        x.shape[3])
        sigma = torch.var(x, dim=(1, 2, 3)).repeat(
            x.shape[2] * x.shape[3], 1).T.reshape(x.shape[0],
                                                        x.shape[1],
                                                        x.shape[2],
                                                        x.shape[3])
        Q = torch.relu(x - mu - (self.gamma * sigma))
        phat = torch.sum(Q, dim=(1, 2, 3)) / (
                torch.sum(torch.sign(Q), dim=(1, 2, 3)) + 1)
        phat = self.activation(phat)
        return phat


def predict_interaction(model, x1, x2):
    phat = model(x1, x2)
    return phat


# def interaction_grad(device, model, x1, x2, y):
#     phat = predict_interaction(model, x1, x2)
#     y = y.to(device)
#     y = Variable(y)
#     p_hat = phat.float()
#     loss = F.binary_cross_entropy(p_hat.float(), y.float())
#     loss.backward()
#
#     y = y.cpu()
#     p_hat = p_hat.cpu()
#
#     with torch.no_grad():
#         guess_cutoff = 0.5
#         p_hat = p_hat.float()
#         p_guess = (guess_cutoff * torch.ones(len(p_hat)) < p_hat).float()
#         y = y.float()
#
#     return loss.item(), p_guess.int().tolist(), y.int().tolist()


# def interaction_eval(device, model, x1, x2, y):
#     accuracy_weight = 0.35
#     c_map, p_hat = predict_cmap_interaction(model, x1, x2)
#     y = y.to(device)
#     y = Variable(y)
#     p_hat = p_hat.float()
#     bce_loss = F.binary_cross_entropy(p_hat.float(), y.float())
#     accuracy_loss = bce_loss
#     representation_loss = torch.mean(c_map_mag)
#     loss = (accuracy_weight * accuracy_loss) + ((1 - accuracy_weight) * representation_loss)
#
#     y = y.cpu()
#     p_hat = p_hat.cpu()
#     with torch.no_grad():
#         guess_cutoff = 0.5
#         p_hat = p_hat.float()
#         p_guess = (guess_cutoff * torch.ones(len(p_hat)) < p_hat).float()
#         y = y.float()
#
#     return loss.item(), p_guess.int().tolist(), y.int().tolist()


def calculate_metrics(labels, predictions):
    labels = np.array(labels)
    predictions = np.array(predictions)
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    print(f"tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}")
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fpr = fp / (fp + tn)
    return accuracy, recall, precision, fpr
