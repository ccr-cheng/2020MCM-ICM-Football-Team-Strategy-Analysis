import torch
import torch.nn as nn

import math


class PositionEncoder(nn.Module):
    def __init__(self, max_len, emb_size, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, emb_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layer, pad_token, max_len=64, dropout_p=0.3):
        super().__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.dropout_p = dropout_p
        self.pad_token = pad_token
        self.scale = math.sqrt(emb_size)

        self.embedding = nn.Embedding(input_size, emb_size)
        self.coor_proj = nn.Linear(4, emb_size)
        self.pos_encoder = PositionEncoder(max_len, emb_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=4,
                                                   dim_feedforward=hidden_size,
                                                   dropout=dropout_p, activation='gelu')
        encoder_norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layer, norm=encoder_norm)
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(emb_size, 1)

        self.initialize_weights(self)

    @staticmethod
    def initialize_weights(m):
        if hasattr(m, 'weight') and m.weight.dim() > 1:
            nn.init.xavier_uniform_(m.weight.data)

    @staticmethod
    def generate_mask(src, pad_token):
        '''
        Generate mask for tensor src
        :param src: tensor with shape (max_src, b)
        :param pad_token: padding token
        :return: mask with shape (b, max_src) where pad_token is masked with 1
        '''
        mask = (src.t() == pad_token)
        return mask.to(src.device)

    @staticmethod
    def generate_submask(src):
        sz = src.size(0)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(src.device)

    def saveModel(self, path):
        print(f'Saving model to {path} ...')
        torch.save({
            'model_state': self.state_dict(),
        }, path)

    def forward(self, src, coords):
        '''
        Forward pass
        :param src: src event indices with shape (max_len, b)
        :param coords: coordinates with shape (max_len, b, 4)
        :return: score with shape (b,)
        '''
        pad_mask = self.generate_mask(src, self.pad_token)
        src = self.embedding(src) * self.scale + self.coor_proj(coords)
        src = self.pos_encoder(src)
        output = self.encoder(src, src_key_padding_mask=pad_mask)
        output = torch.mean(output, dim=0)
        score = self.fc(self.dropout(output)).squeeze(-1)
        return score


if __name__ == '__main__':
    from torch.utils.tensorboard import SummaryWriter

    transformer = TransformerEncoder(50, 64, 128, 3, 11)
    print(transformer)

    test_src = torch.randint(1, 10, (10, 16))
    test_coords = torch.randint(1, 100, (10, 16, 4)).float()

    writer = SummaryWriter('runs/transformer_1')
    writer.add_graph(transformer, (test_src, test_coords))
    writer.close()

    test_output = transformer(test_src, test_coords.float())
    print(test_output.size())
