import torch
import torchvision

import torch.nn.functional as F
import torch.nn as nn

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class Tokenizer(nn.Module):
    def __init__(self):
        super(Tokenizer, self).__init__()


        self.add_module("Conv_Layers",
          nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding = 1),

                nn.Conv2d(64, 256, kernel_size=7, stride=2, padding=3, bias=False),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)
          )
        )

        self.flattener = nn.Flatten(2, 3)

    def sequence_length(self, n_channels=3, height=224, width=224):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        return self.flattener(self.Conv_Layers(x)).transpose(-2, -1)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8): # dim = 256
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads #d
        self.scale = head_dim ** 0.5

        self.q_fc = nn.Linear(dim, dim)
        self.k_fc = nn.Linear(dim, dim)
        self.v_fc = nn.Linear(dim, dim)

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape #B = BATCH_SIZE(64), N = N_KERNELS(256), C = flatten(mapa_caracteristicas)(16x16 = 256)

        q = self.q_fc(x).reshape((B, self.num_heads, N, C // self.num_heads))
        k = self.k_fc(x).reshape((B, self.num_heads, N, C // self.num_heads))
        v = self.v_fc(x).reshape((B, self.num_heads, N, C // self.num_heads))
        #                        64          4       256            64

        attn = (q @ k.transpose(-2, -1)) / self.scale
        #  q.shape  =  64    4      256      64
        #  k.shape  =  64    4       64     256
        attn = attn.softmax(dim=-1)
        #   64     4      256      256

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        #   transpose     =  64   256      4    256
        #   reshape       =  64   256     1024
        out = self.proj(out)
        #   out  =   256
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048):

        super(TransformerEncoderLayer, self).__init__()

        self.pre_norm = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(dim=d_model, num_heads=nhead)


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = F.gelu

    def forward(self, x):
        mha = self.self_attn(self.pre_norm(x))
        out = x + mha
        out = self.norm1(out)

        out2 = self.linear2(self.activation(self.linear1(out)))
        out = out + out2
        return out


class TransformerClassifier(nn.Module):
    def __init__(self,
                 embedding_dim,
                 num_layers,
                 num_heads,
                 num_classes,
                 sequence_length): #Tokenizer

        super().__init__()

        dim_feedforward = int(embedding_dim * 3)
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length
        self.num_tokens = 0

        self.sequence_pooling = nn.Linear(self.embedding_dim, 1)


        #Positional Embedding
        self.positional_emb = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim),
                                                requires_grad=True)
        nn.init.trunc_normal_(self.positional_emb, std=0.2)

        #Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads,
                                    dim_feedforward=dim_feedforward)

            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embedding_dim)

        #Clasificador
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = x + self.positional_emb

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        #sequence pooling
        x = torch.matmul(F.softmax(self.sequence_pooling(x), dim=1).transpose(-1, -2), x).squeeze(-2)

        x = self.fc(x)
        return x


class CCT(nn.Module):
    def __init__(self,
                 img_size=506,
                 embedding_dim=256,
                 num_layers=6, #14
                 num_heads=4, #6
                 num_classes=292,
                 *args, **kwargs):
        super(CCT, self).__init__()

        self.tokenizer = Tokenizer()

        self.classifier = TransformerClassifier(
            sequence_length=self.tokenizer.sequence_length(n_channels=3,
                                                           height=img_size,
                                                           width=img_size),
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.tokenizer(x)
        return self.classifier(x)
    

