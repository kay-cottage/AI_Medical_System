import torch
#import clip
from PIL import Image
import time
#from text_encoder import C
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from typing import Tuple, Union



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)



class Text_Encoder(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length



        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        #512 12 8 tensor(
        #print(transformer_width,transformer_layers,transformer_heads,self.build_attention_mask())

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        #print(vocab_size, transformer_width,self.context_length)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)


        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return torch.FloatTensor

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        #x1 torch.Size([1, 77, 512])
        #x2 torch.Size([1, 77, 512])
        #x3 torch.Size([77, 1, 512])
        #print('x1',x.shape)
        x = x + self.positional_embedding.type(self.dtype)
        #print('x2',x.shape)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        #print('x3',x.shape)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

def save_model():
    model=Text_Encoder(1024,77,49408,512,8,12)
    torch.save(model.state_dict(), "text_encoder.pth")

def load_model(model_path):
    return torch.load(model_path)


def text_encoder(pretrained=False,**kwargs):
    """
    Construct 
    """
    model=Text_Encoder(1024,77,49408,512,8,12)
    if pretrained:
        print('Loading text model')
        model.load_state_dict(torch.load(r'..\model\transformer_block_weight.pth'))     
        print('Loaded successfully')
    return model

'''
# DEMO 
save_model()
model=load_model("text_encoder.pth")
for k,v in model.items():
    print(v.shape)

#DEMO
from images_encoder import ModifiedResNet
img=torch.rand(1,3,224,224)
model1=ModifiedResNet(heads=32,layers=(3, 4, 6, 3) ,output_dim=1024)
img_f=model1(img)

model=Text_Encoder(1024,77,49408,512,8,12)
model.load_state_dict(torch.load(r'transformer_block_weight.pth'))
a=torch.rand(1,77).type(torch.LongTensor)
x = model(a)

print(x.shape)


out=torch.add(img_f,x)
fc=nn.Linear(1024,10)

print(fc(out))
'''
