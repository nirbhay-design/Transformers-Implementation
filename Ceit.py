import torch 
import math
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision.transforms as transforms 

def getHW(size):
    if isinstance(size, int):
        return size, size
    elif isinstance(size, tuple) or isinstance(size, list):
        return size 

    
class ImageToToken(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 
                              out_channels, 
                              kernel_size=kernel_size, 
                              padding = (kernel_size - 1) // 2, 
                              stride = stride)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size = 3,
                                    stride = 2,
                                    padding = 1)
    def forward(self, x):
        return self.maxpool(self.bn(self.conv(x)))
    
class TokensWPosEnc(nn.Module):
    def __init__(self, image_size = 224, in_channels = 3, out_channels = 32, patch_size = 8, embed_size = 128):
        super().__init__()
        self.image_size = getHW(image_size) 
        self.in_channels = in_channels
        self.patch_size = getHW(patch_size) 
        self.embed_size = embed_size 

        assert self.image_size[0] % self.patch_size[0] ==0 and self.image_size[1] % self.patch_size[1]== 0, "image_size should be divisible by patch_size"

        self.img2token = ImageToToken(in_channels, out_channels, kernel_size = 7, stride = 2)
        self.patch_embed = nn.Conv2d(out_channels, 
                                     embed_size, 
                                     kernel_size = patch_size + 1, 
                                     stride = patch_size, 
                                     padding = patch_size // 2)

        self.result_feat_size = ((self.image_size[0] // self.patch_size[0] // 4), 
                                 (self.image_size[1] // self.patch_size[1] // 4))
        self.num_embedding = self.result_feat_size[0] * self.result_feat_size[1]
        self.pos_encoding = nn.Parameter(torch.rand(1, self.num_embedding + 1, self.embed_size))
        self.cls_token = nn.Parameter(torch.rand(1, 1, self.embed_size))

    def forward(self, x):
        B,_,_,_ = x.shape         
        img_tokens = self.img2token(x)
        patch_embeddings = self.patch_embed(img_tokens).permute(0, 2, 3, 1).reshape(B, -1, self.embed_size)
        pos_encoding = self.pos_encoding.repeat(B, 1, 1)
        cls_token = self.cls_token.repeat(B, 1, 1)
        patch_cls = torch.cat([cls_token, patch_embeddings], dim = 1)
        embeddings = patch_cls + pos_encoding 

        return embeddings
    
class LeFF(nn.Module):
    def __init__(self, image_size, embed_dim, expand_ratio, kernel_size = 3):
        super().__init__()
        self.image_size = getHW(image_size)
        self.ed = embed_dim 
        self.er = expand_ratio
        self.ks = kernel_size
        self.hd = self.er * self.ed

        self.linear_proj = nn.Conv2d(self.ed, self.hd, kernel_size = 1, padding = 0)
        self.dw_conv = nn.Conv2d(self.hd, 
                                 self.hd, 
                                 kernel_size = kernel_size, 
                                 padding = (kernel_size - 1) // 2, 
                                 stride = 1, 
                                 groups = self.hd)
        self.bn1 = nn.BatchNorm2d(self.hd)
        self.bn2 = nn.BatchNorm2d(self.ed)
        self.gelu = nn.GELU()
        self.linear_proj_back = nn.Conv2d(self.hd, self.ed, kernel_size = 1, padding = 0)

    def forward(self, x):
        # x.shape -> [B, N + 1, C]
        B, _, C = x.shape
        cls, features = x[:, 0:1, :], x[:, 1:, :]
        features = features.reshape(B, *self.image_size, C).permute(0, 3, 1, 2)
        features = self.gelu(self.bn1(self.linear_proj(features)))
        features = self.gelu(self.bn1(self.dw_conv(features)))
        features = self.gelu(self.bn2(self.linear_proj_back(features))) # [B, C, H, W]
        features = features.permute(0, 2, 3, 1).reshape(B, -1, C)
        final_tokens = torch.cat([cls, features], dim = 1)
        return final_tokens
    

class MSA(nn.Module):
    def __init__(self, num_heads = 8, embed_dim = 128, dropout = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "num heads should be a multiple of embed_dim"
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(self.head_dim, 3 * self.head_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.proj_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def attention(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = -1)  
        attention = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim), dim = -1), v)
        return attention
    
    def forward(self, x):
        B, tokens, embed_dim = x.shape
        x = x.reshape(B, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = self.attention(x) # B, num_heads, tokens, head_dim
        x = x.permute(0, 2, 1, 3).reshape(B, tokens, embed_dim)
        x = self.proj_drop(self.proj_out(x))
        return x   

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, image_size, heads, embed_dim, expand_ratio, dropout=0.0):
        super().__init__()

        self.msa = MSA(heads, embed_dim, dropout=dropout)
        self.mlp = LeFF(image_size, embed_dim, expand_ratio, kernel_size = 3)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        int_x = self.dropout(self.msa(self.layer_norm(x)) + x)
        final_x = self.mlp(self.layer_norm(int_x)) + int_x

        return final_x
    
class Ceit(nn.Module):
    def __init__(self, 
                 image_size = 224, 
                 in_channels = 3,
                 mid_channels = 32, 
                 patch_size = 8, 
                 encoder_layers = 6, 
                 msa_heads = 8, 
                 embed_dim = 128, 
                 expand_ratio = 4, 
                 num_class = 10,
                 dropout=0.0):
        
        super().__init__()

        self.image_tokens = TokensWPosEnc(image_size,
                                          in_channels,
                                          mid_channels,
                                          patch_size,
                                          embed_dim)
        
        self.image_size = self.image_tokens.result_feat_size

        self.encoder_layers = nn.ModuleList()

        for _ in range(encoder_layers):
            self.encoder_layers.append(
                TransformerEncoder(self.image_size,
                                    msa_heads,
                                    embed_dim,
                                    expand_ratio,
                                    dropout = dropout)
            )

        self.LCA = MSA(msa_heads, embed_dim, dropout = 0)
        self.mlp = MLP(embed_dim, int(expand_ratio * embed_dim), num_class, dropout = 0.0)

    def forward(self, x):
        self.layer_cls = []
        x = self.image_tokens(x)
        self.layer_cls.append(x[:, 0:1, :])
        for layer in self.encoder_layers:
            x = layer(x)
            self.layer_cls.append(x[:, 0:1, :])
        layer_cls = torch.cat(self.layer_cls, dim = 1)
        cls_attention = self.LCA(layer_cls)
        cls_features = cls_attention[:, -1, :]
        cls_layer = self.mlp(cls_features)
        return cls_layer
    
def ceit_T(image_size, num_class):
    ceit = Ceit(
        image_size = image_size,
        in_channels = 3,
        mid_channels=32,
        patch_size = 4,
        encoder_layers = 12,
        msa_heads = 3,
        embed_dim = 192,
        expand_ratio = 4,
        num_class = num_class,
        dropout = 0.6
    )

    return ceit

def ceit_S(image_size, num_class):
    ceit = Ceit(
        image_size = image_size,
        in_channels = 3,
        mid_channels=32,
        patch_size = 4,
        encoder_layers = 12,
        msa_heads = 6,
        embed_dim = 384,
        expand_ratio = 4,
        num_class = num_class,
        dropout = 0.6
    )

    return ceit

def ceit_B(image_size, num_class):
    ceit = Ceit(
        image_size = image_size,
        in_channels = 3,
        mid_channels=32,
        patch_size = 4,
        encoder_layers = 12,
        msa_heads = 12,
        embed_dim = 768,
        expand_ratio = 4,
        num_class = num_class,
        dropout = 0.6
    )

    return ceit

if __name__ == "__main__":
    a = torch.rand(2,3,224,224)
    params = lambda x: sum([y.numel() for y in x.parameters()])
    
    ceit = ceit_B(224, 1000)
    out = ceit(a)
    print(params(ceit))
    print(out.shape)