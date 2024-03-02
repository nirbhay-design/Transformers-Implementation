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


class PatchEmbedding(nn.Module):
    def __init__(self, image_size, in_channels = 3, patch_size = 8, embed_size = 128):
        super().__init__()
        self.patch_size_h, self.patch_size_w = getHW(patch_size)
        h,w = getHW(image_size)
        assert h % self.patch_size_h == 0 and w % self.patch_size_w == 0, "image_size should be divisible by patch_size"
        self.linear = nn.Linear(self.patch_size_h * self.patch_size_w * in_channels, embed_size)
        self.patch_size = patch_size

    def patch_image(self, x):
        B,C,H,W = x.shape
        H_patch = H // self.patch_size_h 
        W_patch = W // self.patch_size_w
        x = x.reshape(B, C, H_patch, self.patch_size_h, W_patch, self.patch_size_w)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, H_patch * W_patch, self.patch_size_h * self.patch_size_w * C)
        return x

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_image(x)
        patch_embeddings = self.linear(x)

        return patch_embeddings
    
class PatchEmbeddingConv(nn.Module):
    def __init__(self, image_size, in_channels = 3, patch_size = 8, embed_size = 128):
        super().__init__()
        self.patch_size_h, self.patch_size_w = getHW(patch_size)
        h,w = getHW(image_size)
        assert h % self.patch_size_h == 0 and w % self.patch_size_w == 0, "image_size should be divisible by patch_size"
        self.patch_conv = nn.Conv2d(in_channels, embed_size, kernel_size=3, padding = 1, stride = (self.patch_size_h, self.patch_size_w))
        self.patch_size = patch_size
        self.embed_dim = embed_size

    def forward(self, x):
        # x: [B, C, H, W]
        B,C,H,W = x.shape
        x = self.patch_conv(x) # [B, embed_dim, H/patch_size_h, W/patch_size_w]
        x = x.permute(0, 2, 3, 1).reshape(B, -1, self.embed_dim)
        return x
    
class PEG(nn.Module):
    def __init__(self, image_size, kernel_size, embed_dim):
        super().__init__()
        self.f = nn.Conv2d(embed_dim, embed_dim, kernel_size=kernel_size, padding = (kernel_size - 1) // 2, groups=embed_dim)
        self.image_size = image_size

    def forward(self, x):
        # x: [B, HW, C]
        B, HW, C = x.shape
        x = x.reshape(B, self.image_size[0], self.image_size[1], C).permute(0, 3, 1, 2)
        x = self.f(x)
        x = x.permute(0, 2, 3, 1).reshape(B, HW, C)
        return x

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim = -1)  
        attention = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.embed_dim), dim = -1), v)
        return attention

class MSA(nn.Module):
    def __init__(self, num_heads = 8, embed_dim = 128, dropout = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "num heads should be a multiple of embed_dim"
        self.head_dim = embed_dim // num_heads
        self.attention = Attention(self.head_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, tokens, embed_dim = x.shape
        x = x.reshape(B, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = self.attention(x) # B, num_heads, tokens, head_dim
        x = x.permute(0, 2, 1, 3).reshape(B, tokens, embed_dim)
        x = self.proj_drop(self.proj(x))
        return x   

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, heads, embed_dim, mlp_hidden_dim = 256, dropout=0.0):
        super().__init__()

        self.msa = MSA(heads, embed_dim, dropout)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        int_x = self.dropout(self.msa(self.layer_norm(x)) + x)
        final_x = self.mlp(self.layer_norm(int_x)) + int_x

        return final_x
    
class CPVT(nn.Module):
    def __init__(self, 
                 image_size = 224, 
                 in_channels = 3, 
                 patch_size = 8, 
                 encoder_layers = 6, 
                 msa_heads = 8, 
                 embed_dim = 128, 
                 hidden_dim = 256, 
                 num_class = 10,
                 dropout=0.0):
        
        super().__init__()

        self.patch_creation = PatchEmbeddingConv(image_size,
                                                 in_channels,
                                                 patch_size,
                                                 embed_dim)
        
        self.transformer_encoder1 = TransformerEncoder(msa_heads, embed_dim, hidden_dim, dropout)

        img_h, img_w = getHW(image_size)
        p_h, p_w = getHW(patch_size)
        self.image_size = (img_h // p_h, img_w // p_w)
        self.peg = PEG(image_size = self.image_size, kernel_size=3, embed_dim = embed_dim)

        self.transformer_encoders = nn.Sequential(
            *[TransformerEncoder(msa_heads, embed_dim, hidden_dim, dropout) for _ in range(encoder_layers - 1)]
        )

        self.adaptive_avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.classification = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        x = self.patch_creation(x)
        te1 = self.transformer_encoder1(x)
        pe = self.peg(te1)
        te1 = te1 + pe
        x = self.transformer_encoders(te1)
        B, _, embed_dim = x.shape
        x = x.reshape(B, *self.image_size, embed_dim).permute(0, 3, 1, 2)
        x = self.adaptive_avg_pooling(x).flatten(1)
        x = self.classification(x)
        return x
    
def cpvt_ti(image_size, num_class):
    cpvt = CPVT(image_size=image_size,
              in_channels=3,
              patch_size=16,
              encoder_layers=12,
              msa_heads=3,
              embed_dim=192,
              hidden_dim=192,
              num_class=num_class,
              dropout=0.6)
    return cpvt

def cpvt_s(image_size, num_class):
    cpvt = CPVT(image_size=image_size,
              in_channels=3,
              patch_size=16,
              encoder_layers=12,
              msa_heads=6,
              embed_dim=384,
              hidden_dim=384,
              num_class=num_class,
              dropout=0.6)
    return cpvt

def cpvt_b(image_size, num_class):
    cpvt = CPVT(image_size=image_size,
              in_channels=3,
              patch_size=16,
              encoder_layers=12,
              msa_heads=12,
              embed_dim=768,
              hidden_dim=768,
              num_class=num_class,
              dropout=0.6)
    return cpvt

if __name__ == "__main__":
    a = torch.rand(2,3,224,224)
    cpvt = cpvt_b(224, 1000)

    out = cpvt(a)
    params = lambda x: sum([y.numel() for y in x.parameters()])
    # print(cpvt)
    print(params(cpvt))
    print(out.shape)
    