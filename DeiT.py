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
    
class ImageWPosEnc(nn.Module):
    def __init__(self, image_size = 224, in_channels = 3, patch_size = 8, embed_size = 128):
        super().__init__()
        self.image_size = getHW(image_size) 
        self.in_channels = in_channels
        self.patch_size = getHW(patch_size) 
        self.embed_size = embed_size 

        self.patch_embed = PatchEmbedding(self.image_size,
                                        self.in_channels,
                                        self.patch_size,
                                        self.embed_size)

        self.num_embedding = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.pos_encoding = nn.Parameter(torch.rand(1, self.num_embedding + 2, self.embed_size))
        self.cls_token = nn.Parameter(torch.rand(1, 1, self.embed_size))
        self.distillation_token = nn.Parameter(torch.rand(1, 1, self.embed_size))

    def forward(self, x):
        B,_,_,_ = x.shape         
        patch_embeddings = self.patch_embed(x)
        pos_encoding = self.pos_encoding.repeat(B, 1, 1)
        cls_token = self.cls_token.repeat(B, 1, 1)
        distillation_token = self.distillation_token.repeat(B, 1, 1)
        patch_cls = torch.cat([cls_token, patch_embeddings, distillation_token], dim = 1)
        embeddings = patch_cls + pos_encoding 

        return embeddings

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
    def __init__(self, num_heads = 8, embed_dim = 128):
        super().__init__()
        assert embed_dim % num_heads == 0, "num heads should be a multiple of embed_dim"
        self.head_dim = embed_dim // num_heads
        self.attention = Attention(self.head_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
    
    def forward(self, x):
        B, tokens, embed_dim = x.shape
        x = x.reshape(B, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = self.attention(x) # B, num_heads, tokens, head_dim
        x = x.permute(0, 2, 1, 3).reshape(B, tokens, embed_dim)
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

        self.msa = MSA(heads, embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        int_x = self.dropout(self.msa(self.layer_norm(x)) + x)
        final_x = self.mlp(self.layer_norm(int_x)) + int_x

        return final_x
    
class Deit(nn.Module):
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

        self.patch_creation = ImageWPosEnc(
            image_size = image_size, 
            in_channels = in_channels, 
            patch_size = patch_size, 
            embed_size = embed_dim
        )

        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoder(msa_heads, embed_dim, hidden_dim, dropout) for _ in range(encoder_layers)]
        )

        self.classification = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        x = self.patch_creation(x)
        x = self.transformer_encoder(x)
        cls_x = x[:, 0, :]
        dist_x = x[:, -1, :]
        cls_x, dist_x = self.classification(cls_x), self.classification(dist_x)
        return cls_x, dist_x
    
class DeitLoss(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction = "batchmean") 
        self.lambd = lambd

    def forward(self, cls_token, distillation_token, ground_truth, soft_labels):
        self.ce_loss = self.ce(cls_token, ground_truth)
        self.kl_div_loss = self.kl(
            F.log_softmax(distillation_token, dim = -1),
            F.softmax(soft_labels, dim = -1)
        )
        self.final_loss = self.ce_loss + self.lambd * self.kl_div_loss

        return self.final_loss

if __name__ == "__main__":
    a = torch.rand(2,3,224,224)
    deit = Deit(image_size=224,
              in_channels=3,
              patch_size=16,
              encoder_layers=8,
              msa_heads=8,
              embed_dim=256,
              hidden_dim=256,
              num_class=10,
              dropout=0.6)

    cls, dist = deit(a)
    params = lambda x: sum([y.numel() for y in x.parameters()])
    # print(deit)
    print(params(deit))
    print(cls.shape, dist.shape)
    