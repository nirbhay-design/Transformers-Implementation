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
        self.pos_encoding = nn.Parameter(torch.rand(1, self.num_embedding, self.embed_size))

    def forward(self, x):
        B,_,_,_ = x.shape         
        patch_embeddings = self.patch_embed(x)
        pos_encoding = self.pos_encoding.repeat(B, 1, 1)
        embeddings = patch_embeddings + pos_encoding 

        return embeddings
    
class SpatialReduction(nn.Module):
    def __init__(self, image_res, reduction_ratio, embed_dim):
        super().__init__()
        self.image_res = getHW(image_res)
        self.rr = getHW(reduction_ratio)
        assert self.image_res[0] % self.rr[0] == 0 and self.image_res[1] % self.rr[1] == 0, "image_size should be divisible by reduction ratio"
        self.reduction_fc = nn.Linear(self.rr[0] * self.rr[1] * embed_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x.shape -> [B, HW, embed_dim]
        B, HW, embed_dim = x.shape
        x = x.reshape(B, self.image_res[0], self.image_res[1], embed_dim)
        x = x.reshape(B, self.image_res[0] // self.rr[0], self.rr[0], self.image_res[1] // self.rr[1], self.rr[1], embed_dim).permute(0, 1, 3, 2, 4, 5).reshape(B, -1, self.rr[0] * self.rr[1] * embed_dim) #[B, HW/rr, rrc]
        x = self.reduction_fc(x)
        x = self.ln(x)
        return x
    
class SpatialReductionConv(nn.Module):
    def __init__(self, image_res, reduction_ratio, embed_dim):
        super().__init__()
        self.image_res = getHW(image_res)
        self.rr = getHW(reduction_ratio)
        assert self.image_res[0] % self.rr[0] == 0 and self.image_res[1] % self.rr[1] == 0, "image_size should be divisible by reduction ratio"
        self.reduction_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding = 1, stride = reduction_ratio)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x.shape -> [B, HW, embed_dim]
        B, HW, embed_dim = x.shape
        # x.shape -> [B, embed_dim, H, W]
        x = x.reshape(B, self.image_res[0], self.image_res[1], embed_dim).permute(0, 3, 1, 2)
        x = self.reduction_conv(x)
        x = x.permute(0, 2, 3, 1).reshape(B, -1, embed_dim)
        x = self.ln(x)
        return x
    
class MSA(nn.Module):
    def __init__(self, num_heads = 8, embed_dim = 128):
        super().__init__()
        assert embed_dim % num_heads == 0, "num heads should be a multiple of embed_dim"
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(self.head_dim, self.head_dim)
        self.k = nn.Linear(self.head_dim, self.head_dim)
        self.v = nn.Linear(self.head_dim, self.head_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)

    def attention(self, q, k, v):
        q, k, v = self.q(q), self.k(k), self.v(v)
        attention = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim), dim = -1), v)
        return attention
    
    def forward(self, q, k, v):
        B, tokens, embed_dim = q.shape
        kB, ktokens, _ = k.shape
        vB, vtokens, _ = v.shape
        q = q.reshape(B, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(kB, ktokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(vB, vtokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = self.attention(q, k, v) # B, num_heads, tokens, head_dim
        x = x.permute(0, 2, 1, 3).reshape(B, tokens, embed_dim)
        x = self.proj(x)
        return x   
    
class SRA(nn.Module):
    def __init__(self, image_res, reduction_ratio, num_heads = 8, embed_dim = 128):
        super().__init__()

        self.image_res = image_res
        self.reduction_ratio = reduction_ratio
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        self.msa = MSA(self.num_heads, self.embed_dim)
        self.spatial_reduction = SpatialReductionConv(self.image_res, self.reduction_ratio, self.embed_dim)

    def forward(self, x):
        sr_x = self.spatial_reduction(x)
        return self.msa(
            x,
            sr_x,
            sr_x
        )

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim_ratio, dropout=0.0):
        super().__init__()
        hidden_dim = int(hidden_dim_ratio * embed_dim)
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
    def __init__(self, image_size, reduction_ratio, heads, embed_dim, hidden_dim_ratio, dropout=0.0):
        super().__init__()

        self.sra = SRA(image_size, reduction_ratio, heads, embed_dim)
        self.mlp = MLP(embed_dim, hidden_dim_ratio, dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        int_x = self.dropout(self.sra(self.layer_norm(x)) + x)
        final_x = self.mlp(self.layer_norm(int_x)) + int_x

        return final_x
    
class PVTStage(nn.Module):
    def __init__(self, image_size, in_channels, patch_size, encoder_layers, reduction_ratio, heads, embed_dim, hidden_dim_ratio, dropout=0.0):
        super().__init__()

        self.image_size = getHW(image_size)
        self.patch_size = getHW(patch_size)
        self.image_with_pos_enc = ImageWPosEnc(self.image_size, in_channels, self.patch_size, embed_dim)
        self.target_image_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoder(self.target_image_size, reduction_ratio, heads, embed_dim, hidden_dim_ratio, dropout) for _ in range(encoder_layers)])
    
    def forward(self, x):
        x = self.image_with_pos_enc(x)
        for layers in self.transformer_encoder:
            x = layers(x)
        B, _, embed_dim = x.shape
        x = x.reshape(B, *self.target_image_size, embed_dim).permute(0, 3, 1, 2)
        return x
    
class PVT(nn.Module):
    def __init__(self, 
                 image_size = 224, 
                 in_channels = 3, 
                 patch_size = [4, 8, 16, 32], 
                 encoder_layers = [4, 6, 8, 10], 
                 msa_heads = [2, 4, 8, 16], 
                 embed_dim = [128, 128, 256, 256], 
                 hidden_dim_ratio = [1,1,1,1], 
                 reduction_ratio = [1,1,1,1],
                 num_class = 10,
                 dropout=0.0):
        
        super().__init__()

        self.stages = []
        self.image_size = getHW(image_size)
        self.in_channels = in_channels
        for i in range(len(patch_size)):
            self.stages.append(
                PVTStage(image_size = self.image_size,
                         in_channels = self.in_channels,
                         patch_size = patch_size[i],
                         encoder_layers=encoder_layers[i],
                         reduction_ratio=reduction_ratio[i],
                         heads = msa_heads[i],
                         embed_dim = embed_dim[i],
                         hidden_dim_ratio=hidden_dim_ratio[i],
                         dropout=dropout)
            )
            self.in_channels = embed_dim[i]
            self.cur_patch_size = getHW(patch_size[i])
            self.image_size = (self.image_size[0] // self.cur_patch_size[0], self.image_size[1] // self.cur_patch_size[1])

        self.stages = nn.ModuleList(self.stages)

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_layer = nn.Linear(embed_dim[-1], num_class)
        self.silu = nn.SiLU()
        
    def forward(self, x):
        feat_maps = []
        for stage in self.stages:
            x = stage(x)
            feat_maps.append(x)
        x = self.fc_layer(self.silu(self.adaptive_avg_pool(x).flatten(1)))
        return x, feat_maps
    
def PVT_tiny(image_size, num_classes):
    pvt = PVT(
        image_size=image_size,
        in_channels=3,
        patch_size=[4,2,2,2],
        encoder_layers = [2,2,2,2],
        msa_heads = [1,2,5,8],
        embed_dim = [64,128,320,512],
        hidden_dim_ratio= [8,8,4,4],
        reduction_ratio = [8,4,2,1],
        num_class = num_classes,
        dropout = 0.6
    )

    return pvt

def PVT_small(image_size, num_classes):
    pvt = PVT(
        image_size=image_size,
        in_channels=3,
        patch_size=[4,2,2,2],
        encoder_layers = [3,3,6,3],
        msa_heads = [1,2,5,8],
        embed_dim = [64,128,320,512],
        hidden_dim_ratio= [8,8,4,4],
        reduction_ratio = [8,4,2,1],
        num_class = num_classes,
        dropout = 0.6
    )

    return pvt

def PVT_medium(image_size, num_classes):
    pvt = PVT(
        image_size=image_size,
        in_channels=3,
        patch_size=[4,2,2,2],
        encoder_layers = [3,3,18,3],
        msa_heads = [1,2,5,8],
        embed_dim = [64,128,320,512],
        hidden_dim_ratio= [8,8,4,4],
        reduction_ratio = [8,4,2,1],
        num_class = num_classes,
        dropout = 0.6
    )

    return pvt

def PVT_large(image_size, num_classes):
    pvt = PVT(
        image_size=image_size,
        in_channels=3,
        patch_size=[4,2,2,2],
        encoder_layers = [3,8,27,3],
        msa_heads = [1,2,5,8],
        embed_dim = [64,128,320,512],
        hidden_dim_ratio= [8,8,4,4],
        reduction_ratio = [8,4,2,1],
        num_class = num_classes,
        dropout = 0.6
    )

    return pvt

if __name__ == "__main__":
    a = torch.rand(2,3,224,224)
    params = lambda x: sum([y.numel() for y in x.parameters()])

    pvt = PVT_large(224, 20)
    print(params(pvt))
    out, feat_maps = pvt(a)
    for i in feat_maps:
        print(i.shape)
    print(out.shape)
    