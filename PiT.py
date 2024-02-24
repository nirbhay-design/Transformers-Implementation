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

        # self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        self.device = torch.device("cpu")

        self.patch_embed = PatchEmbedding(self.image_size,
                                        self.in_channels,
                                        self.patch_size,
                                        self.embed_size)

        self.num_embedding = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1])
        self.pos_encoding = nn.Parameter(torch.rand(1, self.num_embedding + 1, self.embed_size, device=self.device))
        self.cls_token = nn.Parameter(torch.rand(1, 1, self.embed_size, device=self.device))

    def forward(self, x):
        B,_,_,_ = x.shape         
        patch_embeddings = self.patch_embed(x)
        pos_encoding = self.pos_encoding.repeat(B, 1, 1)
        cls_token = self.cls_token.repeat(B, 1, 1)
        patch_cls = torch.cat([cls_token, patch_embeddings], dim = 1)
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
    
class PoolingLayer(nn.Module):
    def __init__(self, image_size, embed_dim, kernel_size, stride):
        super().__init__()
        self.image_size = getHW(image_size)
        self.embed_dim = embed_dim

        self.cls_fc = nn.Linear(self.embed_dim, 2 * self.embed_dim)
        self.depth_wise_conv = nn.Conv2d(
            embed_dim, 
            2 * embed_dim, 
            kernel_size = kernel_size, 
            padding = (kernel_size - 1) // 2, 
            stride = stride, 
            groups = embed_dim)
        
    def forward(self, x):
        # x: shape: [B, num_embedding + 1, embed_dim]
        cls_token = x[:, 0:1, :] # [B, 1, embed_dim]
        x = x[:, 1:, :] # [B, num_embedding (h x w), embed_dim]
        B, _, embed_dim = x.shape
        cls_fc = self.cls_fc(cls_token) # [B, 1, 2 * embed_dim]
        x = x.reshape(B, self.image_size[0], self.image_size[1], embed_dim).permute(0, 3, 1, 2)
        conv_x = self.depth_wise_conv(x) # [B, 2 * embed_dim, h/2, w/2]
        conv_x = conv_x.permute(0, 2, 3, 1).reshape(B, -1, 2 * embed_dim)
        cls_conv = torch.cat([cls_fc, conv_x], dim = 1)
        return cls_conv
    
class Pit(nn.Module):
    def __init__(self, 
                 image_size = 224, 
                 in_channels = 3, 
                 patch_size = 8, 
                 stride = 2,
                 msa_heads = [4, 8, 16],
                 depths = [3, 7, 9], 
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
        patch_size = getHW(patch_size)
        image_size = getHW(image_size)

        image_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        transformer_layer_list = []

        for i in range(len(depths)):
            transformer_layer_list.extend(
                [TransformerEncoder(
                    heads = msa_heads[i],
                    embed_dim = embed_dim,
                    mlp_hidden_dim = hidden_dim,
                    dropout = dropout) for _ in range(depths[i])]
            )

            transformer_layer_list.append(PoolingLayer(
                image_size,
                embed_dim,
                kernel_size = stride + 1,
                stride = stride,
            ))  

            image_size = (image_size[0] // 2, image_size[1] // 2)
            embed_dim *= 2
        
        self.transformer_layer_list = nn.ModuleList(transformer_layer_list)
        self.classification = nn.Linear(embed_dim, num_class)

    def forward(self, x):
        x = self.patch_creation(x)
        for layer in self.transformer_layer_list:
            x = layer(x)
        x = x[:,0,:]
        x = x.flatten(1)
        x = self.classification(x)
        return x

if __name__ == "__main__":
    a = torch.rand(2,3,224,224)
    pit = Pit(image_size=224,
              in_channels=3,
              patch_size=14,
              msa_heads=[2, 4, 8],
              stride = 2,
              depths = [2, 6, 4],
              embed_dim=144,
              hidden_dim=144,
              num_class=10,
              dropout=0.6)

    out = pit(a)
    params = lambda x: sum([y.numel() for y in x.parameters()])
    print(pit)
    print(params(pit))
    print(out.shape)
    