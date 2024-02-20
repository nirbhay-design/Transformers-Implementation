import torch 
import math
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision.transforms as transforms 

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels = 3, patch_size = 8, C = 128):
        super().__init__()
        self.linear = nn.Linear(patch_size * patch_size * in_channels, C)
        self.patch_size = patch_size

    def patch_image(self, x):
        B,C,H,W = x.shape
        H_patch = H // self.patch_size 
        W_patch = W // self.patch_size
        x = x.reshape(B, C, H_patch, self.patch_size, W_patch, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, H_patch * W_patch, self.patch_size * self.patch_size * C)
        return x

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_image(x)
        patch_embeddings = self.linear(x)

        return patch_embeddings
    
class ImageWPosEnc(nn.Module):
    def __init__(self, image_size = 224, in_channels = 3, patch_size = 8, embed_size = 128):
        super().__init__()
        self.image_size = image_size 
        self.in_channels = in_channels
        self.patch_size = patch_size 
        self.embed_size = embed_size 

        assert self.image_size % self.patch_size == 0, "patch_size should be a multiple of image_size"
        # self.device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")
        self.device = torch.device("cpu")

        self.patch_embed = PatchEmbedding(self.in_channels,
                                          self.patch_size,
                                          self.embed_size)

        self.num_embedding = (self.image_size // self.patch_size) ** 2
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
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_dim = embed_dim
    
    def forward(self, x, bias=None):
        # x:shape -> [B, HW/MM, num_heads, MM, head_dim]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        qkt = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.embed_dim)
        if bias == None:
            attention = torch.matmul(F.softmax(qkt, dim = -1), v)
        else:
            attention = torch.matmul(F.softmax(qkt + bias, dim = -1), v)
        attention = self.dropout(self.proj(attention))
        return attention

class WSWMSA(nn.Module):
    def __init__(self, image_resolution, window_size = 2, num_heads = 8, embed_dim = 128, shift_size = 0, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "num heads should be a multiple of embed_dim"
        h,w = image_resolution
        assert h % window_size == 0  and w % window_size == 0, "height and width must be divisible by window size"
        self.head_dim = embed_dim // num_heads
        self.attention = Attention(self.head_dim, dropout)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.bias = nn.Parameter(torch.zeros(2 * window_size - 1, 2 * window_size - 1))

    def patch_format(self, x, image_res):
        B, _, embed_dim = x.shape
        H,W = image_res
        x = x.reshape(B, H, W, embed_dim)
        return x

    def window_partition(self, x, image_res, window_size):
        x = self.patch_format(x, image_res)
        B, H, W, embed_dim = x.shape
        win_h = H // window_size
        win_w = W // window_size
        window_tokens_dim = win_h * win_w
        window_dim = window_size ** 2
        # x-> [B, H/M, M, W/M, M, C] -> [B, H/M, W/M, M,M,C] -> [B, HW/MM, MM, C]
        x = x.reshape(B, win_h, window_size, win_w, window_size, embed_dim).permute(0,1,3,2,4,5)\
            .reshape(B, window_tokens_dim, window_dim, embed_dim)

        return x, window_tokens_dim
    
    def reverse_window(self, x):
        B, _, _, embed_dim = x.shape
        return x.reshape(B, -1, embed_dim)
    
    def cycle_shift(self, x, image_res, shift_size):
        x = self.patch_format(x, image_res)
        B, _, _, embed_dim = x.shape

        x1 = x[:, :shift_size, :shift_size, :]
        x2 = x[:, shift_size:, :shift_size, :]
        x3 = x[:, :shift_size, shift_size:, :]
        x4 = x[:, shift_size:, shift_size:, :]

        win1 = torch.cat([x4, x2], dim = -2)
        win2 = torch.cat([x3, x1], dim = -2)
        cross_window = torch.cat([win1, win2], dim = 1)
        cross_window = cross_window.reshape(B, -1, embed_dim)

        return cross_window
    
    def reverse_cycle_shift(self, x, image_res, shift_size):
        x = self.patch_format(x, image_res)
        B, _, _, embed_dim = x.shape

        x1 = x[:, -shift_size:, -shift_size:, :]
        x2 = x[:, :-shift_size, -shift_size:, :]
        x3 = x[:, -shift_size:, :-shift_size, :]
        x4 = x[:, :-shift_size, :-shift_size, :]
        
        win1 = torch.cat([x1, x3], dim = -2)
        win2 = torch.cat([x2, x4], dim = -2)
        orig_win = torch.cat([win1, win2], dim = 1)
        orig_win = orig_win.reshape(B, -1, embed_dim)

        return orig_win
    
    def window_attention(self, x):
        B = x.shape[0]
        x, window_tokens_dim = self.window_partition(x, self.image_resolution, self.window_size)
        window_dim = self.window_size ** 2
        # x -> [B, HW/MM, MM, heads, head_dim] -> [B, HW/MM, heads, MM, head_dim]
        x = x.reshape(B, window_tokens_dim, window_dim, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        x = self.attention(x)
        # x -> [B, HW/MM, MM, heads, head_dim] -> [B, HW/MM, MM, heads*head_dim] 
        x = x.permute(0,1,3,2,4).flatten(-2)
        # [B, HW, embed_dim]
        x = self.reverse_window(x)
        return x
    
    def shifted_window_attention(self, x):
        shifted_x = self.cycle_shift(x, self.image_resolution, shift_size = self.shift_size)

        

        reverse_shift_x = self.reverse_cycle_shift(x, self.image_resolution, shift_size = self.shift_size)

    def forward(self, x):
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
    
class PatchMerging(nn.Module):
    def __init__(self, image_resolution, dim):
        super().__init__()
        self.image_resolution = image_resolution
        self.c = dim
        self.layer_norm = nn.LayerNorm(4 * self.c)
        self.linear = nn.Linear(4 * self.c, 2 * self.c)

    def forward(self, x):
        # x:shape => [B, H * W, C]
        B, _, C = x.shape
        H, W = self.image_resolution
        x = x.reshape(B, H, W, C)
        x1 = x[:, 0::2, 0::2, :]
        x2 = x[:, 1::2, 0::2, :]
        x3 = x[:, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, :]
        x = torch.cat([x1,x2,x3,x4], dim = -1)
        x = x.reshape(B, -1, 4*C)
        norm_x = self.layer_norm(x)
        linear_x = self.linear(norm_x)

        return linear_x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(embed_dim=dim,
                       hidden_dim=dim,
                       dropout = dropout)
        self.wmsa = None
        self.msa = None

    def forward(self, x):
        pass       

if __name__ == "__main__":
    a = torch.rand(2,3,224,224)
    patch_size = 4
    pe = PatchEmbedding(in_channels=3,
                        patch_size=patch_size,
                        C=128)
    final_patch = pe(a)
    print(final_patch.shape)
    window_size = 4
    wmsa = WSWMSA(image_resolution= (a.shape[2]//patch_size, a.shape[3]//patch_size),
                window_size=window_size,
                num_heads = 8,
                embed_dim = 128,
                dropout=0.3)
    wattn = wmsa(final_patch)
    print(wattn.shape)
    # pm = PatchMerging(image_resolution= (),
    #                   C=128)
    # out = pm(final_patch)
    # print(out.shape)
    
    