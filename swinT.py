import torch 
import math
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
import torchvision.transforms as transforms 
import warnings
warnings.filterwarnings("ignore")

def getHW(size):
    if isinstance(size, int):
        return size, size 
    elif isinstance(size, list) or isinstance(size, tuple):
        return size

class PatchEmbedding(nn.Module):
    def __init__(self, image_resolution, in_channels = 3, patch_size = 8, C = 128):
        super().__init__()
        H, W = image_resolution
        self.patch_size_h, self.patch_size_w = getHW(patch_size)
        assert H % self.patch_size_h == 0 and W % self.patch_size_w == 0, "patch size should be divisible by input height and width"
        self.linear = nn.Linear(self.patch_size_h * self.patch_size_w * in_channels, C)
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

class Attention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, x, bias=None, mask = None):
        # x:shape -> [B, HW/MM, num_heads, MM, head_dim]
        # mask:shape -> [1, HW/MM, MM, MM]
        # bias:shape -> [MM, MM]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        qkt = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.embed_dim)
        if bias is not None:
            qkt = qkt + bias 
        if mask is not None:
            qkt = qkt + mask.unsqueeze(2)
        attention = torch.matmul(F.softmax(qkt, dim = -1), v)
        return attention

class SWMSA(nn.Module):
    def __init__(self, image_resolution, window_size = 2, num_heads = 8, embed_dim = 128, shift_size = 0, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "num heads should be a multiple of embed_dim"
        h,w = image_resolution
        window_size_h, window_size_w = getHW(window_size)
        shift_size_h, shift_size_w = getHW(shift_size)
        assert h % window_size_h == 0  and w % window_size_w == 0, f"height: {h} and width: {w} must be divisible by window size: {window_size_h}, {window_size_w}"
        assert shift_size_h <= window_size_h and shift_size_w <= window_size_w, "shift_size should be less than window_size"
        self.head_dim = embed_dim // num_heads
        self.attention = Attention(self.head_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.window_size = window_size
        self.shift_size = shift_size

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.relative_position_bias_table = nn.Parameter(
             torch.zeros((2 * window_size_h - 1) * (2 * window_size_w - 1), num_heads)
        ) # 2*Wh-1 * 2*Ww-1, nH

        coords_h = torch.arange(window_size_h)
        coords_w = torch.arange(window_size_w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def patch_format(self, x, image_res):
        B, _, embed_dim = x.shape
        H,W = image_res
        x = x.reshape(B, H, W, embed_dim)
        return x

    def window_partition(self, x, window_size): # x_shape: [B, H, W, dim]
        B, H, W, embed_dim = x.shape
        window_size_h, window_size_w = getHW(window_size)
        win_h = H // window_size_h
        win_w = W // window_size_w
        window_tokens_dim = win_h * win_w
        window_dim = window_size_h * window_size_w
        # x-> [B, H/M, M, W/M, M, C] -> [B, H/M, W/M, M,M,C] -> [B, HW/MM, MM, C]
        x = x.reshape(B, win_h, window_size_h, win_w, window_size_w, embed_dim).permute(0,1,3,2,4,5)\
            .reshape(B, window_tokens_dim, window_dim, embed_dim)

        return x
    
    def reverse_window(self, x):
        B, _, _, embed_dim = x.shape
        return x.reshape(B, -1, embed_dim)
    
    def cycle_shift(self, x, shift_size):# x_shape: [B, H, W, dim]
        B, _, _, embed_dim = x.shape
        shift_size_h, shift_size_w = getHW(shift_size)
        x1 = x[:, :shift_size_h, :shift_size_w, :]
        x2 = x[:, shift_size_h:, :shift_size_w, :]
        x3 = x[:, :shift_size_h, shift_size_w:, :]
        x4 = x[:, shift_size_h:, shift_size_w:, :]

        win1 = torch.cat([x4, x2], dim = -2)
        win2 = torch.cat([x3, x1], dim = -2)
        cross_window = torch.cat([win1, win2], dim = 1)
        cross_window = cross_window.reshape(B, -1, embed_dim)

        return cross_window
    
    def reverse_cycle_shift(self, x, shift_size):# x_shape: [B, H, W, dim]
        B, _, _, embed_dim = x.shape
        shift_size_h, shift_size_w = getHW(shift_size)
        x1 = x[:, -shift_size_h:, -shift_size_w:, :]
        x2 = x[:, :-shift_size_h, -shift_size_w:, :]
        x3 = x[:, -shift_size_h:, :-shift_size_w, :]
        x4 = x[:, :-shift_size_h, :-shift_size_w, :]
        
        win1 = torch.cat([x1, x3], dim = -2)
        win2 = torch.cat([x2, x4], dim = -2)
        orig_win = torch.cat([win1, win2], dim = 1)
        orig_win = orig_win.reshape(B, -1, embed_dim)

        return orig_win
    
    def window_attention(self, x, mask=None):
        x = self.patch_format(x, self.image_resolution)
        x = self.window_partition(x, self.window_size)
        B, window_tokens_dim, window_dim, _ = x.shape
        # x -> [B, HW/MM, MM, heads, head_dim] -> [B, HW/MM, heads, MM, head_dim]
        x = x.reshape(B, window_tokens_dim, window_dim, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        
        window_size_h, window_size_w = getHW(self.window_size)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            window_size_h * window_size_w, window_size_h * window_size_w, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        
        x = self.attention(x, mask=mask, bias = relative_position_bias)
        # x -> [B, HW/MM, MM, heads, head_dim] -> [B, HW/MM, MM, heads*head_dim] 
        x = x.permute(0,1,3,2,4).flatten(-2)
        # [B, HW, embed_dim]
        x = self.reverse_window(x)
        return x
    
    def create_mask(self, image_resolution, window_size, shift_size):
        H,W = image_resolution
        image_mask = torch.zeros(1, H, W, 1)
        window_size_h, window_size_w = getHW(window_size)
        shift_size_h, shift_size_w = getHW(shift_size)
        h_slices = (slice(0, -window_size_h),
                    slice(-window_size_h, -shift_size_h),
                    slice(-shift_size_h, None))
        w_slices = (slice(0, -window_size_w),
                    slice(-window_size_w, -shift_size_w),
                    slice(-shift_size_w, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                image_mask[:,h,w,:] = cnt
                cnt += 1

         # size: [1, HW/MM, MM, 1] -> [1, HW/MM, MM]
        mask_window = self.window_partition(image_mask, window_size).reshape(1, -1, window_size_h * window_size_w)
        # target_dim = [1, HW/MM, 1, MM, MM]
        mask_window = mask_window.unsqueeze(2) - mask_window.unsqueeze(3) # size: [1, HW/MM, MM, MM]
        mask_window[mask_window != 0] = float(-100)
        # print(mask_window.unique())
        return mask_window
        
    def shifted_window_attention(self, x):
        x = self.patch_format(x, self.image_resolution)
        shifted_x = self.cycle_shift(x, shift_size = self.shift_size)
        mask = self.create_mask(self.image_resolution,
                                self.window_size,
                                self.shift_size)
        # mask based window attention
        shifted_attention = self.window_attention(shifted_x, mask=mask)

        shifted_attention = self.patch_format(shifted_attention, self.image_resolution)
        reverse_shift_x = self.reverse_cycle_shift(shifted_attention, shift_size = self.shift_size)

        return reverse_shift_x
    
    def forward(self, x):
        if self.shift_size == 0:
            # simple window attention
            x = self.window_attention(x)
        else:
            # shfited window attention
            x = self.shifted_window_attention(x)
        x = self.dropout(self.proj(x))
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
    def __init__(self, image_resolution, num_heads, window_size, shift_size, dim, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(embed_dim=dim,
                       hidden_dim=dim,
                       dropout = dropout)
        self.wmsa = SWMSA(image_resolution=image_resolution,
                          window_size=window_size,
                          num_heads=num_heads,
                          embed_dim=dim,
                          shift_size=0,
                          dropout = dropout
                        )
        self.swmsa = SWMSA(image_resolution=image_resolution,
                         window_size=window_size,
                         num_heads=num_heads,
                         shift_size=shift_size,
                         embed_dim=dim,
                         dropout=dropout)

    def forward(self, x):
        x = self.wmsa(self.ln1(x)) + x   
        x = self.mlp(self.ln1(x)) + x
        x = self.swmsa(self.ln2(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x
    
class SwinTransformer(nn.Module):
    def __init__(self, 
                 image_resolution, 
                 num_classes, 
                 patch_size, 
                 num_heads, 
                 window_size, 
                 shift_size, 
                 embed_dim, 
                 layers, 
                 dropout=0.0):
        super().__init__()
        assert len(num_heads) == len(layers), "number of heads and layer should have same dimension for one-one match"
        self.image_patch_embedding = PatchEmbedding(image_resolution = image_resolution,
                                                    in_channels=3,
                                                    patch_size = patch_size, 
                                                    C = embed_dim)# return image with image_size image_resolution // patch_size
        self.image_resolution = self.img_res(image_resolution, patch_size)
        self.stb1 = nn.ModuleList([
                SwinTransformerBlock(
                    image_resolution=self.image_resolution,
                    num_heads = num_heads[0],
                    window_size = window_size,
                    shift_size = shift_size,
                    dim = embed_dim,
                    dropout = dropout
                ) for _ in range(layers[0])])
        
        self.patch_merging1 = PatchMerging(self.image_resolution, embed_dim) # return height and width divided by 2 and channel multiplied by 2
        self.image_resolution = self.img_res(self.image_resolution, 2)

        self.stb2 = nn.ModuleList([
            SwinTransformerBlock(
                image_resolution=self.image_resolution,
                num_heads = num_heads[1],
                window_size = window_size,
                shift_size = shift_size,
                dim = 2 * embed_dim
            ) for _ in range(layers[1])
        ])

        self.patch_merging2 = PatchMerging(self.image_resolution, 2 * embed_dim)
        self.image_resolution = self.img_res(self.image_resolution, 2)

        self.stb3 = nn.ModuleList([
            SwinTransformerBlock(
                image_resolution=self.image_resolution,
                num_heads = num_heads[2],
                window_size = window_size,
                shift_size = shift_size,
                dim = 4 * embed_dim
            ) for _ in range(layers[2])
        ])

        self.patch_merging3 = PatchMerging(self.image_resolution, 4 * embed_dim)
        self.image_resolution = self.img_res(self.image_resolution, 2)

        self.stb4 = nn.ModuleList([
            SwinTransformerBlock(
                image_resolution=self.image_resolution,
                num_heads = num_heads[3],
                window_size = window_size,
                shift_size = shift_size,
                dim = 8 * embed_dim
            ) for _ in range(layers[3])
        ])
        self.silu = nn.SiLU()
        self.fc = nn.Linear(embed_dim * 8, num_classes)
        
    def img_res(self, res, divisor):
        div_h, div_w = getHW(divisor)
        return [res[0] // div_h, res[1] // div_w]
        
    def forward(self, x):
        x = self.image_patch_embedding(x)

        for module in self.stb1:
            x = module(x)

        x = self.patch_merging1(x)
        for module in self.stb2:
            x = module(x)
        
        x = self.patch_merging2(x)
        for module in self.stb3:
            x = module(x)

        x = self.patch_merging3(x)
        for module in self.stb4:
            x = module(x)

        B, _, dim = x.shape
        h,w = self.image_resolution
        x = x.reshape(B, h, w, dim).permute(0, 3, 1, 2)
        x = F.adaptive_avg_pool2d(x, output_size = (1,1))
        x = x.flatten(1)
        x = self.fc(self.silu(x))
        
        return x
    
def swinT_T(image_resolution, num_classes):
    layers = [2, 2, 6, 2]
    embed_dim = 96
    head1_dim = embed_dim // 32
    num_heads = [head1_dim, head1_dim*2, head1_dim*4, head1_dim*8]
    patch_size = (4, 4)
    window_size = (7, 7)
    shift_size = [i // 2 for i in window_size]
    dropout = 0.6
    swint_t = SwinTransformer(
        image_resolution= image_resolution,
        num_classes=num_classes,
        patch_size=patch_size,
        num_heads = num_heads,
        window_size = window_size,
        shift_size = shift_size,
        embed_dim = embed_dim,
        layers = layers,
        dropout = dropout
    )

    return swint_t

def swinT_S(image_resolution, num_classes):
    layers = [2, 2, 18, 2]
    embed_dim = 96
    head1_dim = embed_dim // 32
    num_heads = [head1_dim, head1_dim*2, head1_dim*4, head1_dim*8]
    patch_size = (4, 4)
    window_size = (7, 7)
    shift_size = [i // 2 for i in window_size]
    dropout = 0.6
    swint_s = SwinTransformer(
        image_resolution= image_resolution,
        num_classes=num_classes,
        patch_size=patch_size,
        num_heads = num_heads,
        window_size = window_size,
        shift_size = shift_size,
        embed_dim = embed_dim,
        layers = layers,
        dropout = dropout
    )

    return swint_s

def swinT_B(image_resolution, num_classes):
    layers = [2, 2, 18, 2]
    embed_dim = 128
    head1_dim = embed_dim // 32
    num_heads = [head1_dim, head1_dim*2, head1_dim*4, head1_dim*8]
    patch_size = (4, 4)
    window_size = (7, 7)
    shift_size = [i // 2 for i in window_size]
    dropout = 0.6
    swint_b = SwinTransformer(
        image_resolution= image_resolution,
        num_classes=num_classes,
        patch_size=patch_size,
        num_heads = num_heads,
        window_size = window_size,
        shift_size = shift_size,
        embed_dim = embed_dim,
        layers = layers,
        dropout = dropout
    )

    return swint_b

def swinT_L(image_resolution, num_classes):
    layers = [2, 2, 18, 2]
    embed_dim = 192
    head1_dim = embed_dim // 32
    num_heads = [head1_dim, head1_dim*2, head1_dim*4, head1_dim*8]
    patch_size = (4, 4)
    window_size = (7, 7)
    shift_size = [i // 2 for i in window_size]
    dropout = 0.6
    swint_l = SwinTransformer(
        image_resolution= image_resolution,
        num_classes=num_classes,
        patch_size=patch_size,
        num_heads = num_heads,
        window_size = window_size,
        shift_size = shift_size,
        embed_dim = embed_dim,
        layers = layers,
        dropout = dropout
    )

    return swint_l


if __name__ == "__main__":
    image_res = (224,224)
    num_classes = 1000

    swin = swinT_T(image_res, num_classes)
    params = lambda x: sum([i.numel() for i in x.parameters()])
    print(params(swin))
    print(swin(torch.rand(2,3,*image_res)).shape)
    