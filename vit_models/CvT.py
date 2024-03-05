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

class ConvTokenEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding = (kernel_size - 1) // 2)
        self.ln = nn.LayerNorm(out_channels)

    def forward(self, x):
        # x.shape -> [B, C, H, W] -> [B, HW, C] 
        x = self.conv(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.ln(x)
        return x 

class ConvProjection(nn.Module):
    def __init__(self, channels, kernel_size, stride):
        super().__init__()
        self.dw_conv = nn.Conv2d(channels, 
                                 channels, 
                                 kernel_size = kernel_size, 
                                 padding = (kernel_size - 1) // 2, 
                                 stride = stride,
                                 groups = channels)
        self.bn = nn.BatchNorm2d(channels)
        self.point_conv = nn.Conv2d(channels, channels, kernel_size = 1, padding = 0, stride = 1)

    def forward(self, x):
        # x.shape -> [B, C, H, W] -> [B, HW, C]
        x = self.point_conv(self.bn(self.dw_conv(x)))
        x = x.flatten(2).permute(0, 2, 1)
        return x

class MSA(nn.Module):
    def __init__(self, num_heads = 8, embed_dim = 128, dropout = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "num heads should be a multiple of embed_dim"
        self.head_dim = embed_dim // num_heads
        self.q = nn.Linear(self.head_dim, self.head_dim)
        self.k = nn.Linear(self.head_dim, self.head_dim)
        self.v = nn.Linear(self.head_dim, self.head_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.proj_out = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def attention(self, q, k, v):
        q, k, v = self.q(q), self.k(k), self.v(v)
        attention = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim), dim = -1), v)
        return attention
    
    def forward(self, q, k ,v):
        # x.shape -> [B, HW, C] -> [B, HW, C]
        B, tokens, embed_dim = q.shape
        kB, ktokens, _ = k.shape
        vB, vtokens, _ = v.shape
        q = q.reshape(B, tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(kB, ktokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(vB, vtokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        x = self.attention(q, k, v) # B, num_heads, tokens, head_dim
        x = x.permute(0, 2, 1, 3).reshape(B, tokens, embed_dim)
        x = self.proj_drop(self.proj_out(x))
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
    
class ConvTransformerEncoder(nn.Module):
    def __init__(self, image_size, kernel_size, embed_dim, heads, mlp_hidden_dim_ratio = 2, dropout=0.0):
        super().__init__()

        self.conv_proj_q = ConvProjection(embed_dim, kernel_size = kernel_size, stride = 1)
        self.conv_proj_k = ConvProjection(embed_dim, kernel_size = kernel_size, stride = 2)
        self.conv_proj_v = ConvProjection(embed_dim, kernel_size = kernel_size, stride = 2)
        self.msa = MSA(heads, embed_dim, dropout)
        self.mlp = MLP(embed_dim, int(mlp_hidden_dim_ratio * embed_dim), dropout=0.0)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.image_size = getHW(image_size)
        self.embed_dim = embed_dim

    def forward(self,x):
        # x.shape -> [B, HW, C]
        B, _, C = x.shape
        x2d = x.reshape(B, self.image_size[0], self.image_size[1], C).permute(0, 3, 1, 2)
        proj_q = self.layer_norm(self.conv_proj_q(x2d)) # [B, HW, C]
        proj_k = self.layer_norm(self.conv_proj_k(x2d))
        proj_v = self.layer_norm(self.conv_proj_v(x2d))
        x = self.dropout(self.msa(proj_q, proj_k, proj_v) + x)
        x = self.dropout(self.mlp(self.layer_norm(x)) + x)
        return x # [B, HW, C]

class CVTStage(nn.Module):
    def __init__(self, 
                 image_size,
                 conv_emb_kernel,
                 conv_emb_stride,
                 in_channels,
                 out_channels,
                 conv_proj_kernel,
                 num_heads,
                 mlp_hidden_dim_ratio,
                 num_transformer_layers,
                 dropout = 0.0
                  ):
        super().__init__()
        h,w = getHW(image_size)
        sh, sw = getHW(conv_emb_stride)
        assert h % sh == 0 and w % sw == 0, "image_size should be divisible by conv embed stride"

        self.conv_token_emb = ConvTokenEmbedding(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = conv_emb_kernel, 
            stride = conv_emb_stride
        )

        self.image_size = (h // sh, w // sw)
        
        self.conv_transformer_encoders = nn.ModuleList([
            ConvTransformerEncoder(
                image_size = self.image_size,
                kernel_size = conv_proj_kernel,
                embed_dim = out_channels,
                heads = num_heads,
                mlp_hidden_dim_ratio = mlp_hidden_dim_ratio,
                dropout = dropout
            ) for _ in range(num_transformer_layers) 
        ])

    def forward(self, x):
        # x.shape -> [B, C, H, W] -> [B, embed_dim, H_1, W_1]
         
        x = self.conv_token_emb(x)
        for layer in self.conv_transformer_encoders:
            x = layer(x)

        B, _, embed_dim = x.shape
        x = x.reshape(B, self.image_size[0], self.image_size[0], embed_dim).\
        permute(0, 3, 1, 2)
        return x # [B, embed_dim, H, W]

class CvT(nn.Module):
    def __init__(self, 
                 image_size,
                 in_channels,
                 num_classes,
                 conv_emb_kernels,
                 conv_emb_strides,
                 embed_dims,
                 conv_proj_kernels,
                 num_heads,
                 mlp_hidden_dim_ratios,
                 num_transformer_layers,
                 dropout = 0.0):
        
        super().__init__()

        self.image_size = getHW(image_size)
        self.in_channels = in_channels

        self.stages = nn.ModuleList()
        for i in range(len(embed_dims)):
            cvtstage_i = CVTStage(
                image_size = self.image_size,
                conv_emb_kernel = conv_emb_kernels[i],
                conv_emb_stride = conv_emb_strides[i],
                in_channels = self.in_channels,
                out_channels = embed_dims[i],
                conv_proj_kernel = conv_proj_kernels[i],
                num_heads = num_heads[i],
                mlp_hidden_dim_ratio=mlp_hidden_dim_ratios[i],
                num_transformer_layers = num_transformer_layers[i],
                dropout = dropout
            )
            self.stages.append(cvtstage_i)
            self.image_size = cvtstage_i.image_size
            self.in_channels = embed_dims[i]

        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.classification = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, x):
        for layer in self.stages:
            x = layer(x)
        x = self.adaptive_avg_pool(x).flatten(1)
        x = self.classification(x)
        return x
    
def cvt_13(image_size, num_classes):
    cvt = CvT(
        image_size = image_size,
        in_channels = 3,
        num_classes = num_classes,
        conv_emb_kernels = [7,3,3],
        conv_emb_strides = [4,2,2],
        embed_dims = [64,192,384],
        conv_proj_kernels = [3,3,3],
        num_heads = [1,3,6],
        mlp_hidden_dim_ratios= [4,4,4],
        num_transformer_layers= [1,2,10],
        dropout = 0.5
    )

    return cvt 

def cvt_21(image_size, num_classes):
    cvt = CvT(
        image_size = image_size,
        in_channels = 3,
        num_classes = num_classes,
        conv_emb_kernels = [7,3,3],
        conv_emb_strides = [4,2,2],
        embed_dims = [64,192,384],
        conv_proj_kernels = [3,3,3],
        num_heads = [1,3,6],
        mlp_hidden_dim_ratios= [4,4,4],
        num_transformer_layers= [1,4,16],
        dropout = 0.5
    )

    return cvt 

def cvt_w24(image_size, num_classes):
    cvt = CvT(
        image_size = image_size,
        in_channels = 3,
        num_classes = num_classes,
        conv_emb_kernels = [7,3,3],
        conv_emb_strides = [4,2,2],
        embed_dims = [192,768,1024],
        conv_proj_kernels = [3,3,3],
        num_heads = [3,12,16],
        mlp_hidden_dim_ratios= [4,4,4],
        num_transformer_layers= [2,2,20],
        dropout = 0.5
    )

    return cvt 

if __name__ == "__main__":
    a = torch.rand(2,3,224,224)

    params = lambda x: sum([y.numel() for y in x.parameters()])
    # print(out.shape)

    cvt = cvt_21(image_size = 224, num_classes = 1000)
    
    print(params(cvt))
    print(cvt(a).shape)