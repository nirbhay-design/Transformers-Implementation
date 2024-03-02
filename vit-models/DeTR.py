import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# building positional encoding

class LearnedPosEncoding(nn.Module):
    def __init__(self, in_features=50, feature_encoding=256):
        super().__init__()
        self.xenc = nn.Embedding(in_features, feature_encoding)
        self.yenc = nn.Embedding(in_features, feature_encoding)

    def forward(self, x):
        h,w = x.shape[-2:]
        x_pos = self.xenc(torch.arange(w, device=x.device))
        y_pos = self.yenc(torch.arange(h, device=x.device))

        pos = torch.cat([
            x_pos.unsqueeze(0).repeat(h,1,1),
            y_pos.unsqueeze(1).repeat(1,w,1)
        ], dim=-1).permute(2,0,1).unsqueeze(0).repeat(x.shape[0],1,1,1)
        return pos

class SinPosEncoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        bs, c, h, w = x.shape

        c = c//2

        x_enc = torch.arange(w,device=x.device)
        y_enc = torch.arange(h,device=x.device)

        x_pos_embed = self._get_encoding_one_dim(c,w,x.device).T
        y_pos_embed = self._get_encoding_one_dim(c,h,x.device).T

        pos = torch.cat([
            x_pos_embed.unsqueeze(0).repeat(h,1,1),
            y_pos_embed.unsqueeze(1).repeat(1,w,1)
        ], dim=-1).permute(2,0,1).unsqueeze(0).repeat(x.shape[0],1,1,1)

        return pos

    def _get_encoding_one_dim(self, seqlen, embedlen, device):
        positions = torch.arange(seqlen,dtype=torch.float32,device=device).unsqueeze(1).repeat(1,embedlen)
        div_vec = torch.tensor([10000 ** (2 * i / embedlen) for i in range(embedlen)]).unsqueeze(0).to(device)
        positions = positions/div_vec
        positions[:,0::2] = positions[:,0::2].sin()
        positions[:,1::2] = positions[:,1::2].cos()

        return positions

#building backbone

class Detr_backbone(nn.Module):
    def __init__(self, layers, embed_dim=512):
        super().__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        r50_modules = resnet50._modules
        self.backbone = nn.Sequential(
            *[r50_modules[layer] for layer in layers]
        )
        self._replace(self.backbone, 
                      layer_to_replace_instance=nn.BatchNorm2d, 
                      replace_layer_instance=torchvision.ops.FrozenBatchNorm2d)
        self.reduced_channel = nn.Conv2d(2048, embed_dim, kernel_size=1)

    def forward(self, x):
        backbone_out = self.backbone(x)
        reduced_channel = self.reduced_channel(backbone_out)
        return reduced_channel

    def _replace(self, model, layer_to_replace_instance, replace_layer_instance):
        modules = model._modules
        if len(modules) == 0:
            return
        for key, layer in modules.items():
            if isinstance(layer, layer_to_replace_instance):
                cur_layer_features = modules[key].num_features
                modules[key] = replace_layer_instance(num_features=cur_layer_features)
            else:
                self._replace(modules[key],layer_to_replace_instance, replace_layer_instance)

# creating input + posencoding

class JointIPPE(nn.Module):
    def __init__(self, backbone, pos_encoding):
        super().__init__()

        self.backbone = backbone
        self.pos_encoding = pos_encoding

    def forward(self, x):
        backbone_features = self.backbone(x) # [N, C, H, W]
        pos_encoding = self.pos_encoding(backbone_features) # [N, C, H, W]
        pos_backbone_features = backbone_features + pos_encoding
        return pos_backbone_features.flatten(-2), pos_encoding.flatten(-2) # [N, C, HW]

# feed forward network

class FeedForward(nn.Module):
    def __init__(self, input_dim, embed_dim, out_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, heads, embed_dim, dropout):
        super(TransformerEncoderBlock, self).__init__()
        self.multihead = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)

        self.feedforward = FeedForward(embed_dim, embed_dim*8, embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, query, key, value, pos_query, pos_key):
        query_with_pos = query + pos_query
        key_with_pos = key + pos_key
        out_layer1 = self.layer_norm1(self.dropout(self.multihead(query_with_pos, key_with_pos, value)[0]) + query)
        out_layer2 = self.layer_norm2(out_layer1 + self.dropout(self.feedforward(out_layer1)))

        return out_layer2

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, head, embed_dim, dropout):
        super().__init__()

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(head, embed_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, query, pos):
        output = query
        for layer in self.encoder_layers:
            output = layer(output, output, output, pos, pos)

        return output

class TransformerDecoderBlock(nn.Module):
    def __init__(self, heads, embed_dim, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.masked_multihead_attention = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.encoder_replicate = TransformerEncoderBlock(heads, embed_dim, dropout)

    def _with_pos_enc(self, x, pos):
        return x + pos

    def forward(self, query, memory, pos_query, pos_key):
        query_pos = query + pos_query
        out_layer1 = self.layer_norm(self.dropout(
            self.masked_multihead_attention(
                query_pos,
                query_pos,
                query_pos,
            )[0]) + query)

        out_layer2 = self.encoder_replicate(out_layer1, memory, memory, pos_query, pos_key)

        return out_layer2

class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, heads, embed_dim, dropout):
        super().__init__()

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(heads, embed_dim, dropout)
            for _ in range(n_layers)
        ])

        self.layer_norm = nn.LayerNorm(embed_dim)


    def forward(self, query, memory, pos_query, pos_key):
        output = query

        for layer in self.decoder_layers:
            output = layer(
                output,
                memory,
                pos_query,
                pos_key
            )

        return self.layer_norm(output)

class Transformer(nn.Module):
    def __init__(self, encoder_layers, decoder_layers, encoder_heads, decoder_heads, embed_dim, dropout):
        super().__init__()

        self.encoder = TransformerEncoder(encoder_layers, encoder_heads, embed_dim, dropout)
        self.decoder = TransformerDecoder(decoder_layers, decoder_heads, embed_dim, dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pos_encoding, query_embed):

        # x is the processed feature extracted

        # x shape: [N, C, HW]
        # pos_encoding: [N, C, HW]
        # learned_pos_enc: function

        x = x.transpose(1,2)
        pos_encoding = pos_encoding.transpose(1,2)

        encoder_output = self.encoder(x,pos_encoding)

        query_embed = query_embed.unsqueeze(0).repeat(x.shape[0],1,1)
        object_queries = torch.zeros_like(query_embed,device=query_embed.device)

        decoder_output = self.decoder(
            object_queries,
            encoder_output,
            query_embed,
            pos_encoding
        )

        return decoder_output

class Detr(nn.Module):
    def __init__(self,
                 backbone_layers,
                 num_classes,
                 num_obj_queries,
                 encoder_layers,
                 decoder_layers,
                 encoder_heads,
                 decoder_heads,
                 embed_dim,
                 dropout):
        super().__init__()
        self.num_obj_queries = num_obj_queries
        backbone = Detr_backbone(layers=backbone_layers,embed_dim=embed_dim)
        self.sin_pos_encoding = SinPosEncoding()
        self.joint_vector = JointIPPE(backbone, self.sin_pos_encoding)
        self.query_embed = nn.Embedding(self.num_obj_queries, embed_dim)

        self.transformer = Transformer(
            encoder_layers, decoder_layers, encoder_heads, decoder_heads, embed_dim, dropout
        )

        self.bbox_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Linear(embed_dim, 4)
        )
        self.logits_layer = nn.Linear(embed_dim, num_classes + 1)

    def forward(self, x):
        x, pos_x = self.joint_vector(x)

        x = self.transformer(
            x,
            pos_x,
            self.query_embed.weight
        )

        logits = self.logits_layer(x)
        boxes = self.bbox_layer(x).sigmoid()

        return {'pred_logits':logits, 'pred_boxes':boxes}
    
params = lambda x: torch.tensor([y.numel() for y in x.parameters()]).sum()
    

if __name__ == "__main__":
    detr = Detr(
        backbone_layers=['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4'],
        num_classes=4,
        num_obj_queries=100,
        encoder_layers=6,
        decoder_layers=6,
        encoder_heads=8,
        decoder_heads=8,
        embed_dim=256,
        dropout=0.1
    )

    data = torch.rand(2,3,256,512)
    out_dict = detr(data)
    print(out_dict['pred_logits'].shape, out_dict['pred_boxes'].shape)
