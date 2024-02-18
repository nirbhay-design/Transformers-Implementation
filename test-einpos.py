import torch 
import einops 

tsr = torch.rand(1,3,85,13,13)

# permute 
tsr1 = einops.rearrange(tsr, "b a p h w -> b a h w p")
print(tsr1.shape)


tsr2 = einops.rearrange(tsr, "b a p h w -> b (a h w) p")
print(tsr2.shape)

tsr3 = einops.rearrange(tsr2, "b (a h w) p -> b a p h w", a = 3, h = 13)
print(tsr3.shape)

tsr = torch.rand(1,3,224,224)

tsr4 = einops.repeat(tsr, "b c h w -> b c (h 2) (w 2)")
print(tsr4.shape)