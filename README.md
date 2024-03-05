## Scratch implementation of Transformers research papers for vision task 

## Papers Implemented

- [x] [Rethinking Spatial Dimensions of Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Heo_Rethinking_Spatial_Dimensions_of_Vision_Transformers_ICCV_2021_paper.pdf)

- [x] [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf)

- [x] [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929.pdf)

- [x] [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030.pdf)

- [ ] [SwinIR: Image Restoration Using Swin Transformer](https://openaccess.thecvf.com/content/ICCV2021W/AIM/papers/Liang_SwinIR_Image_Restoration_Using_Swin_Transformer_ICCVW_2021_paper.pdf)

- [x] [Pyramid Vision Transformer: PVT](https://arxiv.org/pdf/2102.12122.pdf)

- [x] [Deep Vision Transformer: DViT](https://arxiv.org/pdf/2103.11886.pdf)

- [x] [Incorporating Convolution Designs into Visual Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_Incorporating_Convolution_Designs_Into_Visual_Transformers_ICCV_2021_paper.pdf)

- [x] [CONDITIONAL POSITIONAL ENCODINGS FOR VISION TRANSFORMERS](https://openreview.net/pdf?id=3KWnuT-R1bh)

- [x] [CvT: Introducing Convolutions to Vision Transformers](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.pdf)

- [x] [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872.pdf)

## Load The model

```bash
# ceit models
python train.py vit_models.Ceit.ceit_T
python train.py vit_models.Ceit.ceit_S
python train.py vit_models.Ceit.ceit_B

# CPVT models
python train.py vit_models.CPVT.cpvt_ti
python train.py vit_models.CPVT.cpvt_s
python train.py vit_models.CPVT.cpvt_b

# cvt models
python train.py vit_models.CvT.cvt_13
python train.py vit_models.CvT.cvt_21
python train.py vit_models.CvT.cvt_w24

# Deit models
python train.py vit_models.Deit.deit_ti
python train.py vit_models.Deit.deit_s
python train.py vit_models.Deit.deit_b

# DeTr models
python train.py vit_models.DeTR.DETR

# DViT models
python train.py vit_models.DViT.dvit_16b
python train.py vit_models.DViT.dvit_24b
python train.py vit_models.DViT.dvit_32b

# PiT models
python train.py vit_models.PiT.pit_ti
python train.py vit_models.PiT.pit_xs
python train.py vit_models.PiT.pit_s
python train.py vit_models.PiT.pit_b

# PVT models
python train.py vit_models.PVT.PVT_tiny
python train.py vit_models.PVT.PVT_small
python train.py vit_models.PVT.PVT_medium
python train.py vit_models.PVT.PVT_large

# swinT models
python train.py vit_models.swinT.swinT_T
python train.py vit_models.swinT.swinT_S
python train.py vit_models.swinT.swinT_B
python train.py vit_models.swinT.swinT_L

# vit models
python train.py vit_models.Vit.vit_base
python train.py vit_models.Vit.vit_large
python train.py vit_models.Vit.vit_huge
```

