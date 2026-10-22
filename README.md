# Ophthalmic Foundation Model Evaluation

This is the official repository for benchmarking foundation models across fundus and optical coherence tomography (OCT) imaging tasks.

## Structure

```text
.
├── src/
│   ├── train/                    # Fine-tuning and linear-probing scripts
│   ├── extract/                  # Feature extraction scripts
│   ├── robustness_ft/            # Robustness evaluation across devices using images from the same eyes
│   ├── robustness_ft_repeated/   # Robustness evaluation on the same eyes using the same device
│   ├── model/                    # Model adapters used in this study
└── model_weights/                # Pretrained  model weights
```


## Pretrained model weights

Create a local `model_weights/` directory and download each pretrained model from its official source. The pretrained files are not included in this repository.

| Model | Parameters | Pretrained weights |
|---|---:|---|
| DINOv3 ViT-S/16 | ~22M | [Model](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) |
| DINOv3 ViT-B/16 | ~85M | [Model](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) |
| DINOv3 ViT-L/16 | ~300M | [Model](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| RETFound-DINOv2-MEH | ~300M | [Model](https://huggingface.co/YukunZhou/RETFound_dinov2_meh) |
| RETFound-DINOv2-Shanghai | ~300M | [Model](https://huggingface.co/YukunZhou/RETFound_dinov2_shanghai) |
| RETFound | ~300M | [Fundus](https://huggingface.co/YukunZhou/RETFound_mae_natureCFP) · [OCT](https://huggingface.co/YukunZhou/RETFound_mae_natureOCT) |
| RetiZero | ~305M | [Model](https://drive.google.com/file/d/14bMmnefO73_NL1Xc4x0A5qFNbuI7GqKM/view?usp=sharing) |
| FLAIR (retinal ResNet backbone) | ~25M | [Model](https://huggingface.co/jusiro2/FLAIR) |
| VisionFM | ~85M | [Fundus](https://drive.google.com/file/d/13uWm0a02dCWyARUcrCdHZIcEgRfBmVA4/view?usp=sharing) · [OCT](https://drive.google.com/file/d/1o6E-ine2QLx2pxap-c77u-SU0FjxwypA/view?usp=sharing) |
| EyeCLIP | ~115M | [Model](https://drive.google.com/file/d/1kWpbDqFCFt4j8RkYqacV4nl-aCKZfqZr/view?usp=sharing) |
| MIRAGE-Base | ~85M | [Model](https://github.com/j-morano/MIRAGE/releases/download/weights/MIRAGE-Base.pth) |
| MIRAGE-Large | ~300M | [Model](https://github.com/j-morano/MIRAGE/releases/download/weights/MIRAGE-Large.pth) |


## 📝Acknowledgements and Citation

This work builds upon the following open-source projects and pretrained models. We thank the authors for making their code and model weights available:

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [RETFound](https://github.com/rmaphoh/RETFound)
- [RetiZero](https://github.com/LooKing9218/RetiZero)
- [FLAIR](https://github.com/jusiro/FLAIR)
- [VisionFM](https://github.com/ABILab-CUHK/VisionFM)
- [EyeCLIP](https://github.com/Michi-3000/EyeCLIP)
- [MIRAGE](https://github.com/j-morano/MIRAGE)

If you use this evaluation pipeline, please cite this repository and the original publications corresponding to every pretrained model used in your experiments.

