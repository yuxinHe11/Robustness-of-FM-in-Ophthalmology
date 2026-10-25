# Robustness of Foundation Models to Real-World Imaging Variations in Ophthalmology

This is the official code repository for evaluating the robustness of 13 foundation models using paired fundus and OCT images of the same eyes. Robustness was assessed under same-device and cross-device imaging variations at both the pretrained representation and fine-tuned prediction levels.

## Structure

```text
.
├── datasets/                      
│   ├── lists/                    # Metadata files for training and evaluation
├── model_weights/                # Pretrained  model weights
├── src/
│   ├── train/                    # Fine-tuning
│   ├── extract/                  # Feature extraction
│   ├── robustness_ft/            # Robustness evaluation across devices using images from the same eyes
│   ├── robustness_ft_repeated/   # Robustness evaluation on the same eyes using the same device
│   ├── model/                    # Model adapters used in this study
```

## Data preparation
- **Create training and validation lists:**

In the `datasets/lists/` directory, create `.csv` file to specify the dataset split.
For training, each line in a `.csv` file should contain:
```
<path><label><split>
```
Example:
```
fundus/000001.png,0,train
fundus/000002.png,1,train
fundus/000003.png,0,val
fundus/000004.png,1,test
fundus/000005.png,0,test
```

For the external validation, each line in a `.csv` should contain:
```
<path><label><patient_id><eye><equipment><shooting_date><sex><age>
```
Example:
```
data/evaluation/000001.png,Glaucoma,P000001,R,DeviceA,2020-01-06,Female,72.1
data/evaluation/000002.png,Glaucoma,P000001,R,DeviceB,2020-01-06,Female,72.1
data/evaluation/000003.png,health,P000002,L,DeviceA,2021-02-04,Male,60
data/evaluation/000004.png,health,P000002,L,DeviceB,2021-02-04,Male,60
data/evaluation/000005.png,AMD,P000003,R,DeviceA,2021-03-03,Female,58
data/evaluation/000006.png,AMD,P000003,R,DeviceB,2021-03-03,Female,58
```
## Installation
Install the required Python packages with:
```
pip install -r requirements.txt
```

## Fine-tuning
Before fine-tuning, download the pretrained model weights form the offcial sources listed below and place them in `model_weights/`.

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


The following command uses DINOv3 as an example. Replace the paths and training options as needed:
```
python -m src.train.train_dinov3 \
  --csv data/lists/train.csv \
  --model_dir model_weights/dinov3_vitb16 \
  --model_name dinov3_vitb16 \
  --output_dir outputs/train/dinov3_vitb16
```

## Robustness evaluation
For pretrained models, first extract image embeddings from the robustness evaluation datasets. Run feature extraction separately for the same-device and cross-device datasets by changing the CSV and output paths as needed:
```
python -m src.extract.extract_dinov3 \
  --csv datasets/lists/metadata.csv \
  --model_dir model_weights/dinov3_vitb16 \
  --output_dir outputs/embeddings/dinov3_vitb16
```

After extracting the embeddings for both settings, calculate the Top-1 and Top-5 retrieval recall and the cross-label outranking ratio:
```
python src.extract.metric_pretrain.py \
        --same-device-dir outputs/embeddings_repeated/dinov3_vitb16 \
        --different-device-dir outputs/embeddings_paired/dinov3_vitb16 \
        --output-dir outputs/pretrain_robustness
```

For fine-tuned models, evaluate prediction robustness separately under cross-device and same-device acquisition settings:
```
python -m src.robustness_ft.eval_dinov3 \
  --csv datasets/lists/paired_metadata.csv \
  --checkpoint outputs/train/dinov3_vitb16/best.pt \
  --output_dir outputs/evaluation/cross_device/dinov3_vitb16

python -m src.robustness_ft_repeated.eval_dinov3 \
  --csv datasets/lists/repeated_metadata.csv \
  --checkpoint outputs/train/dinov3_vitb16/best.pt \
  --output_dir outputs/evaluation/same_device/dinov3_vitb16
```

## 📝Acknowledgements and Citation

This work builds upon the following open-source projects and pretrained models. We thank the authors for making their code and model weights available:

- [DINOv3](https://github.com/facebookresearch/dinov3)
- [RETFound](https://github.com/rmaphoh/RETFound)
- [RetiZero](https://github.com/LooKing9218/RetiZero)
- [FLAIR](https://github.com/jusiro/FLAIR)
- [VisionFM](https://github.com/ABILab-CUHK/VisionFM)
- [EyeCLIP](https://github.com/Michi-3000/EyeCLIP)
- [MIRAGE](https://github.com/j-morano/MIRAGE)

If you use this evaluation pipeline, please cite our paper and the original publications corresponding to every pretrained model used in your experiments.

For inquiries of feedback, don't hesitate to contact yuxinhe@seu.edu.cn

