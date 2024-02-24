# MSPI for AVSP
This project provides the code for **'Audio-Visual Saliency Prediction with Multisensory Perception and Integration'**, Image and Vision Computing, 2024. [Paper link](https://www.sciencedirect.com/science/article/pii/S0262885624000581).

## Download dataset
1. [AVAD](https://sites.google.com/site/minxiongkuo/home)
2. [Coutrot databases](http://antoinecoutrot.magix.net/public/databases.html)
3. [DIEM](https://thediemproject.wordpress.com/videos-and%c2%a0data/)
4. [SumMe](http://cvsp.cs.ntua.gr/research/aveyetracking/) ([Original videos](https://gyglim.github.io/me/vsum/index.html#benchmark))
5. [ETMD](http://cvsp.cs.ntua.gr/research/aveyetracking/)

You can download from the above original links or from the [STAViS's resources](http://cvsp.cs.ntua.gr/research/stavis/).

## Download pretrained backbones
* SlowFast, X3D and MViTv2, [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md)
* Uniformer, [Sense-X/UniFormer](https://github.com/Sense-X/UniFormer/tree/main/video_classification)

* VideoSwin, [SwinTransformer/Video-Swin-Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer)

* MorphMLP, [MTLab/MorphMLP](https://github.com/MTLab/MorphMLP)

* S3D, [kylemin/S3D](https://github.com/kylemin/S3D)

* ResNet18-VGGSound, [hche11/VGGSound](https://github.com/hche11/VGGSound)

😊Thank the above researchers for releasing their codes and sharing model weights!!!

The variants in the paper are testified. If you use other variants of the above models, you should change the corresponding .yaml files and settings in config.py.

## Requirements
For PySlowFast installation, you can refer to [this](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md), but there might not be compatible with our code. 

If you use the PySlowFast codes in this repository, partial model codes' connection to Detectron2 is cut, thus you can **ignore the installation for Detectron2**.
```
timm==0.6.12
torch==1.11.0
```

## Training

The dataset directory structure should be 
```misc
dataset/
    video_frames/ 
        .../ (directories of datasets names) 
    video_audio/ 
        .../ (directories of datasets names)
    annotations/ 
        .../ (directories of datasets names) 
    fold_lists/
        *.txt (lists of datasets splits)
```
### Download weight of image saliency encoder
You can download it from [this](https://github.com/oraclefina/MSPI/releases/tag/v1.0.0). The model is first trained on SALICON and then finetuned on MIT1003.
### Set up config.py for training
Set paths to dataset, pretrained weight files and YAML files.
Set selected backbone and more.
The following setting is crucial:
```
_model_name
cfg.DATA.ROOT
_MOTION_WEIGHTS
cfg.MODEL.IMAGE_SALIENCY_ENCODER_WEIGHT
cfg.MODEL.AUDIO_ENCODER_WEIGHT 
.PATH_CFG
```

Then run the code using
```bash
$ python train.py --session_name --split --num_workers --save_ckpt_freq
```

## Testing
Clone this repository and download the three-split weights of our model from this [link](https://github.com/oraclefina/MSPI/releases/tag/v1.0.0). 
Then run the code using 
```bash
$ python inference.py --weight path/to/weight --path_data path/to/dataset --split split/of/dataset 
```

## Evaluiation
The [MATLAB code](https://github.com/cvzoya/saliency/tree/master/code_forMetrics) is used for evaluation.

## Citation
If you think this project is helpful, please feel free to cite our paper:
```
@article{XIE2024104955,
    title = {Audio-visual saliency prediction with multisensory perception and integration},
    journal = {Image and Vision Computing},
    pages = {104955},
    year = {2024},
    issn = {0262-8856},
    doi = {https://doi.org/10.1016/j.imavis.2024.104955},
    url = {https://www.sciencedirect.com/science/article/pii/S0262885624000581},
    author = {Jiawei Xie and Zhi Liu and Gongyang Li and Yingjie Song}
}
```
        
