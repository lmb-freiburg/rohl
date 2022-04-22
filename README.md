# RoHL
Code accompanying the paper: [Improving robustness against common corruptions with frequency biased models (ICCV
2021)](https://lmb.informatik.uni-freiburg.de/Publications/2021/SB21b/)

<p align="center">
<img src="https://lmb.informatik.uni-freiburg.de/Publications/2021/SB21b/tradeoff.png" width="500" class="center">
</p>

## Setup
Please install the following packages
* pytorch (>=1.6)
* numpy
* scikit-learn
* pandas

## Evaluation
* Note: the imagenet data directory should have the following structure:
```
 imagenet
 └── train
 └── val
 └── corrupted
     └── brightness
     └── contrast
     └── fog
     └── ...
```


* Download pretrained models: [lf_expert](https://lmb.informatik.uni-freiburg.de/resources/binaries/iccv21_rohl/lf_expert/model_final.pth.tar), [hf_expert](https://lmb.informatik.uni-freiburg.de/resources/binaries/iccv21_rohl/hf_expert/model_final.pth.tar)

* Then run the following command to evaluate: 

```
python train.py ./datasets/imagenet --low-high --evaluate --lf-ckpt ./work_dir/lf_expert/model.pth.tar --hf-ckpt ./work_dir/hf_expert/model.pth.tar -b 1024
```

## Example training:
* Non-TV model:

```
python train.py ./datasets/in-100 --num-classes 100 --arch resnet18 -b 64 --lr 0.025 --id=non_tv_model
```
* TV model: 

```
python train.py ./datasets/in-100 --num-classes 100 --epochs 180 --arch resnet18 -b 64 --lr 0.025 --id=tv_model --tv --num-tv-layers 1
```


## Citation
If you use the code or parts of it in your research, you should cite the aforementioned paper:
```
@InProceedings{SB21b,
  author       = "T. Saikia and C. Schmid and T.Brox",
  title        = "Improving robustness against common corruptions with frequency biased models",
  booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
  year         = "2021",
  url          = "http://lmb.informatik.uni-freiburg.de/Publications/2021/SB21b"
}
```
## Author
Tonmoy Saikia (saikiat@cs.uni-freiburg.de)
