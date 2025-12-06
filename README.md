<h2 align="center">
Vision Transformers with Self-Distilled Registers (NeurIPS 2025 Spotlight)
</h2>
<h5 align="center"> If you like our PH-Reg, please give us a star ⭐ on GitHub for the latest update~
</h2>

![Teaser Image](images/teaser2.jpg)

This repository contains the official PyTorch implementation for our **NeurIPS 2025** paper, [Vision Transformers with Self-Distilled Registers](https://arxiv.org/abs/2505.21501).

## Environment Requirements

To train PH-Reg, please install the following packages. We used `Python 3.10` in our experiments.

```
pip install -r requirements_eval.txt
pip install matplotlib scipy scikit-image scikit-learn h5py

pip install openmim
mim install mmengine==0.8.4 
mim install mmcv==2.0.1 
mim install mmsegmentation==1.1.1

pip install transformers==4.37.2
pip install accelerate
pip install diffusers
pip install timm

pip install open-clip-torch==2.31.0
pip install imageio
pip install openai-clip
pip install opencv-python

pip install yapf==0.40.1
pip install numpy==1.26.4
```

## Training
Please download the Flickr30k dataset from https://shannon.cs.illinois.edu/DenotationGraph/

**Reminder:** Before starting training, please make sure to check your dataset.
If you’re using text-based images, do not apply the flipping augmentation, as flipping is only appropriate for natural images.

For a single GPU, please run:
```
python3 distill_main.py --data_root $YOUR_Flickr_PATH$ -- save_dir $YOUR_CHECKPOINT_PATH$ --pretrained_path $YOUR_PRETRAINED_PATH$('facebook/dinov2-base', 'ViT-B/16')
```
For multiple GPUs, please run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --mixed_precision='bf16' distill_main.py --data_root $YOUR_Flickr_PATH$ -- save_dir $YOUR_CHECKPOINT_PATH$ --pretrained_path $YOUR_PRETRAINED_PATH$('facebook/dinov2-base', 'ViT-B/16')
```

## Weights
| Model Name  | Link |
|-------------|------|
| OpenAI CLIP | [link](https://drive.google.com/drive/folders/1IfluySwuhvYMoUvtPEI7VxBoXUs-tgCc?usp=sharing) |
| DINOv2      | [link](https://drive.google.com/file/d/1PVKODN62ln7_30nVncojzOr_zcElaYFh/view?usp=sharing)   |

## Demo
We provide [demo code](https://github.com/0raiser0/PH-Reg/blob/main/CLIP/visualization.ipynb) for performing inference and visualization. You can also find a detailed tutorial on the denoising process in the same file.

Before using it, please download the distilled CLIP weights from [link](https://drive.google.com/drive/folders/1IfluySwuhvYMoUvtPEI7VxBoXUs-tgCc?usp=sharing).

## Evaluation
1. Download the distilled CLIP weights

2. Please follow the [MMSeg data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process the datasets.

      **Remember to modify the dataset paths (`data_root`) in the config files in** `./configs/`.

3. To evaluate our approach on a single benchmark, run the following command:
      ```
      python run_eval.py --config ./configs/cfg_{benchmark_name}.py --work-dir ./logs/{benchmark_name}
      ```

## Citation

If you find our project useful, please consider citing our paper 📝 and giving a star ⭐.

```bibtex
@article{chen2025vision,
  title={Vision transformers with self-distilled registers},
  author={Chen, Yinjie and Yan, Zipeng and Zhou, Chong and Dai, Bo and Luo, Andrew F},
  journal={arXiv preprint arXiv:2505.21501},
  year={2025}
}
```

## Acknowledgments

We gratefully thank the authors of [CLIP](https://github.com/openai/CLIP), [SCLIP](https://github.com/wangf3014/SCLIP), [ClearCLIP](https://github.com/mc-lan/ClearCLIP), [NACLIP](https://github.com/sinahmr/NACLIP), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [DINOv2](https://github.com/facebookresearch/dinov2) on which our code is based.
