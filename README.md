<h2 align="center">Vision Transformers with Self-Distilled Registers
<h5 align="center"> If you like our PH-Reg, please give us a star ⭐ on GitHub for the latest update~
</h2>

This repository contains the official PyTorch implementation for our **NeurIPS 2025** paper, [Vision Transformers with Self-Distilled Registers](https://arxiv.org/abs/2505.21501).

## Environment Requirements

To train PH-Reg, please install the following packages. We used `Python 3.10` in our experiments.

```
pip install -r requirements_eval.txt
pip install numpy==1.26.4
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
```

## Training
Please download the Flickr30k dataset from https://shannon.cs.illinois.edu/DenotationGraph/

For a single GPU, please run:
```
python3 distill_main.py --data_root $YOUR_Flickr_PATH$ -- save_dir $YOUR_CHECKPOINT_PATH$ --pretrained_path 'facebook/dinov2-base'
```
For multiple GPUs, please run:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --mixed_precision='bf16' distill_main.py --data_root $YOUR_Flickr_PATH$ -- save_dir $YOUR_CHECKPOINT_PATH$ --pretrained_path 'facebook/dinov2-base'
```


## Citation

If you find our project useful, please consider citing our paper 📝 and giving a star ⭐.

```bibtex
@misc{chen2025visiontransformersselfdistilledregisters,
      title={Vision Transformers with Self-Distilled Registers}, 
      author={Yinjie Chen and Zipeng Yan and Chong Zhou and Bo Dai and Andrew F. Luo},
      year={2025},
      eprint={2505.21501},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.21501}, 
}
```

## Acknowledgments

We gratefully thank the authors of [CLIP](https://github.com/openai/CLIP), [SCLIP](https://github.com/wangf3014/SCLIP), [ClearCLIP](https://github.com/mc-lan/ClearCLIP), [NACLIP](https://github.com/sinahmr/NACLIP), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [DINOv2](https://github.com/facebookresearch/dinov2) on which our code is based.