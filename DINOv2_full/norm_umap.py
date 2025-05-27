import os, sys
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sys.path.append('./')
# sys.path.append('/data/chenyinjie/CYJcode/traindistill')
import argparse
import torch
assert torch.cuda.is_available()
import torch.nn.functional as F
# from torchvision import transforms
# from accelerate import Accelerator
import numpy as np
from torch.utils.data import DataLoader
# Import dataset
from datasets.flickr30k_dataset import prepare_flickr30_dataloader
# Import shiting augmentation functions
# from utils.denoising import prepare_all_offsets, prepare_flip, prepare_patch_idxs, farthest_point_sample
# Import teacher model and student model
from models.teacherDINOv2 import AugDINOv2Base
from models.studentDINOv2 import RegDINOv2Base
import matplotlib.pyplot as plt


def vis_augmented_teacher():
    parser = argparse.ArgumentParser(description="Train a CLIP model with distillation.")
    parser.add_argument("--data_root", type=str, default="/data/chenyinjie/CYJcode/traindistill/Plots/dataset/high_resolution", help="Dataset root directory.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--save_dir", type=str, default="/data/chenyinjie/CYJcode/traindistill/DINOv2/distilledweights", help="Directory to save model checkpoints.")
    # training settting
    parser.add_argument("--unused_param", type=bool, default=True, help="Some parameters in transformer resblocks are not used when fine-tuning.")
    parser.add_argument("--resolution", type=int, default=518, help="Input Image size")
    parser.add_argument("--shift_frac", type=float, default=0.15, help="Shifting fraction used in shifting augmentation")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--end_lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--counts", type=int, default=10, help="Number of sample points for the shifting augmentaion in Teacher Model.")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Model setting
    parser.add_argument("--patch_size", type=int, default=14, help="Model embedding patch size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Model embedding hidden size")
    parser.add_argument("--attn_implementation", type=str, default='eager', help="Attention Implementation, 'eager' or 'sdpa'")
    parser.add_argument("--pretrained_path", type=str, default="/data/chenyinjie/CYJcode/distillation/DistillDINOv2/pretrained/facebook/dinov2-base", help="Teacher and Student model pretrained weight path")
    parser.add_argument("--distilled_path", type=str, default="/data/chenyinjie/CYJcode/traindistill/DINOv2_full/distilledweights/distilled_dinov2_weights_30.pth")
    parser.add_argument("--weight_frozen", type=bool, default=True, help="Freeze models' weights when fine tuning")
    parser.add_argument("--arch", type=str, default="vanilla", help="Model architecture setting as NACLIP")
    parser.add_argument("--num_of_reg", type=int, default=16, help="Number of register tokens.")
    parser.add_argument("--mse_scale", type=float, default=1.0, help="Scale MSELoss")
    args = parser.parse_args()

    args_dict = vars(args)
    
    # Prepare training image data
    # choose optimal mean and std !!
    test_set, shuffle = prepare_flickr30_dataloader(
        args_dict=args_dict,
        mode='test'
    )
    train_dataloader = DataLoader(
        dataset=test_set,
        batch_size=args_dict['batch_size'],
        shuffle=shuffle,
        drop_last=True,
        num_workers=8
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'else')
    
    # initialization Model

    # print(clip_vit.conv1.weight.dtype)
    # Teacher Model
    teacher_model = AugDINOv2Base(
        pretrained_path=args_dict['pretrained_path'],
        attn_implementation=args_dict['attn_implementation'],
        weight_frozen=args_dict['weight_frozen']
    ).to(torch.float32).to(device)
    teacher_model.eval()
    
    # Student Model
    student_model = RegDINOv2Base(
        pretrained_path=args_dict['pretrained_path'],
        attn_implementation=args_dict['attn_implementation'],
        num_registers=args_dict['num_of_reg'],
        weight_frozen=args_dict['weight_frozen']
    ).to(torch.float32)
    student_model.load_state_dict(torch.load(args_dict["distilled_path"]))
    student_model = student_model.to(device)
    student_model.eval()
    
    for original_images, shifted_images, shifted_idxs in train_dataloader:
        original_images = original_images.to(device)
        shifted_images = shifted_images.to(device)
        shifted_idxs = shifted_idxs.to(device)
        
        h, w = shifted_images.shape[-2], shifted_images.shape[-1]
        n_patches = (h//args_dict["patch_size"], w//args_dict["patch_size"])
        
        # with torch.inference_mode():
        #     teacher_img_feats = teacher_model(
        #                 args_dict=args_dict,
        #                 shifted_images=shifted_images,
        #                 shifted_idxs=shifted_idxs
        #             ) # batch_size, num_patches, hidden_size
        #     teacher_img_feats = teacher_img_feats.to(torch.float32)
        # teacher_img_feats = teacher_img_feats.clone()
        
        with torch.inference_mode():
            raw_teach_feats = teacher_model.forward_images(
                images=original_images,
                output_attentions=False,
                output_hidden_states=False
            )
            raw_teach_feats = raw_teach_feats['last_hidden_state'][:, 1:, :].to(torch.float32)
        raw_teach_feats = raw_teach_feats.clone()
            
        with torch.inference_mode():
            student_img_feats = student_model(
                    images=original_images,
                    output_hidden_states=False,
                    output_attentions=False)
            student_img_feats = student_img_feats['last_hidden_state'][:, 1+args_dict['num_of_reg']:, :]
        student_img_feats = student_img_feats.clone()
         
        break
    
    # save_path = os.path.join('/data/chenyinjie/CYJcode/traindistill/MaskCLIP', 'umap.svg')
    # save_path = save_path[:-4] # delete .jpg / .png
    save_path = '/data/chenyinjie/CYJcode/traindistill/DINOv2_full/norm/'
    if len(original_images.shape) == 4:
        raw_img = original_images[0]
        
    visualize_both_feat_umap(
        raw_teacher_feats=raw_teach_feats[0],
        student_feats=student_img_feats[0],
        m=0.1,
        mx=1.0,
        patch_shape=n_patches,
        resolution=(h, w),
        n_components=3,
        save_path=save_path,
    )






"=================="
import numpy as np
import torch
import torch.nn.functional as F
import umap
import matplotlib.pyplot as plt
import numba

def visualize_both_feat_umap(raw_teacher_feats, student_feats, m, mx, patch_shape=(28, 28), resolution=(448, 448), n_components=3, save_path=None):
    """
    参数说明:
    - raw_img: 原始图像 (C, H, W) or (H, W, C)
    - patch_feats: [num_patches, feat_dim] 例如 [784, 768]
    - patch_shape: (h, w)，例如 (28, 28)
    - resolution: 输出图像大小，例如 (448, 448)
    """
    # Step 1: UMAP 降维
    raw_feats = raw_teacher_feats / raw_teacher_feats.norm(dim=-1, keepdim=True)
    raw_feats_np = raw_feats.detach().cpu().numpy()
    
    denoised_feats = student_feats / student_feats.norm(dim=-1, keepdim=True)
    denoised_feats_np = denoised_feats.detach().cpu().numpy()
    
    cat_feats = np.concatenate([raw_feats_np, denoised_feats_np], axis=0)
    
    @numba.njit()
    def mydist(in1, in2):
        return np.arccos(np.sum(in1*in2))
    
    transformer = umap.UMAP(
        n_components=n_components,
        metric=mydist)
    _ = transformer.fit(cat_feats) # [784, 1024]
    raw_embedding = transformer.transform(raw_feats_np) # [784, 512]
    denoised_embedding = transformer.transform(denoised_feats_np) # [784, 512]
    # embedding = transformer.fit_transform(patch_feats_np)

    # Step 2: reshape 为 [28, 28]
    raw_embedding_map = raw_embedding.reshape(patch_shape[0], patch_shape[1], n_components)
    denoised_embedding_map = denoised_embedding.reshape(patch_shape[0], patch_shape[1], n_components)

    raw_normalized_map = np.zeros_like(raw_embedding_map)
    for c in range(3):  # 对每个颜色通道
        channel = raw_embedding_map[:, :, c]
        c_min = channel.min()
        c_max = channel.max()
        raw_normalized_map[:, :, c] = (channel - c_min) / (c_max - c_min + 1e-8)
    raw_embedding_map = raw_normalized_map
    
    raw_embedding_tensor = torch.tensor(raw_embedding_map, dtype=torch.float32).permute(2, 0, 1)  # [1, H, W]

    denoised_normalized_map = np.zeros_like(denoised_embedding_map)
    for c in range(3):  # 对每个颜色通道
        channel = denoised_embedding_map[:, :, c]
        c_min = channel.min()
        c_max = channel.max()
        denoised_normalized_map[:, :, c] = (channel - c_min) / (c_max - c_min + 1e-8)
    denoised_embedding_map = denoised_normalized_map
    
    denoised_embedding_tensor = torch.tensor(denoised_embedding_map, dtype=torch.float32).permute(2, 0, 1)  # [1, H, W]
    
    # Step 3: 插值为高分辨率
    raw_embedding_up = F.interpolate(raw_embedding_tensor.unsqueeze(0), size=resolution, mode='nearest')
    raw_embedding_up = raw_embedding_up.squeeze()  # [n_comp, H, W]
    if raw_embedding_up.shape[-1] != 3:
        raw_embedding_up = raw_embedding_up[:3, :, :] # :3, :, :, 2:5
        raw_embedding_up = raw_embedding_up.permute(1, 2, 0)
        # embedding_up = embedding_up.mean(-1)
    raw_embedding_up = raw_embedding_up.numpy()
    
    denoised_embedding_up = F.interpolate(denoised_embedding_tensor.unsqueeze(0), size=resolution, mode='nearest')
    denoised_embedding_up = denoised_embedding_up.squeeze()  # [n_comp, H, W]
    if denoised_embedding_up.shape[-1] != 3:
        denoised_embedding_up = denoised_embedding_up[:3, :, :] # :3, :, :, 2:5
        denoised_embedding_up = denoised_embedding_up.permute(1, 2, 0)
        # embedding_up = embedding_up.mean(-1)
    denoised_embedding_up = denoised_embedding_up.numpy()
    
    # Step 4: 原图预处理
    # if isinstance(raw_img, torch.Tensor):
    #     raw_img = raw_img.detach().cpu().numpy()
    # if raw_img.shape[0] == 3:  # [C, H, W] → [H, W, C]
    #     raw_img = np.transpose(raw_img, (1, 2, 0))


    # Step 5: 分别保存原图和UMap
    plt.figure(figsize=(10, 10))
    print(f"raw min: {np.min(raw_embedding_up)}")
    print(f"raw max: {np.max(raw_embedding_up)}")
    plt.imshow(raw_embedding_up, vmin=m, vmax=mx)
    # plt.title("Original Image")
    plt.axis('off')
    plt.tight_layout(pad=0)
    if save_path:
        img_path = save_path + 'raw_teacher.svg'
    else:
        img_path = save_path
    # plt.savefig(img_path, format='svg')
    plt.savefig(
        img_path, 
        format='svg', 
        bbox_inches='tight',  # 自动裁剪空白
        pad_inches=0,        # 完全去除白边
        transparent=True      # 可选：背景透明
        )
    plt.close()
    
    plt.figure(figsize=(10, 10))
    print(f"denoised min: {np.min(denoised_embedding_up)}")
    print(f"denoised max: {np.max(denoised_embedding_up)}")
    plt.imshow(denoised_embedding_up, vmin=m, vmax=mx)
    # plt.title("UMAP Feature Map")
    plt.axis('off')
    # plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
    # fig.colorbar(im, ax=ax, orientation='vertical', shrink=1.0)
    plt.tight_layout(pad=0)
    if save_path:
        img_path = save_path + 'student.svg'
    else:
        img_path = save_path
    # plt.savefig(img_path, format='svg')
    plt.savefig(
        img_path, 
        format='svg', 
        bbox_inches='tight',  # 自动裁剪空白
        pad_inches=0,        # 完全去除白边
        transparent=True      # 可选：背景透明
        )
    plt.close()
    
    
    "Calculate raw teacher norm"
    plt.figure(figsize=(10, 10))
    raw_norm = np.linalg.norm(raw_teacher_feats.cpu().numpy(), axis=-1, keepdims=True)
    raw_norm = torch.tensor(raw_norm)
    raw_norm = raw_norm.reshape(patch_shape[0], patch_shape[1], -1).permute(2, 0, 1) # 1, 28, 28
    raw_norm_up = F.interpolate(raw_norm.unsqueeze(0), size=resolution, mode='nearest') # 1, 448, 448
    raw_norm_up = raw_norm_up.squeeze(0).permute(1, 2, 0).numpy()

    plt.imshow(raw_norm_up)
    # plt.title("UMAP Feature Map")
    plt.axis('off')
    # plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
    # fig.colorbar(im, ax=ax, orientation='vertical', shrink=1.0)
    plt.tight_layout(pad=0)
    if save_path:
        img_path = save_path + 'teacher_norm.svg'
    else:
        img_path = save_path
    # plt.savefig(img_path, format='svg')
    plt.savefig(
        img_path, 
        format='svg', 
        bbox_inches='tight',  # 自动裁剪空白
        pad_inches=0,        # 完全去除白边
        transparent=True      # 可选：背景透明
        )
    plt.close()
    
    
    "Calculate distilled student norm"
    plt.figure(figsize=(10, 10))
    student_norm = np.linalg.norm(student_feats.cpu().numpy(), axis=-1, keepdims=True)
    student_norm = torch.tensor(student_norm)
    student_norm = student_norm.reshape(patch_shape[0], patch_shape[1], -1).permute(2, 0, 1) # 1, 28, 28
    student_norm_up = F.interpolate(student_norm.unsqueeze(0), size=resolution, mode='nearest') # 1, 448, 448
    student_norm_up = student_norm_up.squeeze(0).permute(1, 2, 0).numpy()
    plt.imshow(student_norm_up)
    # plt.title("UMAP Feature Map")
    plt.axis('off')
    # plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
    # fig.colorbar(im, ax=ax, orientation='vertical', shrink=1.0)
    plt.tight_layout(pad=0)
    if save_path:
        img_path = save_path + 'student_norm.svg'
    else:
        img_path = save_path
    # plt.savefig(img_path, format='svg')
    plt.savefig(
        img_path, 
        format='svg', 
        bbox_inches='tight',  # 自动裁剪空白
        pad_inches=0,        # 完全去除白边
        transparent=True      # 可选：背景透明
        )
    plt.close()
    
    "Calculate difference between raw teacher norm and student norm"
    plt.figure(figsize=(10, 10))
    difference_norm = raw_norm - student_norm
    difference_norm = torch.tensor(difference_norm)
    difference_norm = difference_norm.reshape(patch_shape[0], patch_shape[1], -1).permute(2, 0, 1) # 1, 28, 28
    difference_norm_up = F.interpolate(difference_norm.unsqueeze(0), size=resolution, mode='nearest') # 1, 448, 448
    difference_norm_up = difference_norm_up.squeeze(0).permute(1, 2, 0).numpy()
    
    plt.imshow(difference_norm_up)
    # plt.title("UMAP Feature Map")
    plt.axis('off')
    # plt.colorbar(im, orientation='horizontal', fraction=0.046, pad=0.04)
    # fig.colorbar(im, ax=ax, orientation='vertical', shrink=1.0)
    plt.tight_layout(pad=0)
    if save_path:
        img_path = save_path + 'difference_norm.svg'
    else:
        img_path = save_path
    # plt.savefig(img_path, format='svg')
    plt.savefig(
        img_path, 
        format='svg', 
        bbox_inches='tight',  # 自动裁剪空白
        pad_inches=0,        # 完全去除白边
        transparent=True      # 可选：背景透明
        )
    plt.close()


if __name__ == "__main__":
    vis_augmented_teacher()