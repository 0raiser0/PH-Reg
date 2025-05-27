import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def vis_augmented_teacher():
    parser = argparse.ArgumentParser(description="Train a CLIP model with distillation.")
    parser.add_argument("--data_root", type=str, default="/data/kelvinyzp/flickr30k", help="Dataset root directory.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--save_dir", type=str, default="/data/chenyinjie/CYJcode/traindistill/DINOv2/distilledweights", help="Directory to save model checkpoints.")
    # training settting
    parser.add_argument("--unused_param", type=bool, default=True, help="Some parameters in transformer resblocks are not used when fine-tuning.")
    parser.add_argument("--resolution", type=int, default=518, help="Input Image size")
    parser.add_argument("--shift_frac", type=float, default=0.15, help="Shifting fraction used in shifting augmentation")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.")
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
    parser.add_argument("--weight_frozen", type=bool, default=True, help="Freeze models' weights when fine tuning")
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
        shuffle=False,
        drop_last=True,
        num_workers=4
    )
    # accelerator = Accelerator(device_placement=True,
    #                         split_batches=False,
    #                         mixed_precision="bf16")
    # device = accelerator.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'else')
    
    # initialization Model
    # Teacher Model
    teacher_model = AugDINOv2Base(
        pretrained_path=args_dict['pretrained_path'],
        attn_implementation=args_dict['attn_implementation'],
        weight_frozen=args_dict['weight_frozen']
    ).to(torch.float32).to(device)
    print(f"layer_scale: {teacher_model.config.layerscale_value}")
    print(f"drop_path_rate: {teacher_model.config.drop_path_rate}")
    teacher_model.config.drop_path_rate = 0.0
    
    
    # clip_model, unwrapped_model, train_dataloader = accelerator.prepare(clip_model, unwrapped_model, train_dataloader)
    teacher_model.eval()
    for original_images, shifted_images, shifted_idxs in train_dataloader:
        original_images = original_images.to(device)
        shifted_images = shifted_images.to(device)
        shifted_idxs = shifted_idxs.to(device)
        with torch.inference_mode():
            print(f"inference layer_scale: {teacher_model.config.layerscale_value}")
            print(f"inference drop_path_rate: {teacher_model.config.drop_path_rate}")
        with torch.no_grad():
            teacher_img_feats = teacher_model(
                        args_dict=args_dict,
                        shifted_images=shifted_images,
                        shifted_idxs=shifted_idxs,
                        output_attentions=False,
                        output_hidden_states=False
                    )
        teacher_img_feats = teacher_img_feats['last_hidden_state']
        teacher_img_feats = teacher_img_feats.to(torch.float32)
        # with torch.inference_mode():
        #     teacher_img_feats = teacher_model.forward_images(
        #         images=original_images,
        #         output_attentions=False,
        #         output_hidden_states=False
        #     )
        #     teacher_img_feats = teacher_img_feats['last_hidden_state']
        #     teacher_img_feats = teacher_img_feats[:, 1:, :]
        #     teacher_img_feats = teacher_img_feats.to(torch.float32)
        break

    
    fig, axes = plt.subplots(args_dict['batch_size'], 2, figsize=(16, 32))  # Adjust figsize for the grid layout
    
    for i in range(args_dict['batch_size']):
        # Extract the i-th image (shape: 3x224x224)
        image_tensor = original_images[i].to(torch.float32)
        # np_image = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert to NumPy (H x W x C)
        # np_image = (np_image * 255).astype(np.uint8)  # Scale to [0, 255]
        np_image = tensor_to_image(image_tensor)

        # Show the original image
        axes[i, 0].imshow(np_image)
        axes[i, 0].axis("off")
        axes[i, 0].set_title("Input Image" if i == 0 else "")

        # Extract the feature map and reshape
        # smoo = image_embedding[i].cpu().float().detach().numpy().reshape(num_patches, num_patches, -1).astype(np.single)
        # smoo = smoo / np.linalg.norm(smoo, axis=-1, keepdims=True)
        # smoo_flattened = smoo.reshape(-1, smoo.shape[-1])
        feature_map = teacher_img_feats[i].cpu().float().detach().numpy() # [num_patches, hidden_size]
        eps = 1e-8
        mean = feature_map.mean(axis=1, keepdims=True)
        std = feature_map.std(axis=1, keepdims=True, ddof=0)
        whitened_map = (feature_map - mean) / (std + eps)

        # Standardize the data (mean=0, std=1) before PCA
        # scaler = StandardScaler()
        # smoo_standardized = scaler.fit_transform(smoo_flattened)

        # Apply PCA to the standardized data
        pca = PCA(3)
        pca_result = pca.fit_transform(whitened_map)
        num_patches = args_dict["resolution"] // args_dict["patch_size"]
        assert num_patches * args_dict["patch_size"] == args_dict["resolution"]
        out = pca_result.reshape(num_patches, num_patches, 3)
        out = out - np.min(out, axis=-1, keepdims=True)
        out = out / np.max(out, axis=-1, keepdims=True)

        # Show the original PCA feature map
        axes[i, 1].imshow(out)
        axes[i, 1].axis("off")
        axes[i, 1].set_title("PCA" if i == 0 else "")

    # Adjust layout and save the figure
    plt.tight_layout()  # Adjust spacing
    plt.savefig("DINOv2_Aug.png", dpi=300)  # Save as a single image
    plt.show()  # Display the figure
    

def tensor_to_image(tensor):
    """
    Converts a PyTorch tensor (C x H x W) back to a NumPy image (H x W x C) in the range [0, 255].
    """
    # Ensure the tensor is on the CPU and detach gradients if necessary
    tensor = tensor.cpu().clone()
    
    # Undo normalization: multiply by std and add mean (for each channel)
    # Mean and std used in the normalization step
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    
    # Clamp values to [0, 1] to handle any rounding issues
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert from (C x H x W) to (H x W x C) and scale to [0, 255]
    np_image = tensor.permute(1, 2, 0).numpy()  # Reorder dimensions
    np_image = (np_image * 255).astype(np.uint8)  # Scale to [0, 255]
    
    return np_image


if __name__ == "__main__":
    vis_augmented_teacher()