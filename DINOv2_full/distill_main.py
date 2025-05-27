import os, sys
sys.path.append('./')
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
import torch
torch.cuda.empty_cache()
assert torch.cuda.is_available()
import torch.nn.functional as F
# from torchvision import transforms
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm
import json

# Import dataset
from datasets.flickr30k_dataset import prepare_flickr30_dataloader
from torch.utils.data import DataLoader
# Import shiting augmentation functions
# from utils.denoising import prepare_all_offsets, prepare_flip, prepare_patch_idxs, farthest_point_sample
# Import teacher model and student model
from models.teacherDINOv2 import AugDINOv2Base # Teacher Model
from models.studentDINOv2 import RegDINOv2Base # Student Model
from utils.loss import cosine_similarity_loss, mse_loss

from utils.functions import save_trainable_parameters, save_settings


def train_model_distill():
    parser = argparse.ArgumentParser(description="Train a CLIP model with distillation.")
    parser.add_argument("--data_root", type=str, default="/data/kelvinyzp/flickr30k", help="Dataset root directory.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--save_dir", type=str, default="/data/chenyinjie/CYJcode/traindistill/DINOv2_full/distilledweights", help="Directory to save model checkpoints.")
    # training settting
    parser.add_argument("--unused_param", type=bool, default=False, help="Some parameters in transformer resblocks are not used when fine-tuning.")
    parser.add_argument("--resolution", type=int, default=518, help="Input Image size")
    parser.add_argument("--shift_frac", type=float, default=0.15, help="Shifting fraction used in shifting augmentation")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.") # 3e-4
    parser.add_argument("--end_lr", type=float, default=5e-6, help="Learning rate.")
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
    train_set, shuffle = prepare_flickr30_dataloader(
        args_dict=args_dict,
        mode='train'
    )
    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=args_dict['batch_size'],
        shuffle=shuffle,
        drop_last=True,
        num_workers=4
    )
    
    if args_dict['unused_param']:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # 'find_unused_parameters': skip unused parameters
        accelerator = Accelerator(
            device_placement=True,
            split_batches=False,
            mixed_precision="bf16",
            kwargs_handlers=[ddp_kwargs])
    else:
        accelerator = Accelerator(
            device_placement=True,
            split_batches=False,
            mixed_precision="bf16")

    device = accelerator.device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'else')
    
    # initialization Model
    # Teacher Model
    teacher_model = AugDINOv2Base(
        pretrained_path=args_dict['pretrained_path'],
        attn_implementation=args_dict['attn_implementation'],
        weight_frozen=args_dict['weight_frozen']
    ).to(torch.float32)
    # Student Model
    student_model = RegDINOv2Base(
        pretrained_path=args_dict['pretrained_path'],
        attn_implementation=args_dict['attn_implementation'],
        num_registers=args_dict['num_of_reg'],
        weight_frozen=args_dict['weight_frozen'],
    ).to(torch.float32)
    
    for name, param in student_model.named_parameters():
        if param.requires_grad:
            param.requires_grad = False
    
    # Active conv
    student_model.embed.position_embeddings.requires_grad = True
    # Active registers
    student_model.registers.requires_grad = True
    # Active position embedding
    student_model.embed.patch_embeddings.projection.weight.requires_grad = True
    student_model.embed.patch_embeddings.projection.bias.requires_grad = True
    # Active the last transformer blocks
    for param in student_model.encoder.layer[-1].parameters():
        param.requires_grad = True
    
    # active registers and position embedding
    optimizer = optim.AdamW([
    {'params': student_model.registers, 'lr': args_dict['lr'], 'weight_decay': args_dict['weight_decay']},
    {'params': student_model.embed.position_embeddings, 'lr': args_dict['lr'], 'weight_decay': args_dict['weight_decay']}
    ])
    
    # active conv layer
    # optimizer.add_param_group(
    #     {'params': student_model.embed.patch_embeddings.projection.weight,
    #     'lr': args_dict['lr'],
    #     'weight_decay': args_dict['weight_decay']},
    #     {'params': student_model.embed.patch_embeddings.projection.bias,
    #     'lr': args_dict['lr'],
    #     'weight_decay': args_dict['weight_decay']},                      
    # )
    optimizer.add_param_group(
        {'params': student_model.embed.patch_embeddings.projection.weight,
        'lr': args_dict['lr'],
        'weight_decay': args_dict['weight_decay']}
    )
    
    optimizer.add_param_group(
        {'params': student_model.embed.patch_embeddings.projection.bias,
        'lr': args_dict['lr'],
        'weight_decay': args_dict['weight_decay']} 
    )
    
    # active the last two transformer blocks
    optimizer.add_param_group({
        'params': student_model.encoder.layer[-1].parameters(),
        'lr': args_dict['lr'],
        'weight_decay': args_dict['weight_decay']
    })

    
    # set models, optimizer, scheduler to cuda
    teacher_model, student_model, optimizer, train_dataloader = accelerator.prepare(teacher_model, student_model, optimizer, train_dataloader)
    decay_ratio = args_dict['end_lr'] / args_dict['lr']
    
    # Check Model Weight Initialization
    for name, param in student_model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"NaN or Inf in parameter {name}!")
            
        # Check Fine-tuning Parameters
        # if param.requires_grad:
            # print(f"{name}: requires_grad={param.requires_grad}")
    
    save_settings(
        experiment_name='FromAugDINOv2',
        hyperparameters=args_dict,
        model=student_model,
        base_dir=args_dict['save_dir']
    )
    
    loss_list = []
    # Training loop
    print(f"Starting training on {device}.")
    for epoch in range(args_dict['num_epochs']):
        student_model.train()
        teacher_model.eval()
        running_loss = 0.0
        
        # tqdm progress bar for training
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args_dict['num_epochs']}", leave=False)
        for i, (original_images, shifted_images, shifted_idxs) in enumerate(loop):
        # for original_images, shifted_images, shifted_idxs in train_dataloader:
            # images = images.to(device)
            assert not torch.isnan(shifted_images).any() or not torch.isinf(shifted_images).any(), "Input images has unexpected values, NaN or Inf"

            with torch.no_grad():
                with accelerator.autocast():
                    teacher_img_feats = teacher_model(
                        args_dict=args_dict,
                        shifted_images=shifted_images,
                        shifted_idxs=shifted_idxs,
                        output_attentions=False,
                        output_hidden_states=False
                    ) # batch_size, num_patches, hidden_size
            # detach from teacher model -> detach()
            teacher_img_feats = teacher_img_feats['last_hidden_state']
            teacher_img_feats = teacher_img_feats.to(torch.float32)

            # Check for NaN/inf in student features
            if torch.isnan(teacher_img_feats).any() or torch.isinf(teacher_img_feats).any():
                print(f"[ERROR] Teacher features contain NaN or Inf at batch {i}, epoch {epoch+1}. Skipping batch.")
                continue
            
            # Back propagation
            optimizer.zero_grad()
            
            with accelerator.autocast():
                student_img_feats = student_model(
                    images=original_images,
                    output_attentions=False,
                    output_hidden_states=False
                )
                student_img_feats = student_img_feats['last_hidden_state']
                # discard cls and register tokens in feature
                student_img_feats = student_img_feats[:, args_dict["num_of_reg"]+1:, :]
                # batch_size, num_patches, hidden_size
                student_img_feats = student_img_feats.to(torch.float32)
                
                # Check for NaN/inf in student features
                if torch.isnan(student_img_feats).any() or torch.isinf(student_img_feats).any():
                    print(f"[ERROR] Student features contain NaN or Inf at batch {i}, epoch {epoch+1}. Skipping batch.")
                    continue
                
                # add MSELoss to loss!!
                loss1 = cosine_similarity_loss(teacher_img_feats, student_img_feats) # We didn't use F.cos_sim in pytorch
                loss2 = mse_loss(teacher_img_feats, student_img_feats, coeff=args_dict['mse_scale'])
                
                loss = loss1 + loss2
            
                # Check for NaN/inf in loss
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"[ERROR] Loss is NaN or Inf at batch {i}, epoch {epoch+1}. Skipping batch.")
                    continue
                
            # loss.backward()
            accelerator.backward(loss)
            
            optimizer.step()
            
            new_lr = args_dict['lr'] * (decay_ratio ** ((epoch+1) / args_dict['num_epochs']))
            if new_lr <= args_dict['end_lr']:
                new_lr = args_dict['end_lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            
            running_loss = loss1.item() + loss2.item() + running_loss
            
        print(f"Epoch [{epoch + 1}/{args_dict['num_epochs']}], Loss: {running_loss / len(train_dataloader)}")
        loss_list.append(running_loss / len(train_dataloader))
        
        # Save trainable parameters after each epoch
        "Need to be  modifed -> save trainable parameters or all ViT parameters?"
        "!!! Huggingface has specific 'save' api and we can use it to load model using from_pretrained"
        if (epoch+1) % 10 == 0 or epoch == 0:
            accelerator.wait_for_everyone()
            all_weights_path = os.path.join(args_dict['save_dir'], f"distilled_dinov2_weights_{epoch + 1}.pth")
            if accelerator.is_main_process:
                to_save_model = accelerator.unwrap_model(student_model)
                # # save all weights in ViT
                torch.save(to_save_model.state_dict(), all_weights_path)
                
                train_path = '/data/chenyinjie/CYJcode/traindistill/DINOv2_full/trainableweights'
                save_path = os.path.join(train_path, f"model_checkpoint_epoch_{epoch + 1}.pth")
                save_trainable_parameters(to_save_model, optimizer, save_path)
                            
    jsonfile = os.path.join(args_dict['save_dir'], 'loss.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': loss_list
        }
        json.dump(data, f, sort_keys=True, indent=4)
    print("Training completed.")
    

if __name__ == "__main__":
    train_model_distill()
            
            