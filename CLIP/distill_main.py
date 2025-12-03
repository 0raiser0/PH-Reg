import os, sys
sys.path.append('./')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import json
import torch
assert torch.cuda.is_available()
import torch.nn.functional as F
# from torchvision import transforms
import torch.optim as optim
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from tqdm import tqdm

# Import dataset
from datasets.flickr30k_dataset import prepare_flickr30_dataloader
from torch.utils.data import DataLoader
# Import teacher model and student model
from models.teacher_CLIP import TeacherCLIP # Teacher Model
from models.student_CLIP import StudentCLIP # Student Model
from utils.loss import cosine_similarity_loss, mse_loss

from utils.functions import save_settings


def train_model_distill():
    parser = argparse.ArgumentParser(description="Train a CLIP model with distillation.")
    parser.add_argument("--data_root", type=str, default="", help="Dataset root directory.")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--save_dir", type=str, default="", help="Directory to save model checkpoints.")
    # training settting
    parser.add_argument("--unused_param", type=bool, default=True, help="Some parameters in transformer resblocks are not used when fine-tuning.")
    parser.add_argument("--resolution", type=int, default=448, help="Input Image size")
    parser.add_argument("--shift_frac", type=float, default=0.15, help="Shifting fraction used in shifting augmentation")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--end_lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay.")
    parser.add_argument("--counts", type=int, default=10, help="Number of sample points for the shifting augmentaion in Teacher Model.")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    # Model setting
    parser.add_argument("--patch_size", type=int, default=16, help="Model embedding patch size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Model embedding hidden size")
    parser.add_argument("--pretrained_path", type=str, default="", help="Teacher and Student model pretrained weight path")
    parser.add_argument("--weight_frozen", type=bool, default=True, help="Freeze models' weights when fine tuning")
    parser.add_argument("--gaussian_std", type=float, default=5.0, help="Gaussian Window size in NACLIP")
    parser.add_argument("--arch", type=str, default="reduced", help="Model architecture setting as NACLIP")
    parser.add_argument("--num_of_reg", type=int, default=16, help="Number of register tokens.")
    parser.add_argument("--mse_scale", type=float, default=1.0, help="Scale MSELoss")
    args = parser.parse_args()
    
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
        num_workers=8
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
    teacher_model = TeacherCLIP(
        pretrained=args_dict['pretrained_path'],
        gaussian_std=args_dict['gaussian_std'],
        arch=args_dict['arch'],
        patch_size=args_dict['patch_size'],
    )
    # Student Model
    student_model = StudentCLIP(
        pretrained=args_dict['pretrained_path'],
        num_registers=args_dict['num_of_reg'],
        patch_size=args_dict['patch_size'],
        input_resolution=(args_dict['resolution'], args_dict['resolution'])
    )
    
    optimizer = optim.AdamW([
    {'params': student_model.register_tokens, 'lr': args_dict['lr'], 'weight_decay': args_dict['weight_decay']},
    {'params': student_model.vit.positional_embedding, 'lr': args_dict['lr'], 'weight_decay': args_dict['weight_decay']},
    {'params': student_model.vit.conv1.weight, 'lr': args_dict['lr'], 'weight_decay': args_dict['weight_decay']}
    ])
    # conv1 has weights and bias, but openai CLIP does not have bias.
    
    # add new parameters to the optimizer
    optimizer.add_param_group({
    'params': student_model.vit.transformer.resblocks[-2].parameters(),
    'lr': args_dict['lr'],
    'weight_decay': args_dict['weight_decay']
    })
    optimizer.add_param_group({
        'params': student_model.vit.transformer.resblocks[-1].parameters(),
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
        #     print(f"{name}: requires_grad={param.requires_grad}")
    
    save_settings(
        experiment_name='FromNaCLIP',
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
                        shifted_idxs=shifted_idxs
                    ) # batch_size, num_patches, hidden_size
            # detach from teacher model -> detach()
            teacher_img_feats = teacher_img_feats.to(torch.float32)

            # Check for NaN/inf in student features
            if torch.isnan(teacher_img_feats).any() or torch.isinf(teacher_img_feats).any():
                print(f"[ERROR] Teacher features contain NaN or Inf at batch {i}, epoch {epoch+1}. Skipping batch.")
                continue
            
            # Back propagation
            optimizer.zero_grad()
            
            with accelerator.autocast():
                student_img_feats = student_model(original_images)
                # discard cls and register tokens in feature
                student_img_feats = student_img_feats[:,1:-args_dict["num_of_reg"],:]
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
        if (epoch+1) % 10 == 0 or epoch == 0: #  or epoch == 0
            accelerator.wait_for_everyone()
            all_weights_path = os.path.join(args_dict['save_dir'], f"distilled_vit_weights_{epoch + 1}.pth")
            if accelerator.is_main_process:
                to_save_model = accelerator.unwrap_model(student_model)
                # save all weights in ViT
                torch.save(to_save_model.state_dict(), all_weights_path)
    
    # save training loss    
    jsonfile = os.path.join(args_dict['save_dir'], 'loss.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': loss_list
        }
        json.dump(data, f, sort_keys=True, indent=4)
    print("Training completed.")
    

if __name__ == "__main__":
    train_model_distill()
            
            