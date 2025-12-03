import os, json
import torch

def save_settings(experiment_name, hyperparameters, model, base_dir='experiments'):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    
    # Save the hyperparameters
    paramfile = os.path.join(outdir, 'hyperparameters.json')
    with open(paramfile, 'w') as f:
        json.dump(hyperparameters, f, sort_keys=True, indent=4)
        
    # Save model config
    modelfile = os.path.join(outdir, 'model_config.json')
    model_config_dict = {
        name: {
            'size': param.size(),
            'requires_grad': param.requires_grad
        }
        for name, param in model.named_parameters()
    }
    with open(modelfile, 'w') as f:
        json.dump(model_config_dict, f, sort_keys=False, indent=4)


# Save only parameters with require_grad=True
def save_trainable_parameters(model, optimizer, save_path):
    """
    Save only the parameters that have requires_grad=True.
    Args:
        model: The model with parameters.
        optimizer: The optimizer used for training.
        save_path: Path to save the checkpoint.
    """
    trainable_params = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    checkpoint = {
        'trainable_state_dict': trainable_params,
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, save_path)
    print(f"Trainable parameters saved to {save_path}")


# Load trainable parameters into the model
def load_trainable_parameters(model, optimizer, load_path, device):
    """
    Load only the trainable parameters into the model.
    Args:
        model: The model to load the parameters into.
        optimizer: The optimizer to restore the state.
        load_path: Path to the saved checkpoint.
        device: Device to map the loaded parameters (e.g., "cuda" or "cpu").
    """
    checkpoint = torch.load(load_path, map_location=device)
    trainable_state_dict = checkpoint['trainable_state_dict']

    # Copy each trainable parameter back into the model
    model_state_dict = model.state_dict()
    for name, param in trainable_state_dict.items():
        if name in model_state_dict:
            model_state_dict[name].copy_(param)
    # Load the entire model weight
    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Trainable parameters loaded from {load_path}")
    return model, optimizer