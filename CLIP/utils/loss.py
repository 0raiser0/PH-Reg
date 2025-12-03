import torch
import torch.nn.functional as F


# Cosine Similarity Loss
def cosine_similarity_loss(features_a, features_b):
    """
    Compute cosine similarity loss between two feature sets.
    Args:
        features_a: Tensor of shape [batch_size, num_patches, feature_dim]
        features_b: Tensor of shape [batch_size, num_patches, feature_dim]
    Returns:
        Loss value (1 - cosine similarity)
    """
    # Normalize features
    features_a = features_a / (features_a.norm(dim=-1, keepdim=True) + 1e-8)
    features_b = features_b / (features_b.norm(dim=-1, keepdim=True) + 1e-8)
    # torch.sum(a_norm*b_norm) / a_norm.shape[0]
    cosine_sim = (features_a * features_b).sum(dim=-1).mean()

    loss = 1 - cosine_sim  # Minimize 1 - cosine similarity
    return loss


def mse_loss(feature_a, feature_b, coeff=0.2):
    # mse = F.mse_loss(feature_a, feature_b, reduce='mean')
    mse = F.mse_loss(feature_a, feature_b, reduction='mean')
    mse = mse * coeff
    return mse