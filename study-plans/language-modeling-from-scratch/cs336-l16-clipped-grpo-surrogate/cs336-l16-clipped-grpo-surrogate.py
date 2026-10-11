import torch

def clipped_grpo_surrogate(
    new_log_probs: torch.Tensor, old_log_probs: torch.Tensor,
    advantages: torch.Tensor, token_mask: torch.Tensor,
    clip_epsilon: int | float = 0.2,
) -> dict:
    """
    Returns a dict: loss and clipped_fraction (scalars), per_sequence_surrogate (tensor).
    """
    logprobs_ratios = (new_log_probs - old_log_probs).clip(-60, 60)
    ratios = logprobs_ratios.exp()

    advantages = advantages.unsqueeze(-1) # [seq_len, 1]
    pg1 = ratios * advantages
    pg2 = ratios.clip(1 - clip_epsilon, 1 + clip_epsilon) * advantages
    loss = torch.min(pg1, pg2) * token_mask

    clipped = (ratios < 1 - clip_epsilon) | (ratios > 1 + clip_epsilon)
    clipped = clipped * token_mask
    
    # Let's handle sequence's with all masked tokens
    token_count_sum = token_mask.sum().clamp_min(1) # to avoid zero-divison error
    sequence_token_sum = token_mask.sum(dim=-1).clamp_min(1)

    loss_per_token = -loss.sum() / token_count_sum
    clipped_frac = clipped.sum() / token_count_sum

    per_sequence_loss = loss.sum(dim=-1) / sequence_token_sum

    return {
        "loss": loss_per_token,
        "per_sequence_surrogate": per_sequence_loss,
        "clipped_fraction": clipped_frac
    }

    