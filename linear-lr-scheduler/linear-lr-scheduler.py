def linear_lr(step: int, total_steps: int, initial_lr: float, final_lr: float = 0.0, warmup_steps: int = 0) -> float:
    """
    Returns the learning rate as a float.
    """
    # Write code here
    if 0 < warmup_steps > step:
        return float(initial_lr * (step / warmup_steps))
    elif warmup_steps <= step < total_steps:
        step -= warmup_steps
        total_steps -= warmup_steps
        return float(initial_lr + (step / total_steps) * (final_lr - initial_lr))
    else:
        return float(final_lr)