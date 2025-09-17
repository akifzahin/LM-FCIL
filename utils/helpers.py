def set_global_seed(seed: int):
    """Set global RNG seeds and deterministic flags (best-effort)."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-gpu
    # cuDNN deterministic behavior (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Force PyTorch deterministic algorithms where available (may raise on some ops)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # if your torch version doesn't support or some ops are non-deterministic,
        # you can catch the error and proceed (but note reproducibility risk)
        pass

def worker_init_fn(worker_id):
    """
    DataLoader worker init to make per-worker RNG deterministic.
    When creating DataLoader, pass: worker_init_fn=worker_init_fn
    """
    # seed for this worker derived from global RNG + worker_id
    seed = torch.initial_seed() % (2**32)  # torch.initial_seed accounts for base seed + worker
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)

