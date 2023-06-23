NAME: str = "GoNNet"

BATCH_SIZE: int = 1024
BOARD_SIZE: int = 9
EVAL_MATCH_CNT: int = 16
NUM_SIMS: int = 512
PROCESSES: int = 16
UPDATE_MATCH_CNT: int = 32
UPDATE_THRESHOLD: float = 0.51

if NAME == "ResNet":
    from .ResNet import GoNNet, GoNNetWrapper, net_config
elif NAME == "GoNNet":
    from .GoNNet import GoNNet, GoNNetWrapper, net_config


import torch.multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except:
    pass
