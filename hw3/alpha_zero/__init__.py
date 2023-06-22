NAME: str = "ResNet"

BATCH_SIZE: int = 1024
BOARD_SIZE: int = 9
PROCESSES: int = 16
UPDATE_THRESHOLD: float = 0.51

if NAME == "ResNet":
    from .ResNet import GoNNet, GoNNetWrapper, net_config
elif NAME == "GoNNet":
    from .GoNNet import GoNNet, GoNNetWrapper, net_config
