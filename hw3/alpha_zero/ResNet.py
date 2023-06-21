import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from .GoBoard import Stone
from .GoGame import GoGame

net_config: Dict[str, Any] = {
    "batch_size": 64,
    "cuda": torch.cuda.is_available(),
    "dropout": 0.3,
    "epochs": 10,
    "lr": 0.001,
    "num_channels": 64,
}


def get_encoded_state(board: np.ndarray) -> np.ndarray:
    return np.stack(
        arrays=(board == Stone.BLACK, board == Stone.WHITE, board == Stone.EMPTY)
    )


class ResBlock(nn.Module):
    conv_1: nn.Conv2d
    batch_norm_1: nn.BatchNorm2d
    conv_2: nn.Conv2d
    batch_norm_2: nn.BatchNorm2d

    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, padding=1
        )
        self.batch_norm_1 = nn.BatchNorm2d(num_features=num_hidden)
        self.conv_2 = nn.Conv2d(
            in_channels=num_hidden, out_channels=num_hidden, kernel_size=3, padding=1
        )
        self.batch_norm_2 = nn.BatchNorm2d(num_features=num_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.batch_norm_1(self.conv_1(x)))
        x = self.batch_norm_2(self.conv_2(x))
        x += residual
        x = F.relu(x)
        return x


class GoNNet(nn.Module):
    action_size: int
    args: Dict[str, Any]
    board_x: int
    board_y: int

    start_block: nn.Sequential
    back_bone: nn.ModuleList
    policy_head: nn.Sequential
    value_head: nn.Sequential

    def __init__(self, game: GoGame, args: Dict[str, Any]) -> None:
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        super().__init__()
        self.action_size = game.action_size()
        self.args = args
        self.board_x, self.board_y = game.obs_size()
        num_res_blocks: int = 4
        num_hidden: int = 64

        self.start_block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )

        self.back_bone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_res_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * self.board_x * self.board_y, game.action_size()),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * self.board_x * self.board_y, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        x = x.view(-1, 3, self.board_x, self.board_y)
        x = self.start_block(x)
        for res_block in self.back_bone:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value


class GoNNetWrapper:
    action_size: int
    board_x: int
    board_y: int
    nnet: GoNNet

    def __init__(self, game: GoGame):
        self.action_size: int = game.action_size()
        self.board_x, self.board_y = game.obs_size()
        self.nnet: GoNNet = GoNNet(game, net_config)

        if net_config["cuda"]:
            self.nnet.cuda()

    def train(
        self, training_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    ) -> List[Tuple[float, float]]:
        """
        training_data: list of (board, policy, value)
        """
        batch_size: int = self.nnet.args["batch_size"]
        cuda: bool = self.nnet.args["cuda"]
        epochs: int = self.nnet.args["epochs"]
        lr: float = self.nnet.args["lr"]

        optimizer: optim.Adam = optim.Adam(
            params=self.nnet.parameters(), lr=lr, weight_decay=1e-4
        )

        loss_list: List[Tuple[float, float]] = []

        for epoch in range(epochs):
            print(f"Epoch[{str(epoch + 1)}] ")
            self.nnet.train()

            batch_count: int = len(training_data) // batch_size

            t: tqdm = tqdm(range(batch_count), desc="Training ResNet")
            for _ in t:
                sample_ids = np.random.randint(len(training_data), size=batch_size)
                boards, pis, vs = list(zip(*[training_data[i] for i in sample_ids]))
                boards = torch.FloatTensor(
                    np.array([get_encoded_state(state) for state in boards]).astype(
                        np.float64
                    )
                )
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if cuda:
                    boards, target_pis, target_vs = (
                        boards.contiguous().cuda(),
                        target_pis.contiguous().cuda(),
                        target_vs.contiguous().cuda(),
                    )

                ######################################
                #        YOUR CODE GOES HERE         #
                ######################################
                # Compute loss and backprop
                policy: torch.Tensor
                value: torch.Tensor
                policy, value = self.nnet(boards)
                loss = F.mse_loss(
                    input=value, target=target_vs, reduction="sum"
                ) + F.cross_entropy(input=policy, target=target_pis, reduction="sum")
                optimizer.zero_grad()
                loss = loss.sum()
                loss_list.append((datetime.now().timestamp(), loss.item()))
                loss.backward()
                optimizer.step()

        return loss_list

    def predict(self, board: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        predict the policy and value of the given board
        @param board: np array
        @return (policy, value): a policy vector for the given board, and a float value
        """
        cuda: bool = self.nnet.args["cuda"]

        board_tensor: torch.Tensor = torch.FloatTensor(get_encoded_state(board))
        if cuda:
            board_tensor = board_tensor.contiguous().cuda()
        board_tensor = board_tensor.view(3, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi: torch.Tensor
            v: torch.Tensor
            pi, v = self.nnet(board_tensor)

        return (
            torch.softmax(pi, dim=1).squeeze(dim=0).data.cpu().numpy(),
            v.squeeze(dim=0).data.cpu().numpy(),
        )

    def save_checkpoint(
        self,
        folder: Union[str, Path] = "checkpoint",
        filename: str = "checkpoint.pth.tar",
    ):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(
        self,
        folder: Union[str, Path] = "checkpoint",
        filename: str = "checkpoint.pth.tar",
    ):
        cuda: bool = self.nnet.args["cuda"]

        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if cuda else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
