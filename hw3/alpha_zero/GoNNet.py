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

from . import BATCH_SIZE
from .GoGame import GoGame

net_config: Dict[str, Any] = {
    "batch_size": BATCH_SIZE,
    "cuda": torch.cuda.is_available(),
    "dropout": 0.3,
    "epochs": 10,
    "lr": 0.001,
    "num_channels": 256,
}


class GoNNet(nn.Module):
    action_size: int
    args: Dict[str, Any]
    board_x: int
    board_y: int

    conv_1: nn.Conv2d
    conv_2: nn.Conv2d
    conv_3: nn.Conv2d
    conv_4: nn.Conv2d
    batch_norm_1: nn.BatchNorm2d
    batch_norm_2: nn.BatchNorm2d
    batch_norm_3: nn.BatchNorm2d
    batch_norm_4: nn.BatchNorm2d
    linear_1: nn.Linear
    linear_batch_norm_1: nn.BatchNorm1d
    linear_2: nn.Linear
    linear_batch_norm_2: nn.BatchNorm1d
    linear_3: nn.Linear
    linear_4: nn.Linear

    def __init__(self, game: GoGame, args: Dict[str, Any]) -> None:
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        super().__init__()
        self.action_size = game.action_size()
        self.args = args
        self.board_x, self.board_y = game.obs_size()
        num_channels: int = args["num_channels"]

        self.conv_1 = nn.Conv2d(
            in_channels=1, out_channels=num_channels, kernel_size=3, stride=1, padding=1
        )
        self.conv_2 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_3 = nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1
        )
        self.conv_4 = nn.Conv2d(
            in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1
        )

        self.batch_norm_1 = nn.BatchNorm2d(num_features=num_channels)
        self.batch_norm_2 = nn.BatchNorm2d(num_features=num_channels)
        self.batch_norm_3 = nn.BatchNorm2d(num_features=num_channels)
        self.batch_norm_4 = nn.BatchNorm2d(num_features=num_channels)

        self.linear_1 = nn.Linear(
            in_features=num_channels * (self.board_x - 4) * (self.board_y - 4),
            out_features=512,
        )
        self.linear_batch_norm_1 = nn.BatchNorm1d(num_features=512)

        self.linear_2 = nn.Linear(in_features=512, out_features=256)
        self.linear_batch_norm_2 = nn.BatchNorm1d(num_features=256)

        self.linear_3 = nn.Linear(in_features=256, out_features=self.action_size)

        self.linear_4 = nn.Linear(in_features=256, out_features=1)

    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        dropout: float = self.args["dropout"]
        num_channels: int = self.args["num_channels"]

        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu(self.batch_norm_1(self.conv_1(s)))
        s = F.relu(self.batch_norm_2(self.conv_2(s)))
        s = F.relu(self.batch_norm_3(self.conv_3(s)))
        s = F.relu(self.batch_norm_4(self.conv_4(s)))
        s = s.view(-1, num_channels * (self.board_x - 4) * (self.board_y - 4))

        s = F.dropout(
            F.relu(self.linear_batch_norm_1(self.linear_1(s))),
            p=dropout,
            training=self.training,
        )
        s = F.dropout(
            F.relu(self.linear_batch_norm_2(self.linear_2(s))),
            p=dropout,
            training=self.training,
        )

        policy = self.linear_3(s)
        value = self.linear_4(s)

        return F.log_softmax(policy, dim=1), torch.tanh(value)


class GoNNetWrapper:
    action_size: int
    board_x: int
    board_y: int
    nnet: GoNNet

    def __init__(self, game: GoGame):
        self.action_size: int = game.action_size()
        self.board_x, self.board_y = game.obs_size()
        self.nnet: GoNNet = GoNNet(game, net_config)
        self.nnet.share_memory()

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

            t: tqdm = tqdm(range(batch_count), desc="Training NNet")
            for _ in t:
                sample_ids = np.random.randint(len(training_data), size=batch_size)
                boards, pis, vs = list(zip(*[training_data[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
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
                assert policy.shape == (batch_size, self.action_size)
                assert policy.shape == target_pis.shape
                assert value.shape == (batch_size, 1)
                assert value.shape == target_vs.shape
                loss = F.mse_loss(
                    input=value, target=target_vs, reduction="sum"
                ) - torch.sum(policy * target_pis)
                optimizer.zero_grad()
                loss = loss / batch_size
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

        board_tensor: torch.Tensor = torch.FloatTensor(board.astype(np.float64))
        if cuda:
            board_tensor = board_tensor.contiguous().cuda()
        board_tensor = board_tensor.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi: torch.Tensor
            v: torch.Tensor
            pi, v = self.nnet(board_tensor)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

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
