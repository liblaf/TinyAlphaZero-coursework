import sys

from util import *

sys.path.append("..")

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

net_config = dotdict(
    {
        "lr": 0.001,
        "dropout": 0.3,
        "epochs": 10,
        "batch_size": 64,
        "cuda": torch.cuda.is_available(),
        "num_channels": 512,
    }
)


class GoNNet(nn.Module):
    def __init__(self, game, args):
        super(GoNNet, self).__init__()
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################

    def forward(self, s):
        ######################################
        #        YOUR CODE GOES HERE         #
        ######################################
        return None


class GoNNetWrapper:
    def __init__(self, game):
        self.nnet = GoNNet(game, net_config)
        self.board_x, self.board_y = game.obs_size()
        self.action_size = game.action_size()

        if net_config.cuda:
            self.nnet.cuda()

    def train(self, training_data):
        """
        training_data: list of (board, pi, v)
        """
        optimizer = optim.Adam(self.nnet.parameters(), lr=net_config.lr)

        for epoch in range(net_config.epochs):
            print(f"Epoch[{str(epoch + 1)}] ")
            self.nnet.train()

            batch_count = int(len(training_data) / net_config.batch_size)

            t = tqdm(range(batch_count), desc="Training NNet")
            for _ in t:
                sample_ids = np.random.randint(
                    len(training_data), size=net_config.batch_size
                )
                boards, pis, vs = list(zip(*[training_data[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                if net_config.cuda:
                    boards, target_pis, target_vs = (
                        boards.contiguous().cuda(),
                        target_pis.contiguous().cuda(),
                        target_vs.contiguous().cuda(),
                    )

                ######################################
                #        YOUR CODE GOES HERE         #
                ######################################
                # Compute loss and backprop

    def predict(self, board):
        """
        predict the policy and value of the given board
        @param board: np array
        @return (pi, v): a policy vector for the given board, and a float value
        """
        board = torch.FloatTensor(board.astype(np.float64))
        if net_config.cuda:
            board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)
        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet(board)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def save_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
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

    def load_checkpoint(self, folder="checkpoint", filename="checkpoint.pth.tar"):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception("No model in path {}".format(filepath))
        map_location = None if net_config.cuda else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint["state_dict"])
