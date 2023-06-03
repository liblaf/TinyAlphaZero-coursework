import argparse
from typing import Union

from Player import *
from TicTacToeGame import TicTacToeGame
from tqdm import trange

AbstractPlayer = Union[
    RandomPlayer,
    AlphaBetaTicTacToePlayer,
    MCTSTicTacToePlayer,
    HumanTicTacToePlayer,
]


game_parser: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
game_parser.add_argument("--width", "-w", dest="width", default=3, type=int)
game_parser.add_argument("--height", dest="height", default=3, type=int)


eval_parser = argparse.ArgumentParser(add_help=False)
eval_parser.add_argument(
    "player1", type=str, choices=["AlphaBeta", "Human", "MCTS", "Random"]
)
eval_parser.add_argument(
    "player2", type=str, choices=["AlphaBeta", "Human", "MCTS", "Random"]
)
eval_parser.add_argument("--n-test", dest="n_test", default=10000, type=int)
eval_parser.add_argument(
    "--bilateral", dest="bilateral", action="store_true", default=True
)
eval_parser.add_argument(
    "--no-bilateral",
    dest="bilateral",
    action="store_false",
    default=True,
)

parser = argparse.ArgumentParser(parents=[game_parser, eval_parser])
parser.add_argument("--params", dest="params", default=None, type=str)


N_TEST: int = 80
BOARD_WIDTH: int = 3
BOARD_HEIGHT: int = 3
WIN_NUM: int = 3
N_PLAY_OUT: int = 50


game: TicTacToeGame = TicTacToeGame(h=BOARD_HEIGHT, w=BOARD_WIDTH, n=WIN_NUM)
random_player: RandomPlayer = RandomPlayer(game, 1)
alphabeta_player: AlphaBetaTicTacToePlayer = AlphaBetaTicTacToePlayer(game, 1)
mcts_player: MCTSTicTacToePlayer = MCTSTicTacToePlayer(
    game, 1, n_playout=N_PLAY_OUT, C=1
)
human_player: HumanTicTacToePlayer = HumanTicTacToePlayer(game, 1)


PLAYER_DICT: dict[str, AbstractPlayer] = {
    "AlphaBeta": alphabeta_player,
    "Human": human_player,
    "MCTS": mcts_player,
    "Random": random_player,
}


def single_match(
    player1: AbstractPlayer,
    player2: AbstractPlayer,
    game: TicTacToeGame,
    display: bool = False,
) -> list[int]:
    """
    Do a single match between two players, return scores.
    @return: one-hot array: [player1-win, draw, player2-win]
    """
    player1.player, player2.player = 1, -1
    state = game.reset()
    if display:
        game.display(state)
        print("---------------------------------")
    score = [0, 0, 0]  # player1-win, draw, player2-win

    while True:
        # player 1 move
        action = player1.play(state)
        state = game.next_state(state, 1, action)

        if display:
            game.display(state)
            print("---------------------------------")

        # player 2 move
        if game.is_terminal(state, 1) == 0:  # not end
            action = player2.play(state)
            state = game.next_state(state, -1, action)

        if display:
            game.display(state)
            print("---------------------------------")

        # if end game
        game_end = game.is_terminal(state, -1)  # return -1 for player 1 win
        if game_end != 0:
            score[(int(game_end // 1) + 1)] += 1
            break

    return score


def test_multi_match(
    player1: AbstractPlayer,
    player2: AbstractPlayer,
    n_test: int = 100,
    print_result: bool = True,
    bilateral: bool = True,
) -> tuple[int, int, int]:
    player1_win, player2_win, draw = 0, 0, 0

    for _ in trange(n_test // 2 if bilateral else n_test):
        score = single_match(player1, player2, game)
        player1_win += score[0]
        player2_win += score[2]
        draw += score[1]

    for _ in trange(n_test // 2 if bilateral else 0):  # reverse side
        score = single_match(player2, player1, game)
        player1_win += score[2]
        player2_win += score[0]
        draw += score[1]

    tot_match = (n_test // 2) * 2
    if print_result:
        print("Test result: ")
        print(
            f"    player1({str(player1)})-win:",
            player1_win,
            f"{100 * player1_win / tot_match:.2f}%",
        )
        print(
            f"    player2({str(player2)})-win:",
            player2_win,
            f"{100 * player2_win / tot_match:.2f}%",
        )
        print("    draw:", draw, f"{100 * draw / tot_match:.2f}%")
    return player1_win, player2_win, draw


if __name__ == "__main__":
    args = parser.parse_args()
    game.w = args.width
    game.h = args.height
    N_TEST = args.n_test
    if args.params is not None:
        mcts_player.load_params(args.params)
    mcts_player.eval()
    player1 = PLAYER_DICT[args.player1]
    player2 = PLAYER_DICT[args.player2]
    test_multi_match(
        player1=player1, player2=player2, n_test=N_TEST, bilateral=args.bilateral
    )

    # ** Modify this part to test different players **
    # mcts_player.eval()
    # player1 = mcts_player
    # player2 = alphabeta_player
    # # test win-lose rate
    # test_multi_match(player1, player2, N_TEST)

    # visualize games
    # player1 play first
    # single_match(player1, player2, game, display=True)
    # player2 play first
    # single_match(player2, player1, game, display=True)

    # play with human
    # player1 = alphabeta_player
    # player2 = human_player
    # single_match(player1, player2, game, display=True)
