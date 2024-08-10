from Constants import START_POS_P1, GOAL_P1, START_POS_P2, GOAL_P2
from Heuristics import both_goals_evaluation_function, statistic_simulation_random_player
from Players import AlphaBetaPlayer, RandomPlayer, HeuristicPlayer
from game_faster import Quoridor


if __name__ == '__main__':

    heuristic_player = HeuristicPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        evaluation_function=lambda x: statistic_simulation_random_player(x,10)
    )
    random_player = RandomPlayer(
        id=2,
        pos=START_POS_P2,
        goal=GOAL_P2,
    )

    quoridor = Quoridor(heuristic_player, random_player)
    result = quoridor.play_game()

