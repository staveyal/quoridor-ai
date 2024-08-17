import math
import time

from Constants import START_POS_P1, GOAL_P1, START_POS_P2, GOAL_P2
from Heuristics import both_goals_evaluation_function, statistic_simulation_random_player,\
    walls_dist_heuristic, shortest_opponent_path, naive_self_dist_from_goal_evaluation_function, \
    shortest_self_dist_from_goal_evaluation_function, shortest_opponent_dist_from_goal_evaluation_function, \
    naive_opponent_dist_from_goal_evaluation_function, exp_shortest_opponent_dist_from_goal_evaluation_function, \
    exp_shortest_self_dist_from_goal_evaluation_function, prevent_loop_function
from Players import RandomPlayer, HeuristicPlayer, AlphaBetaPlayer
from game_faster import Quoridor
import matplotlib.pyplot as plt
import random

def naive_vs_shortest():
    for i in range(5):
        random.seed(1)
        alphabeta_1 = AlphaBetaPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
            depth=1,
            evaluation_function=lambda x: naive_self_dist_from_goal_evaluation_function(x) + (i+1)*naive_opponent_dist_from_goal_evaluation_function(x) - 100*prevent_loop_function(x),
        )
        alphabeta_2 = AlphaBetaPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
            depth=1,
            evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(x) + (i+1)*shortest_opponent_dist_from_goal_evaluation_function(x) - 100*prevent_loop_function(x),

        )
        quoridor = Quoridor(alphabeta_1, alphabeta_2)
        result = quoridor.play_game()
        print('winner')

    for i in range(5):
        random.seed(1)
        alphabeta_2 = AlphaBetaPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
            depth=1,
            evaluation_function=lambda x: naive_self_dist_from_goal_evaluation_function(
                x) + 2 * naive_opponent_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(x),
        )
        alphabeta_1 = AlphaBetaPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
            depth=1,
            evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(
                x) + 2 * shortest_opponent_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(x),

        )
        quoridor = Quoridor(alphabeta_1, alphabeta_2)
        result = quoridor.play_game()
        print('winner')


def exp_vs_normal():
    for i in range(5):
        alphabeta_1 = AlphaBetaPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
            depth=1,
            evaluation_function=lambda x:  shortest_self_dist_from_goal_evaluation_function(x) + (shortest_opponent_dist_from_goal_evaluation_function(x))**2 - 100*prevent_loop_function(x),
        )
        alphabeta_2 = AlphaBetaPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
            depth=1,
            evaluation_function=lambda x: exp_shortest_self_dist_from_goal_evaluation_function(x) + (exp_shortest_opponent_dist_from_goal_evaluation_function(x))**2- 100*prevent_loop_function(x),
        )
        quoridor = Quoridor(alphabeta_1, alphabeta_2)
        result = quoridor.play_game()
        print("winner")


def both_vs_self():
    alphabeta_1 = AlphaBetaPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        depth=1,
        evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(
            x) + shortest_opponent_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(x),
    )
    alphabeta_2 = AlphaBetaPlayer(
        id=2,
        pos=START_POS_P2,
        goal=GOAL_P2,
        depth=1,
        evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(
            x) - 100 * prevent_loop_function(
            x))
    quoridor = Quoridor(alphabeta_1, alphabeta_2)
    result = quoridor.play_game()


def heuristic_simulation_vs_self_dist():
    sim = HeuristicPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        evaluation_function=lambda x: statistic_simulation_random_player(x, 40),
    )
    dist = RandomPlayer(
        id=2,
        pos=START_POS_P2,
        goal=GOAL_P2
    )
    quoridor = Quoridor(sim, dist)
    result = quoridor.play_game()


def opponent_factor_evaluation():
    num_of_turns = []
    for i in range(20):
        alphabeta = AlphaBetaPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
            evaluation_function=lambda x: both_goals_evaluation_function(x, (i - 5) / 5),
            depth=1
        )
        random = AlphaBetaPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
            evaluation_function=lambda x: both_goals_evaluation_function(x, (i - 5) / 5),
        )
        quoridor = Quoridor(alphabeta, random)
        result = quoridor.play_game()
        print('winner')
        print(result.total_moves)
        num_of_turns.append(result.total_moves)
    plt.plot([(i - 5) / 5 for i in range(20)], num_of_turns)
    plt.title("Game Length of different opponent factors")
    plt.xlabel("Opponent factor")
    plt.ylabel("Number of turns")
    plt.show()


if __name__ == '__main__':
    opponent_factor_evaluation()

