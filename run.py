from Constants import START_POS_P1, GOAL_P1, START_POS_P2, GOAL_P2
from Heuristics import statistic_simulation_random_player, naive_opponent_dist_from_goal_evaluation_function, \
    naive_self_dist_from_goal_evaluation_function, shortest_self_dist_from_goal_evaluation_function, \
    shortest_opponent_dist_from_goal_evaluation_function, prevent_loop_function
from Players import AlphaBetaPlayer, RandomPlayer, HeuristicPlayer
from game_faster import Quoridor


def naive_vs_shortest():
    alphabeta_1 = AlphaBetaPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        depth=2,
        evaluation_function=lambda x: naive_self_dist_from_goal_evaluation_function(
            x) + 1.5 * naive_opponent_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(x),
    )
    alphabeta_2 = AlphaBetaPlayer(
        id=2,
        pos=START_POS_P2,
        goal=GOAL_P2,
        depth=2,
        evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(
            x) + 1.5 * shortest_opponent_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(x),

    )
    quoridor = Quoridor(alphabeta_1, alphabeta_2)
    result = quoridor.play_game()


def both_vs_self():
    alphabeta_1 = AlphaBetaPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        depth=1,
        evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(
            x) + shortest_opponent_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(x),
        is_wall_first_game=False
    )
    alphabeta_2 = AlphaBetaPlayer(
        id=2,
        pos=START_POS_P2,
        goal=GOAL_P2,
        depth=1,
        evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(
            x),
        is_wall_first_game=False
    )
    quoridor = Quoridor(alphabeta_1, alphabeta_2)
    result = quoridor.play_game()


def heuristic_simulation_vs_self_dist():
    sim = HeuristicPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        evaluation_function=lambda x: statistic_simulation_random_player(x, 40),
        is_wall_first_game=False,
        # just_movement = True
    )
    dist = RandomPlayer(
        id=2,
        pos=START_POS_P2,
        goal=GOAL_P2,
        is_wall_first_game=False
    )
    quoridor = Quoridor(sim, dist)
    result = quoridor.play_game()


if __name__ == '__main__':
    both_vs_self()
    # heuristic_simulation_vs_self_dist()
    # naive_vs_shortest()
    # exp_vs_normal()
