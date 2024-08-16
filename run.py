from Constants import START_POS_P1, GOAL_P1, START_POS_P2, GOAL_P2
from Heuristics import both_goals_evaluation_function, statistic_simulation_random_player, combined_heuristic
from Players import AlphaBetaPlayer, RandomPlayer, HeuristicPlayer
from game_faster import Quoridor


if __name__ == '__main__':

    # heuristic_player = HeuristicPlayer(
    #     id=1,
    #     pos=START_POS_P1,
    #     goal=GOAL_P1,
    #     evaluation_function=lambda x: statistic_simulation_random_player(x,10)
    # )
    # random_player = RandomPlayer(
    #     id=2,
    #     pos=START_POS_P2,
    #     goal=GOAL_P2,
    # )
    #
    # quoridor = Quoridor(heuristic_player, random_player)
    # result = quoridor.play_game()

    alphabeta_player = AlphaBetaPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        depth=2,
        evaluation_function=lambda x: both_goals_evaluation_function(x, 0.5),
        is_wall_first_game=True
    )
    random_player = RandomPlayer(
        id=2,
        pos=START_POS_P2,
        goal=GOAL_P2,
        is_wall_first_game=True
    )
    new_alphabeta_player = AlphaBetaPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        depth=2,
        evaluation_function=combined_heuristic,
        is_wall_first_game=True
    )

    quoridor = Quoridor(alphabeta_player, random_player)
    result = quoridor.play_game()

