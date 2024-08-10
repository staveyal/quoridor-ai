from Constants import GOAL_P2, GOAL_P1, START_POS_P1, START_POS_P2
from Players import RandomPlayer
from game_faster import Quoridor


def null_evaluation_function(game_state):
    return 0

def self_dist_from_goal_evaluation_function(game_state):
    return -__get_player_dist_from_goal(game_state.current_player)


def opponent_dist_from_goal_evaluation_function(game_state):
    return __get_player_dist_from_goal(game_state.waiting_player)


def __get_player_dist_from_goal(player):
    return abs(int(player.pos[-1]) - int(player.goal))


def both_goals_evaluation_function(game_state,opponent_factor):
    return self_dist_from_goal_evaluation_function(
        game_state) + opponent_factor * opponent_dist_from_goal_evaluation_function(game_state)

def statistic_simulation_random_player(game_state, num_to_simulate):
    for _ in range(num_to_simulate):
        random_player_1 = RandomPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
        )
        random_player_2 = RandomPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
        )

        quoridor = Quoridor(random_player_1, random_player_2)
        result = quoridor.play_game()