from Constants import GOAL_P2, GOAL_P1
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


def both_goals_evaluation_function(game_state, opponent_factor):
    return self_dist_from_goal_evaluation_function(
        game_state) + opponent_factor * opponent_dist_from_goal_evaluation_function(game_state)


def statistic_simulation_random_player(game_state, num_to_simulate):
    wins = [0, 0]
    for _ in range(num_to_simulate):
        random_player_1 = RandomPlayer(
            id=0,
            pos=game_state.current_player.pos,
            goal=game_state.current_player.goal,
        )
        random_player_2 = RandomPlayer(
            id=1,
            pos=game_state.waiting_player.pos,
            goal=game_state.waiting_player.goal,
        )
        quoridor = Quoridor(random_player_1, random_player_2)

        for wall in game_state.placed_walls:
            quoridor._make_wall_move(quoridor.board, wall)
        result = quoridor.play_game(simulate=True)
        wins[result.winner.id] += 1
    return wins[0] / num_to_simulate
