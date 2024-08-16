import math

from Constants import GOAL_P2, GOAL_P1
from Players import RandomPlayer
from game_faster import Quoridor


def null_evaluation_function(game_state):
    return 0


def self_dist_from_goal_evaluation_function(game_state):
    return -math.exp(-__get_player_dist_from_goal(game_state.current_player))


def opponent_dist_from_goal_evaluation_function(game_state):
    return math.exp(__get_player_dist_from_goal(game_state.waiting_player))


def __get_player_dist_from_goal(player):
    return abs(int(player.pos[-1]) - int(player.goal))


def both_goals_evaluation_function(game_state, opponent_factor=1):
    return self_dist_from_goal_evaluation_function(
        game_state) + opponent_factor * opponent_dist_from_goal_evaluation_function(game_state)


def prevent_loop_function(game_state, opponent_factor):
    loop_penalty = len(game_state.current_player.position_history) - len(set(game_state.current_player.position_history))
    return self_dist_from_goal_evaluation_function(
        game_state) + opponent_factor * opponent_dist_from_goal_evaluation_function(game_state) - loop_penalty * 10


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
        quoridor._switch_player()
        result = quoridor.play_game(simulate=True)
        wins[result.winner.id] += 1
    return wins[0] / num_to_simulate


def clean_board_ahead_player(game_state):
    count = 0
    for wall in game_state.placed_walls:
        if game_state.current_player.position < wall[1] < game_state.current_player.goal or \
                game_state.current_player.position > wall[1] > game_state.current_player.goal:
            count += 1
    return -count


def dirty_board_ahead_opponent(game_state):
    count = 0
    for wall in game_state.placed_walls:
        if game_state.waiting_player.goal < wall[1] < game_state.waiting_player.position or \
                game_state.waiting_player.goal > wall[1] > game_state.waiting_player.position:
            count += 1
    return count


def opponent_walls(game_state):
    return -game_state.waiting_player.walls


def wall_efficiency_heuristic(game_state):
    opponent = game_state.waiting_player
    current_path_length = len(game_state.get_shortest_path(opponent.pos, opponent.goal))
    total_path_increase = 0

    for wall in game_state.current_player.placed_walls:
        # Create a temporary copy of the board
        temp_board = {k: v[:] for k, v in game_state.board.items()}

        # Temporarily remove the wall
        game_state._remove_connections(temp_board, wall)

        # Calculate the new path length without the wall
        new_path_length = len(game_state.get_shortest_path(opponent.pos, opponent.goal))

        # Calculate the increase in path length
        path_increase = new_path_length - current_path_length
        total_path_increase += path_increase

        # We don't need to add the wall back since we're working with a copy

    return total_path_increase


def blocking_opponent_path_heuristic(game_state):
    opponent = game_state.waiting_player
    opponent_path = game_state.get_shortest_path(opponent.pos, opponent.goal)
    blocking_walls = 0

    for wall in game_state.current_player.placed_walls:
        if any(wall[:2] == step or wall[2:] == step for step in opponent_path):
            blocking_walls += 1

    return blocking_walls


def combined_heuristic(game_state):
    return (
        both_goals_evaluation_function(game_state) +
        2 * wall_efficiency_heuristic(game_state) +
        blocking_opponent_path_heuristic(game_state) -
        0.5 * opponent_walls(game_state)
    )