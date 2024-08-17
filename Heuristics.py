import math

from Constants import GOAL_P2, GOAL_P1
from Players import RandomPlayer
from game_faster import Quoridor


def null_evaluation_function(game_state):
    return 0


def exp_shortest_self_dist_from_goal_evaluation_function(game_state):
    return math.exp(-len(game_state.get_shortest_path(game_state.current_player.pos,game_state.current_player.goal)))

def shortest_self_dist_from_goal_evaluation_function(game_state):
    return -len(game_state.get_shortest_path(game_state.current_player.pos,game_state.current_player.goal))

def naive_self_dist_from_goal_evaluation_function(game_state):
    return -naive_player_dist_from_goal(game_state.current_player)

def exp_shortest_opponent_dist_from_goal_evaluation_function(game_state):
    return math.exp(-len(game_state.get_shortest_path(game_state.waiting_player.pos,game_state.waiting_player.goal)))

def shortest_opponent_dist_from_goal_evaluation_function(game_state):
    return -len(game_state.get_shortest_path(game_state.waiting_player.pos,game_state.waiting_player.goal))

def naive_opponent_dist_from_goal_evaluation_function(game_state):
    return -naive_player_dist_from_goal(game_state.waiting_player)

def naive_player_dist_from_goal(player):
    return abs(int(player.pos[-1]) - int(player.goal))


def both_goals_evaluation_function(game_state, opponent_factor=1):
    return self_dist_from_goal_evaluation_function(
        game_state) + opponent_factor * opponent_dist_from_goal_evaluation_function(game_state)


def prevent_loop_function(game_state):
    loop_penalty = len(game_state.current_player.position_history + [game_state.current_player.pos]) - \
                   len(set(game_state.current_player.position_history + [game_state.current_player.pos]))
    return loop_penalty

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


def walls_dist_heuristic(game_state):
    opponent = game_state.waiting_player
    total_dist = 0

    for wall in game_state.placed_walls:
        total_dist += max(abs(ord(opponent.pos[0]) - ord(wall[0])), abs(ord(opponent.pos[1]) - ord(wall[1])))

    return -total_dist


def blocking_opponent_path_heuristic(game_state):
    opponent = game_state.waiting_player
    opponent_path = game_state.get_shortest_path(opponent.pos, opponent.goal)
    blocking_walls = 0

    for wall in game_state.current_player.placed_walls:
        if any(wall[:2] == step or wall[2:] == step for step in opponent_path):
            blocking_walls += 1

    return blocking_walls

def shortest_opponent_path(game_state):
    return len(game_state.get_shortest_path(game_state.waiting_player.pos, game_state.waiting_player.goal))

def combined_heuristic(game_state):
    return (
        self_dist_from_goal_evaluation_function(game_state)
    )