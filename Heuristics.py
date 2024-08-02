def null_evaluation_function(game_state):
    return 0

def self_dist_from_goal_evaluation_function(game_state):
    return -get_player_dist_from_goal(game_state.current_player)


def opponent_dist_from_goal_evaluation_function(game_state):
    return get_player_dist_from_goal(game_state.waiting_player)


def get_player_dist_from_goal(player):
    return abs(int(player.pos[-1]) - int(player.goal))


def both_goals_evaluation_function(game_state):
    lose_punishment_factor = 2
    return self_dist_from_goal_evaluation_function(
        game_state) + lose_punishment_factor * opponent_dist_from_goal_evaluation_function(game_state)