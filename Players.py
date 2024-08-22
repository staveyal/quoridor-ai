import math
import random
from dataclasses import field, dataclass
from typing import List

import numpy as np

from Constants import START_WALLS, GameStatus

@dataclass
class Player:
    """
    Represents a player in the Quoridor game.

    Attributes
    ----------
    id : int
        The player's ID.
    pos : str
        The player's current position on the board.
    goal : str
        The player's goal on the board.
    walls : int, optional
        The number of walls the player has, by default `START_WALLS`.
    position_history : list of str, optional
        A list of postions the player has been ordered by turn, by default `[]`
    placed_walls : list of str, optional
        A list of walls the player has placed, by default `[]`.
    """

    id: int
    pos: str
    goal: str
    walls: int = START_WALLS
    position_history: List[str] = field(default_factory=lambda: [])
    placed_walls: List[str] = field(default_factory=lambda: [])

    def get_action(self, game_state):
        return input("Your move: ")


class RandomPlayer(Player):
    """
    Random player that half of the turns choose a random pawn move and otherwise choose a random legal action
    """
    def get_action(self, game_state):
        if random.random() < .5:
            moves = list(game_state.get_legal_pawn_moves())
        else:
            moves = filter_moves(game_state)
        return random.choice(moves)

class HeuristicPlayer(Player):
    """
    Player that choose every turn the best move according to a given evaluation function
    """
    def __init__(self, id, pos, goal, evaluation_function,walls=START_WALLS,
                 position_history=[], placed_walls=[], just_movement=False):
        super().__init__(id, pos, goal, walls, position_history, placed_walls)
        self.evaluation_function = evaluation_function
        self.just_movement = just_movement
        self.position_history = []
        self.placed_walls = []
        self.branching_factors = []

    def get_action(self, game_state):
        if self.just_movement: # So it would make moves and not only walls # random.random() < .5 or
            moves = list(game_state.get_legal_pawn_moves())
        else:
            if len(filter_moves(game_state)) < 10:
                print('h')
            moves = filter_moves(game_state)
        best_move = moves[0]
        best_score = -math.inf
        self.branching_factors.append(len(moves))

        for move in moves:
            game_state.make_move(move)
            score = self.__evaluate_state(game_state)
            print(f"move: {move}, score:{score}")
            game_state.undo_move()
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def __evaluate_state(self, game_state):
        game_state._switch_player()
        score = self.evaluation_function(game_state)
        game_state._switch_player()
        return score


class AlphaBetaPlayer(Player):
    """
    Minimax player that uses alpha beta prunning
    """
    def __init__(self, id, pos, goal, evaluation_function,walls=START_WALLS, position_history=None, placed_walls=None, depth=1):
        super().__init__(id, pos, goal, walls, position_history, placed_walls)
        self.depth = depth
        self.position_history = []
        self.placed_walls = []
        self.evaluation_function = evaluation_function

    def get_action(self, game_state):
        value, action = self.__recursive_minimax(game_state, self.depth, True, np.inf)
        return action

    def __recursive_minimax(self, game_state, depth, is_max, best_other):
        if game_state.status == GameStatus.COMPLETED:
            if depth == 2:
                print('a')
            return (np.inf, "") if not is_max else (-np.inf, "")
        if depth <= 0:
            return self.evaluation_function(game_state), game_state.get_legal_moves()[0]
        value = -np.inf if is_max else np.inf
        filtered = filter_moves(game_state)
        action = filtered[0]

        for next_action in filtered:
            game_state.make_move(next_action)
            depth_sub = 1 if len(next_action) == 2 else 3
            next_value, _ = self.__recursive_minimax(game_state, depth - depth_sub if not is_max else depth, not is_max, value)
            game_state.undo_move()
            if is_max:
                if smaller_or_equals_with_chance(value, next_value):
                    value = next_value
                    action = next_action
                if best_other < value:
                    break
            else:
                if not smaller_or_equals_with_chance(value, next_value):
                    value = next_value
                    action = next_action
                if best_other > value:
                    break
        return value, action

def dist_from_cell(move, pos):
    """
    Measures the distance between a given move and a given pos
    """
    return max(abs(ord(move[0]) - ord(pos[0])), abs(ord(move[1]) - ord(pos[1])))


def filter_moves(game_state):
    """
    Filter the legal moves based on our assumption about the game: walls should be placed near other walls/players
    Used in order to reduce the branching factor and speed up the agents
    """
    filtered_moves = []
    for move in game_state.get_legal_moves():
        if len(move) == 2:
            filtered_moves.append(move)
        elif dist_from_cell(move, game_state.current_player.pos) <= 1:
            filtered_moves.append(move)
        elif dist_from_cell(move, game_state.waiting_player.pos) <= 1:
            filtered_moves.append(move)
        else:
            for wall in game_state.placed_walls:
                if dist_from_cell(move, wall[:2]) <= 1:
                    filtered_moves.append(move)
                    break
    return filtered_moves


def smaller_or_equals_with_chance(value1, value2):
    """
    Tie breaking comparison that returns a random result if the values are equal
    """
    if value1 == value2:
        return random.choice([True, False])
    return value1 < value2



