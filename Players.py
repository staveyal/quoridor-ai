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
    def get_action(self, game_state):
        return random.choice(game_state.get_legal_moves())


class HeuristicPlayer(Player):
    def __init__(self, id, pos, goal, evaluation_function, walls=START_WALLS, position_history=None, placed_walls=None):
        super().__init__(id, pos, goal, walls, position_history, placed_walls)
        self.evaluation_function = evaluation_function
        self.position_history = []
        self.placed_walls = []

    def get_action(self, game_state):
        best_move = game_state.get_legal_moves()[0]
        best_score = -math.inf
        for move in game_state.get_legal_moves():
            game_state.make_move(move)
            score = self.evaluate_state(game_state)
            # print(f"pos: {game_state.current_player.pos}, move: {move}, score:{score}")
            game_state.undo_move()
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def evaluate_state(self, game_state):
        game_state._switch_player()
        score = self.evaluation_function(game_state)
        game_state._switch_player()
        return score


class AlphaBetaPlayer(Player):
    def __init__(self, id, pos, goal, evaluation_function, walls=START_WALLS, position_history=None, placed_walls=None, depth=1):
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
            return (np.inf, "") if not is_max else (-np.inf, "")
        if depth == 0:
            return self.evaluation_function(game_state), game_state.get_legal_moves()[0]
        value = -np.inf if is_max else np.inf
        action = ""
        filtered = self.filter_moves(game_state.get_legal_moves(), game_state)
        # print(filtered)
        for next_action in filtered:
            game_state.make_move(next_action)
            next_value, _ = self.__recursive_minimax(game_state, depth - 1 if not is_max else depth, not is_max, value)
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

    def dist_from_cell(self, move, pos):
        return max(abs(ord(move[0]) - ord(pos[0])), abs(ord(move[1]) - ord(pos[1])))

    def filter_moves(self, legal_moves, game_state):
        filtered_moves = []
        for move in legal_moves:
            if len(move) == 2:
                filtered_moves.append(move)
            elif self.dist_from_cell(move, game_state.current_player.pos) <= 1:
                filtered_moves.append(move)
            elif self.dist_from_cell(move, game_state.waiting_player.pos) <= 1:
                filtered_moves.append(move)
            else:
                for wall in game_state.placed_walls:
                    if self.dist_from_cell(move, wall[:2]) <= 1:
                        filtered_moves.append(move)
        return filtered_moves


def smaller_or_equals_with_chance(value1, value2):
    if value1 == value2:
        return random.choice([True, False])
    return value1 < value2



