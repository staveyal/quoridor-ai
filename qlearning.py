"""
# Q-Learning
In this file, we implement the gym environment required in order to train our model. 
We wrap the Quoridor game environment with it, and use it to train against other players.
"""
from collections import defaultdict
import numpy as np
from Constants import START_WALLS
from game_faster import Quoridor
from Players import Player

class QLearningPlayer(Player):
    def __init__(self, id, pos, goal, evaluation_function, walls=START_WALLS,
                 position_history=[], placed_walls=[], 
                 alpha: float =1.0, epsilon: float =0.05, gamma: float=0.8, 
                 num_training: int = 10):
        super().__init__(id, pos, goal, walls, position_history, placed_walls)
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = gamma
        self.num_training = num_training
        # A dictionary that returns 0 on non existent keys
        self.q_values = defaultdict(float) 
        
    def stop_learning(self):
        self.alpha, self.epsilon = 0, 0

    def get_q_value(self, game_state: Quoridor, action: str) -> float:
        """
        Returns Q(state,action)
        Should return 0.0 if we never seen
        a state or (state,action) tuple
        """
        state_action_pair = (str(game_state), action)
        return self.q_values[state_action_pair]

    def get_policy(self, game_state: Quoridor) -> str:
        """
        Compute the best action to take in a state.  Note that if there
        are no legal actions, which is the case at the terminal state,
        you should return None.
        """
        legal_actions = game_state.get_legal_moves()

        best_action = legal_actions[0] 
        best_value = float('-inf')

        for action in legal_actions:
            q_value = self.get_q_value(game_state, action)
            if q_value > best_value:
                best_value = q_value
                best_action = action

        return best_action

    def get_action(self, game_state: Quoridor) -> str:
        """
        Get the action to play on the board based on epsilon-greedy policy.
        """
        legal_actions = game_state.get_legal_moves()

        # With probability epsilon, choose a random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)

        # Otherwise, choose the best action according to the current policy
        return self.get_policy(game_state)

    def get_value(self, game_state: Quoridor):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        legal_actions = game_state.get_legal_moves()

        return max(
            self.get_q_value(game_state, action) for action in legal_actions
        )

    
    def update(self, state: Quoridor, action: str, nextState: Quoridor, reward: float):
        """
        This function observes a state,action => nextState and reward transition.
        Since we can play the game by ourselves - we can just take state.make_move(action)
        and then do state.undo_move()
        """

        state_action_pair = (str(state), action)
        next_value = self.get_value(nextState)
        old_value = self.q_values[state_action_pair]

        self.q_values[state_action_pair] = \
            old_value + self.alpha * (reward + self.discount * next_value - old_value)



        

     