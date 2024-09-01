import pickle
from numpy import average, number
import tqdm
import math
import time
from itertools import product

from Constants import START_POS_P1, GOAL_P1, START_POS_P2, GOAL_P2
from Heuristics import both_goals_evaluation_function, statistic_simulation_random_player,\
    walls_dist_heuristic, shortest_opponent_path, naive_self_dist_from_goal_evaluation_function, \
    shortest_self_dist_from_goal_evaluation_function, shortest_opponent_dist_from_goal_evaluation_function, \
    naive_opponent_dist_from_goal_evaluation_function, exp_shortest_opponent_dist_from_goal_evaluation_function, \
    exp_shortest_self_dist_from_goal_evaluation_function, prevent_loop_function
from Players import RandomPlayer, HeuristicPlayer, AlphaBetaPlayer
from game_faster import Quoridor
import matplotlib.pyplot as plt
import random
import datetime
from qlearning import QLearningPlayer

def get_time_date() -> str:
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

def naive_vs_shortest():
    for i in range(5):
        random.seed(1)
        alphabeta_1 = AlphaBetaPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
            depth=1,
            evaluation_function=lambda x: naive_self_dist_from_goal_evaluation_function(x) + (i+1)*naive_opponent_dist_from_goal_evaluation_function(x) - 100*prevent_loop_function(x),
        )
        alphabeta_2 = AlphaBetaPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
            depth=1,
            evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(x) + (i+1)*shortest_opponent_dist_from_goal_evaluation_function(x) - 100*prevent_loop_function(x),

        )
        quoridor = Quoridor(alphabeta_1, alphabeta_2)
        result = quoridor.play_game()
        print('winner')

    for i in range(5):
        random.seed(1)
        alphabeta_2 = AlphaBetaPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
            depth=1,
            evaluation_function=lambda x: naive_self_dist_from_goal_evaluation_function(
                x) + 2 * naive_opponent_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(x),
        )
        alphabeta_1 = AlphaBetaPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
            depth=1,
            evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(
                x) + 2 * shortest_opponent_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(x),

        )
        quoridor = Quoridor(alphabeta_1, alphabeta_2)
        result = quoridor.play_game()
        print('winner')


def exp_vs_normal():
    for i in range(5):
        alphabeta_1 = AlphaBetaPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
            depth=1,
            evaluation_function=lambda x:  shortest_self_dist_from_goal_evaluation_function(x) + (shortest_opponent_dist_from_goal_evaluation_function(x))**2 - 100*prevent_loop_function(x),
        )
        alphabeta_2 = AlphaBetaPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
            depth=1,
            evaluation_function=lambda x: exp_shortest_self_dist_from_goal_evaluation_function(x) + (exp_shortest_opponent_dist_from_goal_evaluation_function(x))**2- 100*prevent_loop_function(x),
        )
        quoridor = Quoridor(alphabeta_1, alphabeta_2)
        result = quoridor.play_game()
        print("winner")


def both_vs_self():
    alphabeta_1 = AlphaBetaPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        depth=1,
        evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(
            x) + shortest_opponent_dist_from_goal_evaluation_function(x) - 100 * prevent_loop_function(x),
    )
    alphabeta_2 = AlphaBetaPlayer(
        id=2,
        pos=START_POS_P2,
        goal=GOAL_P2,
        depth=1,
        evaluation_function=lambda x: shortest_self_dist_from_goal_evaluation_function(
            x) - 100 * prevent_loop_function(
            x))
    quoridor = Quoridor(alphabeta_1, alphabeta_2)
    result = quoridor.play_game()


def heuristic_simulation_vs_self_dist():
    sim = HeuristicPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        evaluation_function=lambda x: statistic_simulation_random_player(x, 40),
    )
    dist = RandomPlayer(
        id=2,
        pos=START_POS_P2,
        goal=GOAL_P2
    )
    quoridor = Quoridor(sim, dist)
    result = quoridor.play_game()


def opponent_factor_evaluation():
    num_of_turns = []
    for i in range(20):
        alphabeta = AlphaBetaPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
            evaluation_function=lambda x: both_goals_evaluation_function(x, (i - 5) / 5),
            depth=1
        )
        random = AlphaBetaPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
            evaluation_function=lambda x: both_goals_evaluation_function(x, (i - 5) / 5),
        )
        quoridor = Quoridor(alphabeta, random)
        result = quoridor.play_game()
        print('winner')
        print(result.total_moves)
        num_of_turns.append(result.total_moves)
    plt.plot([(i - 5) / 5 for i in range(20)], num_of_turns)
    plt.title("Game Length of different opponent factors")
    plt.xlabel("Opponent factor")
    plt.ylabel("Number of turns")
    plt.show()

def random_vs_random():
    num_of_turns = []
    for i in range(5):
        alphabeta = RandomPlayer(
            id=1,
            pos=START_POS_P1,
            goal=GOAL_P1,
        )
        random = RandomPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
        )
        quoridor = Quoridor(alphabeta, random)
        result = quoridor.play_game()
        print('winner')
        print(result.total_moves)
        num_of_turns.append(result.total_moves)
    plt.plot([(i - 5) / 5 for i in range(20)], num_of_turns)
    plt.title("Game Length of different opponent factors")
    plt.xlabel("Opponent factor")
    plt.ylabel("Number of turns")
    plt.show()

def random_vs_learning(alpha=1, epsilon=0.3, gamma=0.8, number_of_matches: int = 100, 
                       number_of_training_matches: int = 50):
    num_of_turns = []
    q_learner = QLearningPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        alpha=alpha,
        epsilon=epsilon,
        gamma=gamma
    )

    q_counter = 0
    num_of_turns = []
    match_number = []
    for i in range(number_of_matches):
        if (i == number_of_training_matches):
            q_learner.epsilon = 0.05
            print(f"wins before stopped learning: {q_counter}")
            q_counter = 0
        q_learner.pos = START_POS_P1
        q_learner.goal = GOAL_P1
        random = RandomPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2
        )
        quoridor = Quoridor(q_learner, random)
        result = quoridor.play_game(simulate=True)

        id = result.winner.id
        if (i >= number_of_training_matches):
            match_number.append(i - number_of_training_matches + 1)
            num_of_turns.append(result.total_moves)
        if (id == 1):
            q_counter += 1
       
    model_parameters = f'alpha-{alpha}-gamma-{gamma}' + \
                    f'-epsilon-{epsilon}'
    
    mean_num_of_turns = average(num_of_turns)
    plt.figure()
    plt.scatter(match_number, num_of_turns)
    plt.axhline(y = mean_num_of_turns, color='r', linestyle='--', label=f'mean = {mean_num_of_turns}')
    plt.title(rf"$\alpha = {alpha}$, $\gamma = {gamma}$, $\epsilon={epsilon}$, trained for {number_of_training_matches} matches,"
              + f" winrate: {q_counter / (number_of_matches - number_of_training_matches)}")
    plt.xlabel("Match number")
    plt.ylabel("Number of turns until winning")
    plt.legend()
    plt.savefig(f"graphs/trained-for-{number_of_training_matches}-{get_time_date()}-{model_parameters}-scatter.jpg")
    plt.close()
    # plt.plot(match_number, num_of_turns)
    # plt.title("Game Length of different opponent factors")
    # plt.xlabel("Match number")
    # plt.ylabel("Number of turns until winning")
    # plt.savefig(f"trained-for-{number_of_training_matches}-{get_time_date()}" +
    #             f"-{model_parameters}-plot.png")
    # print(f"wins after stopped learning: {q_counter}")
    # print(f"mean number of turns: {mean_num_of_turns}")

    # with open(
    #     f"q-values/q-values-{get_time_date()}-trained-for-{number_of_training_matches}-{model_parameters}.pkl",
    #           'wb') as file:
    #     pickle.dump(q_learner.q_values, file)


    
def learning_vs_alphabeta(q_values_path: str, depth: int, number_of_matches: int = 10):
    q_learner = QLearningPlayer(
        id=1,
        pos=START_POS_P1,
        goal=GOAL_P1,
        epsilon=0.1
    )

    q_learner.import_q_values(q_values_path)

    q_counter = 0
    for i in tqdm.tqdm(range(number_of_matches)):
        q_learner.pos = START_POS_P1
        q_learner.goal = GOAL_P1
        random = AlphaBetaPlayer(
            id=2,
            pos=START_POS_P2,
            goal=GOAL_P2,
            depth=depth,
            evaluation_function=lambda x: both_goals_evaluation_function(x, (i - 5) / 5),
        )
        quoridor = Quoridor(q_learner, random)
        result = quoridor.play_game(simulate=True)

        id = result.winner.id

        if (id == 1):
            q_counter += 1
            # print("victory")
        

        print(f"game {i} ended")
        print(f"player with id: {id} won")
    print(f"q_learner wins: {q_counter}")



if __name__ == '__main__':
    gammas = {0.2, 0.5, 0.8}
    training_matches_numbers ={10} 
    epsilons = {0.1, 0.3, 0.5}
    alphas = {1, 0.5, 0.75}
    evalutaing_matches_number = 50 
    random_vs_learning(number_of_matches=evalutaing_matches_number+50, 
                        number_of_training_matches=50,
                        gamma=0.5
        )
    # for training_matches, epsilon, alpha, gamma in tqdm.tqdm(list(product(training_matches_numbers, 
    #                                                                  epsilons, alphas, gammas))):
    #     random_vs_learning(number_of_matches=evalutaing_matches_number+training_matches, 
    #                         number_of_training_matches=training_matches,
    #                         alpha=alpha,
    #                         epsilon=epsilon,
    #                         gamma=gamma
    #         )

    # opponent_factor_evaluation()
    # learning_vs_alphabeta(r'/home/ec2-user/quoridor-ai/q-values-20240824-172759-trained-for-50.0',
                        #   2)


