import jax.numpy as jnp
import numpy as np
from agents_engineered import pi_random

def zero_layer_params(m, n):
    return jnp.ones(shape=(n, m)), jnp.ones(shape=(n,))


def init_eligibility(sizes):
    return [zero_layer_params(m, n) for m, n in zip(sizes[:-1], sizes[1:])]


def update_eligibility(eligibility, decay, grad_value):
    return [(decay * z_w + dv_dw, decay * z_b + dv_db) for (z_w, z_b), (dv_dw, dv_db) in zip(eligibility, grad_value)]


def train(agent, game, episodes, learning_rate, epsilon, lmbda, discount=1, checks=False, iprint=100):
    outcomes = np.zeros(shape=episodes)

    for episode in range(episodes):
        game.reset()
        finished = False
        eligibility = init_eligibility([agent.input_units, agent.hidden_units, 1])
        new_value = agent.value(game.board)

        while not finished:
            value = new_value
            grad_value = agent.value_gradient(game.board)

            game.play_move(agent.policy(game, epsilon), checks)

            reward = game.reward()
            new_value = agent.value(game.board)
            TD_error = reward + discount * new_value - value

            eligibility = update_eligibility(eligibility, discount * lmbda, grad_value)

            agent.update_params((1 - 2 * game.turn) * learning_rate * TD_error, eligibility)

            if game.winner != -1:
                finished = True
                outcomes[episode] = game.winner

        if episode % iprint == 0 and episode > 0:
            print(f'Trained for {episode} episodes, won {100 * (1 - np.sum(outcomes) / episode)}% so far! '
                  f'({100 * (1 - np.sum(outcomes[episode - iprint:episode]) / iprint)}% of the last set)')

    return outcomes

# Start simpler: train vs random, with the agent always being player 0
def train_vs_random(agent, game, episodes, learning_rate, epsilon, lmbda, discount=1, iprint=100):
    outcomes = np.zeros(shape=episodes)

    for episode in range(episodes):
        game.reset()
        finished = False
        eligibility = init_eligibility([agent.input_units, agent.hidden_units, 1])
        new_value = agent.value(game.board, game.turn)

        while not finished:
            if game.turn == 0:
                value = new_value
                grad_value = agent.value_gradient(game.board, game.turn)

                game.play_move(agent.policy(game, epsilon))
                while game.turn == 1 and game.winner == -1:
                    pi_random(game)

                reward = game.reward()
                # if reward != 0:
                #     print(f'agent sees reward {reward}!, board:')
                #     print(game.board)
                new_value = agent.value(game.board, game.turn)
                TD_error = reward + discount * new_value - value

                eligibility = update_eligibility(eligibility, discount * lmbda, grad_value)

                agent.update_params(learning_rate * TD_error, eligibility)
            else:
                print('err')
                pi_random(game)

            if game.winner != -1:
                finished = True
                outcomes[episode] = game.winner

        if episode % iprint == 0 and episode > 0:
            print(f'Trained for {episode} episodes, won {100 * (1 - np.sum(outcomes) / episode)}% so far! '
                  f'({100 * (1 - np.sum(outcomes[episode-iprint:episode]) / iprint)}% of the last set)')

    return outcomes


def train_vs_previous_self(agent, agent_previous, game, episodes, learning_rate, epsilon, lmbda, discount=1, iprint=100):
    outcomes = np.zeros(shape=episodes)

    for episode in range(episodes):
        game.reset()
        finished = False
        eligibility = init_eligibility([agent.input_units, agent.hidden_units, 1])
        new_value = agent.value(game.board, game.turn)

        agent_previous.set_params(agent.get_params())

        while not finished:
            if game.turn == 0:
                value = new_value
                grad_value = agent.value_gradient(game.board, game.turn)

                game.play_move(agent.policy(game, epsilon))
                while game.turn == 1 and game.winner == -1:
                    game.play_move(agent_previous.policy(game, epsilon))

                reward = game.reward()
                # if reward != 0:
                #     print(f'agent sees reward {reward}!, board:')
                #     print(game.board)
                new_value = agent.value(game.board, game.turn)
                TD_error = reward + discount * new_value - value

                eligibility = update_eligibility(eligibility, discount * lmbda, grad_value)

                agent.update_params(learning_rate * TD_error, eligibility)
            else:
                print('err')
                pi_random(game)

            if game.winner != -1:
                finished = True
                outcomes[episode] = game.winner

        if episode % iprint == 0 and episode > 0:
            agent_previous.set_params(agent.get_params())
            print(f'Trained for {episode} episodes, won {100 * (1 - np.sum(outcomes) / episode)}% so far! '
                  f'({100 * (1 - np.sum(outcomes[episode-iprint:episode]) / iprint)}% of the last set)')

    return outcomes
