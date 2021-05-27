import jax.numpy as jnp
import numpy as np
from agents_engineered import pi_random
import time
from jax import jit
from functools import partial


def zero_layer_params(m, n):
    return jnp.ones(shape=(n, m)), jnp.ones(shape=(n,))


def init_eligibility(sizes):
    return [zero_layer_params(m, n) for m, n in zip(sizes[:-1], sizes[1:])]


@jit
def update_eligibility(eligibility, decay, grad_value):
    return [(decay * z_w + dv_dw, decay * z_b + dv_db) for (z_w, z_b), (dv_dw, dv_db) in zip(eligibility, grad_value)]


@partial(jit, static_argnums=3)
def get_TD_error(new_value, value, reward, winner, discount):
    if winner == -1:
        return reward + discount * new_value - value
    else:
        # when the game has been won, the reward itself is the total return, shouldn't bootstrap anymore.
        return reward - value


def train(agent, game, episodes, learning_rate, epsilon, lmbda, discount=1, checks=False, iprint=100, save=False,
          learning_rate_decay=1):
    outcomes = np.zeros(shape=episodes + 1, dtype='int8')
    start = time.time()
    total_moves = 0

    for episode in range(episodes + 1):
        game.reset()
        finished = False
        eligibility = init_eligibility([agent.input_units, agent.hidden_units, 1])

        new_value = agent.value(game.board, game.turn)

        while not finished:
            value = new_value
            grad_value = agent.value_gradient(game.board, game.turn)

            move = agent.policy(game, epsilon)
            game.play_move(move, checks)

            # no matter who's playing, we always want to estimate the probability of winning of player 0
            # the only point where turn number should enter is in choosing the next move, which is max/min on that
            reward = game.reward()
            new_value = agent.value(game.board, game.turn)

            TD_error = get_TD_error(new_value, value, reward, game.winner, discount)

            eligibility = update_eligibility(eligibility, discount * lmbda, grad_value)
            agent.update_params(learning_rate * TD_error, eligibility)

            if game.winner != -1:
                finished = True
                outcomes[episode] = game.winner
                total_moves += game.move_count

        learning_rate *= learning_rate_decay

        if episode % iprint == 0 and episode > 0:
            end = time.time()
            wins_1 = np.sum(outcomes[episode - iprint:episode])
            print(f'episodes: {episode}; wins: {iprint - wins_1:d}-{wins_1:d}; time: '
                  f'{(end - start) / total_moves * 1000:.2f} s/(1k moves) ({end-start:.1f}s tot); '
                  f'{total_moves/iprint:.0f} moves per game.')
            start = end
            total_moves = 0
            if save:
                agent.save_params(save + f'-episodes{episode:d}')

    return outcomes


# The 2 functions below were just to test everything before adding complications of self play.
# They haven't been updated and probably won't work, need to either get rid of them or update them.
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
