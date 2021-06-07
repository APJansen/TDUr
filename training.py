import jax.numpy as jnp
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
def get_TD_error(new_value, value, reward, has_finished, discount):
    if has_finished:
        # when the game has been won, the reward itself is the total return, shouldn't bootstrap anymore.
        return reward - value
    else:
        return reward + discount * new_value - value


def compose_name(agent, learning_rate, epsilon, lmbda, lr_decay, search_plies=1):
    name = f'N{agent.hidden_units:d}'
    name += f'-alpha{learning_rate:.3f}'
    name += f'-lambda{lmbda:.2f}'
    name += f'-epsilon{epsilon:.5f}'
    name += f'-dalpha{lr_decay:.6f}'
    if search_plies > 1:
        name += f'-plies{search_plies}'

    return name


def train(agent, game, episodes, learning_rate, epsilon, lmbda, discount=1, search_plies=1, iprint=100, save=False,
          learning_rate_decay=1, episode_start=0, custom_name=False):
    red_wins = 0
    start = time.time()
    total_moves = 0

    if custom_name:
        name = custom_name
    else:
        name = compose_name(agent, learning_rate, epsilon, lmbda, learning_rate_decay, search_plies)

    for episode in range(episode_start, episode_start + episodes + 1):
        game.reset()
        finished = False
        eligibility = init_eligibility([agent.input_units, agent.hidden_units, 1])

        new_value = agent.value(game.board, game.turn)

        while not finished:
            value = new_value
            grad_value = agent.value_gradient(game.board, game.turn)

            move = agent.policy(game, epsilon=epsilon, plies=search_plies)
            game.play_move(move)

            # no matter who's playing, we always want to estimate the probability of winning of player 0
            # the only point where turn number should enter is in choosing the next move, which is max/min on that
            reward = game.reward()
            new_value = agent.value(game.board, game.turn)

            TD_error = get_TD_error(new_value, value, reward, game.has_finished(), discount)

            eligibility = update_eligibility(eligibility, discount * lmbda, grad_value)
            agent.update_params(learning_rate * TD_error, eligibility)

            if game.has_finished():
                finished = True
                red_wins += 1 - game.winner
                total_moves += game.move_count

        learning_rate *= learning_rate_decay

        if episode % iprint == 0 and episode > episode_start:
            end = time.time()
            print(f'episodes: {episode}; red wins: {red_wins:d}; time: '
                  f'{(end - start) / total_moves * 1000:.2f} s/(1k moves) ({end-start:.1f}s tot); '
                  f'{total_moves/iprint:.0f} moves per game.')
            start = end
            total_moves = 0
            red_wins = 0
            if save:
                agent.save_params(name + f'-episodes{episode:d}')
