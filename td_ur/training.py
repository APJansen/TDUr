import jax.numpy as jnp
import time
from jax import jit
from functools import partial


def train(agent, game, episodes, learning_rate, epsilon, lmbda, discount=1, search_plies=1, iprint=100, save=False,
          learning_rate_decay=1, episode_start=0, custom_name=False):
    """Train an agent through self play of a game using TD(lambda).

    Update agent's params attribute after every move.

    Args:
        agent: The agent instance to train.
        game: The game instance to play.
        episodes: The number of games to train for.
        learning_rate: Learning rate used in parameter updates.
        epsilon: Exploration parameter.
        lmbda: The lambda parameter in TD(lambda).
        discount: Discounting parameter for the returns.
        search_plies: Optional, 1 or 2, how many ply the agent looks ahead, defaults to 1.
        iprint: Optional, after how many episodes information is printed and parameters are saved, defaults to 100.
        save: Optional, whether to save the parameters to a file, defaults to False.
        learning_rate_decay: Optional, coefficient of exponential learning rate decay, defaults to 1 (no decay).
        episode_start: Optional, if starting from an already trained agent, will reflect this in save name.
        custom_name: Optional string, a custom name of the file to which the parameters are saved.
    """
    red_wins = 0
    start = time.time()
    total_moves = 0

    if custom_name:
        name = custom_name
    else:
        name = compose_name(agent.hidden_units, learning_rate, epsilon, lmbda, learning_rate_decay, search_plies)

    for episode in range(episode_start, episode_start + episodes + 1):
        game.reset()
        eligibility = init_eligibility(agent.input_units, agent.hidden_units)

        new_value = agent.value(game.board, game.turn)

        # play until the game is finished, and then do one more update on the value of the final state
        finished = False
        while not finished:
            finished = game.has_finished()

            value = new_value
            grad_value = agent.value_gradient(game.board, game.turn)

            if not finished:
                move = agent.policy(game, epsilon=epsilon, plies=search_plies)
                game.play_move(move)

                new_value = agent.value(game.board, game.turn)

                TD_error = discount * new_value - value
            else:

                # once the game is finished it gives a reward, 1 if player 0 won, else 0
                # still need to update the value of the final state
                reward = game.reward()
                TD_error = reward - value

            eligibility = update_eligibility(eligibility, discount * lmbda, grad_value)
            agent.update_params(learning_rate * TD_error, eligibility)

        red_wins += 1 - game.winner
        total_moves += game.move_count

        learning_rate *= learning_rate_decay

        if episode % iprint == 0 and episode > episode_start:
            end = time.time()
            print(f'episodes: {episode}; red wins: {red_wins:d}; time: '
                  f'{(end - start) / total_moves * 1000:.2f} s/(1k moves) ({end - start:.1f}s tot); '
                  f'{total_moves / iprint:.0f} moves per game.')
            start = end
            total_moves = 0
            red_wins = 0
            if save:
                agent.save_params(name + f'-episodes{episode:d}')


@partial(jit, static_argnums=3)
def get_TD_error(new_value, value, reward, has_finished, discount):
    """Return TD error.

    Args:
        new_value: Value at next state.
        value: Value at current state.
        reward: Reward in transitioning to next state.
        has_finished: If the game has finished.
        discount: Discount factor.

    Returns:
        The TD error.
    """
    if has_finished:
        # when the game has been won, the reward itself is the total return, shouldn't bootstrap anymore.
        return reward - value
    else:
        return reward + discount * new_value - value


@partial(jit, static_argnums=(0, 1))
def init_eligibility(input_units, hidden_units):
    """Initialize the eligibility trace to zero."""
    sizes = [input_units, hidden_units, 1]
    return [(jnp.zeros(shape=(n, m)), jnp.zeros(shape=(n,))) for m, n in zip(sizes[:-1], sizes[1:])]


@jit
def update_eligibility(eligibility, scalar, grad_value):
    """Update the parameters according to TD(lambda).

    Args:
        eligibility: The eligibility trace.
        scalar: The product of the learning rate and the TD-error.
        grad_value: The value gradient.

    Returns:
        parameters, list of jax.numpy tensors
    """
    return [(scalar * z_w + dv_dw, scalar * z_b + dv_db) for (z_w, z_b), (dv_dw, dv_db) in zip(eligibility, grad_value)]


def compose_name(hidden_units, learning_rate, epsilon, lmbda, lr_decay, search_plies=1):
    """Return name for parameter save file based on the hyperparameters."""
    name = f'N{hidden_units:d}'
    name += f'-alpha{learning_rate:.3f}'
    name += f'-lambda{lmbda:.2f}'
    name += f'-epsilon{epsilon:.5f}'
    name += f'-dalpha{lr_decay:.6f}'
    if search_plies > 1:
        name += f'-plies{search_plies}'

    return name
