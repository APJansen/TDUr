import jax.numpy as jnp
import numpy as np
from jax import grad, random


def relu(x):
    return jnp.maximum(0, x)


def sigma(x):
    return 1. / (1. + jnp.exp(-x))


def compute_value(params, board, turn):
    activations = jnp.reshape(board, -1)
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.reshape(jnp.dot(final_w, activations) + final_b, (()))

    return sigma(logits) if turn == 0 else (1 - sigma(logits))


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


class TDUr:

    def __init__(self, hidden_units=40, key=random.PRNGKey(42)):
        self.input_units = 34
        self.hidden_units = hidden_units
        self.params = init_network_params([self.input_units, self.hidden_units, 1], key)

    def value(self, board, turn):
        return compute_value(self.params, board, turn)

    def value_gradient(self, board, turn):
        return grad(compute_value)(self.params, board, turn)

    def get_params(self):
        return self.params

    def set_params(self, parameter_values):
        self.params = parameter_values

    def init_params(self):
        self.params = init_network_params([self.input_units, self.hidden_units, 1])

    def update_params(self, scalar, eligibility):
        self.params = [(w + scalar * z_w, b + scalar * z_b) for (w, b), (z_w, z_b) in zip(self.params, eligibility)]

    def policy(self, game, epsilon=0, checks=False):
        moves = game.legal_moves()
        if moves == ['pass']:
            return 'pass'

        if np.random.uniform() < epsilon:  # TODO: replace with jax's random
            return np.random.choice(moves)

        values = []
        rewards = []
        for move in moves:
            # TODO: possible optimization here: get features of all moves and multiply as matrix
            reward, board = game.simulate_move(move, checks)
            turn = 0 if board[0, game.turn_index] == 1 else 0  # TODO: make nicer
            values.append(self.value(board, turn))
            rewards.append(reward)
            # what do I do with rewards? Include here or not?

        chosen_move = moves[jnp.argmax(jnp.array(values))]
        return chosen_move

    def TD_error(self, game, move):
        v_current = self.value(game.board)
        game.play_move(self.policy(game))
        v_next = self.value(game.board)
        reward = game.reward()

        return reward + v_next - v_current
