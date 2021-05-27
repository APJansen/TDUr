import jax.numpy as jnp
import numpy as np
from jax import grad, random, jit, vmap
from jax.nn import relu, sigmoid
import os
import pickle


# shouldn't value(flip(board)) = 1 - value(board)? how to impose?
# can use this symmetry: always compute value of board where it's player 0's turn.
# so if it's player 0's turn, compute value (can get rid of turn indices) and just return it
# but if it's player 1's turn, flip the board, compute the value of that board, and return 1 - value.
# this way what's returned is still always the value from player 0's perspective.
# this should enforce a 50/50 win rate in self play (up to starting player's advantage)
@jit
def compute_value(params, board, turn):
    # make it so that it always analyzes boards where it's player 0's turn
    other = (turn + 1) % 2
    board = jnp.array([board[turn], board[other]])

    activations = jnp.reshape(board, -1)
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logit = jnp.reshape(jnp.dot(final_w, activations) + final_b, (()))
    value = sigmoid(logit)

    return other * value + turn * (1 - value)


value_grad = jit(grad(compute_value))
compute_values = jit(vmap(compute_value, in_axes=(None, 0, 0)))


@jit
def min_max_move(values, turn):
    return jnp.argmax((1 - 2 * turn) * values)


@jit
def get_new_params(params, scalar, eligibility):
    return [(w + scalar * z_w, b + scalar * z_b) for (w, b), (z_w, z_b) in zip(params, eligibility)]


def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


class TDUr:

    def __init__(self, hidden_units=40, key=random.PRNGKey(42)):
        self.input_units = 32
        self.hidden_units = hidden_units
        self.key = key
        self.params = init_network_params([self.input_units, self.hidden_units, 1], key)

    def value(self, board, turn):
        """
        Computes value as seen from player 0.
        Input: board, turn
        """
        return compute_value(self.params, board, turn)

    def value_gradient(self, board, turn):
        return value_grad(self.params, board, turn)

    def get_params(self):
        return self.params

    def set_params(self, parameter_values):
        self.params = parameter_values

    def init_params(self):
        self.params = init_network_params([self.input_units, self.hidden_units, 1], self.key)

    def save_params(self, name, directory='parameters'):
        pickle.dump(self.params, open(os.path.join(directory, name + '.pkl'), "wb"))

    def update_params(self, scalar, eligibility):
        self.params = get_new_params(self.params, scalar, eligibility)

    def policy(self, game, epsilon=0):
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]

        if np.random.uniform() < epsilon:
            return np.random.choice(moves)

        boards, turns = game.simulate_moves(moves)
        values = compute_values(self.params, boards, turns)

        chosen_move = moves[min_max_move(values, game.turn)]
        return chosen_move
