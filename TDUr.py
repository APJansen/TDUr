import jax.numpy as jnp
import numpy as np
from jax import grad, random, jit, vmap
from jax.nn import relu, sigmoid
from functools import partial


# def relu(x):
#     return jnp.maximum(0, x)
#
#
# def sigma(x):
#     return 1. / (1. + jnp.exp(-x))

# shouldn't value(flip(board)) = 1 - value(board)? how to impose?
# can use this symmetry: always compute value of board where it's player 0's turn.
# so if it's player 0's turn, compute value (can get rid of turn indices) and just return it
# but if it's player 1's turn, flip the board, compute the value of that board, and return 1 - value.
# this way what's returned is still always the value from player 0's perspective.
# this should enforce a 50/50 win rate in self play (up to starting player's advantage)
@jit
def compute_value(params, board):
    # make it so that it always analyzes boards where it's player 0's turn
    zero, one = board[:, -1]
    board = np.flip(board[board[:, -1]], axis=0)

    activations = jnp.reshape(board, -1)
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logit = jnp.reshape(jnp.dot(final_w, activations) + final_b, (()))
    value = sigmoid(logit)

    return zero * value + one * (1 - value)


# value_grad_turn_zero = jit(grad(compute_value_turn_zero))
# value_grad_turn_one = jit(grad(compute_value_turn_one))
#
#
# @jit
# def compute_value_turn_one(params, board):
#     return 1 - compute_value(params, jnp.flip(board, axis=0))
#
#
# def compute_value(params, board, turn):
#     if turn == 0:
#         return compute_value_turn_zero(params, board)
#     else:
#         return compute_value_turn_one(params, board)
#
#
# def value_grad(params, board, turn):
#     if turn == 0:
#         return value_grad_turn_zero(params, board)
#     else:
#         return value_grad_turn_one(params, board)


value_grad = jit(grad(compute_value))
compute_values = jit(vmap(compute_value, in_axes=(None, 0)))


@jit
def min_max_move(moves, values, turn):
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
        self.input_units = 34
        self.hidden_units = hidden_units
        self.key = key
        self.params = init_network_params([self.input_units, self.hidden_units, 1], key)

    def value(self, board):
        # computes value as seen from player 0
        return compute_value(self.params, board)
        # if turn == 0:
        #     return compute_value(self.params, board)
        # else:
        #     return compute_value_reversed(self.params, board)

    def value_gradient(self, board):
        return value_grad(self.params, board)

    def get_params(self):
        return self.params

    def set_params(self, parameter_values):
        self.params = parameter_values

    def init_params(self):
        self.params = init_network_params([self.input_units, self.hidden_units, 1], self.key)

    def save_params(self, name, directory=''):
        return # TODO: implement save params

    def update_params(self, scalar, eligibility):
        self.params = get_new_params(self.params, scalar, eligibility)

    def policy(self, game, epsilon=0, checks=False):
        moves = game.legal_moves()
        if moves == ['pass']:
            return 'pass'

        if np.random.uniform() < epsilon:  # TODO: replace with jax's random
            return np.random.choice(moves)

        # boards = jnp.array([game.simulate_move(move, checks)[1] for move in moves]) # TODO: parallelize?
        boards = game.simulate_moves(moves)
        values = compute_values(self.params, boards)
        # if game.turn == 0:
        #     values = compute_values(self.params, boards)
        # else:
        #     # values = 1 - compute_values(self.params, jnp.array([jnp.flip(board, axis=0) for board in boards]))
        #     values = 1 - compute_values(self.params, jnp.flip(boards, axis=1))

        #chosen_move = moves[jnp.argmax(jnp.array(values)) if game.turn == 0 else jnp.argmin(jnp.array(values))]
        chosen_move = moves[min_max_move(moves, values, game.turn)]
        return chosen_move
