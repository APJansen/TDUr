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


# def random_layer_params(m, n, key, scale=1e-2):
#     w_key, b_key = random.split(key)
#     return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
#
#
# def init_network_params(sizes, key):
#     keys = random.split(key, len(sizes))
#     return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]


def random_layer_params(n_in, n_out, key, activation):
    w_key, b_key = random.split(key)
    if activation == 'sigmoid':
        scale = jnp.sqrt(1 / n_in)
        weights = scale * random.normal(w_key, (n_out, n_in))
        biases = jnp.zeros(shape=(n_out,))
    else:  # activation == 'relu'
        scale = jnp.sqrt(2 / n_in)
        weights = scale * random.normal(w_key, (n_out, n_in))
        biases = jnp.zeros(shape=(n_out,)) + 0.1

    return weights, biases


def init_network_params(n_in, n_hidden, n_out, key):
    keys = random.split(key, 2)
    layer_1 = random_layer_params(n_in=n_in, n_out=n_hidden, key=keys[0], activation='relu')
    layer_2 = random_layer_params(n_in=n_hidden, n_out=n_out, key=keys[1], activation='sigmoid')
    return [layer_1, layer_2]


class TDUr:

    def __init__(self, hidden_units=40, key=random.PRNGKey(42)):
        self.input_units = 32
        self.hidden_units = hidden_units
        self.key = key
        self.params = init_network_params(self.input_units, self.hidden_units, 1, key)

    def value(self, board, turn):
        """Return value as seen from player 0's perspective.

        Relies on jitted `compute_value`.

        Args:
            board: A game board, a 2D numpy array of integers.
            turn: Integer, 0 or 1, indicating whose turn it is.

        Returns:
            Estimated win probability of player 0.
        """
        return compute_value(self.params, board, turn)

    def value_gradient(self, board, turn):
        """Return the gradient of the value function.

        Relies on jitted grad of `compute_value`: `value_grad`.

        Args:
            board: A game board, a 2D numpy array of integers.
            turn: Integer, 0 or 1, indicating whose turn it is.

        Returns:
            The gradient of the value function, a list of jax.numpy tensors of the same shape as the agent's parameters.
        """
        return value_grad(self.params, board, turn)

    def get_params(self):
        """Return agent parameters."""
        return self.params

    def set_params(self, parameter_values):
        """Set agent parameters. Must be compatible with `hidden_units` attribute."""
        self.params = parameter_values

    def init_params(self):
        """Initialize parameters."""
        self.params = init_network_params(self.input_units, self.hidden_units, 1, self.key)

    def save_params(self, name, directory='parameters'):
        """Save the current agent parameters to a file, pickled.

        Keyword argument: directory='parameters', the subdirectory.
        Args:
            name: Name of the file to create.
            directory: Optional subdirectory to safe to, default `parameters`.
        """
        pickle.dump(self.params, open(os.path.join(directory, name + '.pkl'), "wb"))

    def update_params(self, scalar, eligibility):
        """Update agent parameters.

        Relies on jitted `get_new_params`.

        Args:
            scalar: the product of the learning rate and TD-error.
            eligibility: The eligibility trace.
        """
        self.params = get_new_params(self.params, scalar, eligibility)

    def policy(self, game, plies=1, epsilon=0):
        """Return the greedy (or epsilon-greedy) move based on the input game's state and the agent's value function.

        Args:
            game: The game instance to play on.
            plies: Optional, 1 or 2, how many plies to search, defaults to 1.
            epsilon: Optional exploration parameter, defaults to 0.

        Returns:
            The (epsilon-)greedy move, integer.
        """
        moves = game.legal_moves()
        if len(moves) == 1:
            return moves[0]

        if np.random.uniform() < epsilon:
            return np.random.choice(moves)

        values = self.move_values(game, moves=moves, plies=plies)

        chosen_move = moves[min_max_move(values, game.turn)]
        return chosen_move

    def move_values(self, game, moves=False, plies=1):
        """Return afterstate values for moves given.

        Args:
            game: The game instance to play on.
            moves: List of legal moves, or False in which case list is generated (for external use).
            plies: Optional, 1 or 2, how many plies to search, defaults to 1.

        Returns:
            List of afterstate values for each of the given moves.
        """
        if moves is False:
            moves = game.legal_moves()

        boards, turns, wins = game.simulate_moves(moves)

        if plies == 1:
            values = compute_values(self.params, boards, turns)
        else:  # plies == 2
            game.backup()

            values = np.zeros(shape=len(moves))
            for i, (board, turn, win) in enumerate(zip(boards, turns, wins)):
                if win != -1:
                    values[i] = self.value(board, turn)
                else:
                    roll_values = np.zeros(shape=len(game.rolls))
                    for roll in game.rolls:
                        game.set_state((board, turn, roll, win, 0))
                        game.play_move(self.policy(game))
                        roll_values[roll] = self.value(game.board, game.turn)

                    values[i] = np.sum(game.probabilities * roll_values)

            game.restore_backup()

        return values
