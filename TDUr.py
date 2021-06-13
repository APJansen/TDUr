import jax.numpy as jnp
import numpy as np
from jax import grad, random, jit, vmap
from jax.nn import relu, sigmoid
import os
import pickle


class TDUr:
    """A value based agent for playing Ur.

    Attributes:
        input_units: The number of input units of the value network.
        hidden_units: The number of hidden units of the value network.
        params: The parameters of the value network.
        key: A jax random key.
    """
    def __init__(self, hidden_units=40, key=random.PRNGKey(42)):
        """Construct a TD-Ur agent.

        Args:
            hidden_units: The number of hidden units to use in the value network.
            key: Optional, jax random key, defaults to `random.PRNGKey(42)`.
        """
        self.input_units = 32
        self.hidden_units = hidden_units
        self.key = key
        self.params = self._init_network_params()

        # jitted functions
        self._value_grad = jit(grad(self._compute_value))
        self._compute_values = jit(vmap(self._compute_value, in_axes=(None, 0, 0)))

    def value(self, board, turn):
        """Return value as seen from player 0's perspective.

        Relies on jitted `_compute_value`.

        Args:
            board: A game board, a 2D numpy array of integers.
            turn: Integer, 0 or 1, indicating whose turn it is.

        Returns:
            Estimated win probability of player 0.
        """
        return self._compute_value(self.params, board, turn)

    def value_gradient(self, board, turn):
        """Return the gradient of the value function.

        Relies on jitted grad of `_compute_value`: `_value_grad`.

        Args:
            board: A game board, a 2D numpy array of integers.
            turn: Integer, 0 or 1, indicating whose turn it is.

        Returns:
            The gradient of the value function, a list of jax.numpy tensors of the same shape as the agent's parameters.
        """
        return self._value_grad(self.params, board, turn)

    def get_params(self):
        """Return agent parameters."""
        return self.params

    def set_params(self, parameter_values):
        """Set agent parameters. Must be compatible with `hidden_units` attribute."""
        self.params = parameter_values

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
        self.params = self._get_new_params(self.params, scalar, eligibility)

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

        chosen_move = moves[self._min_max_move(values, game.turn)]
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
            values = self._compute_values(self.params, boards, turns)
        else:  # plies == 2
            game.backup()

            values = np.zeros(shape=len(moves))
            for i, (board, turn, win) in enumerate(zip(boards, turns, wins)):
                if win != -1:
                    # if already won after 1 ply, no need to go further
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

    def _init_network_params(self):
        """Initialize the parameters of the neural network.

        Uses sigmoid activation for both layers.
        """
        keys = random.split(self.key, 2)
        n_in, n_hidden, n_out = self.input_units, self.hidden_units, 1
        layer_1 = self._random_layer_params(n_in=n_in, n_out=n_hidden, key=keys[0])
        layer_2 = self._random_layer_params(n_in=n_hidden, n_out=n_out, key=keys[1])
        return [layer_1, layer_2]

    @staticmethod
    @jit
    def _compute_value(params, board, turn):
        """Return the value of the given state computed with the given parameters.

        Imposes reflection symmetry (of board and players) by transforming the board to make it always player 0's turn.
        Independently from this, the value returned is always the value as seen from player 0's perspective.
        So if it is player 0's turn, compute the value and return it.
        But if it is player 1's turn, flip the board to make it player 0's turn, compute the value of that board,
        and return 1 - value.

        Args:
            params: The neural network parameters.
            board: The game board on which to compute a state.
            turn: Integer (0 or 1) indicating whose turn it is.

        Returns:
            The computed value.
        """
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

    @staticmethod
    @jit
    def _min_max_move(values, turn):
        """Return the index of the maximum (turn==0) or minimum (turn==1), depending on the turn.

        Args:
            values: A list of values.
            turn: Integer indicating whose turn it is (0 or 1).

        Returns:
            The index with maximal or minimal value.
        """
        return jnp.argmax((1 - 2 * turn) * values)

    @staticmethod
    @jit
    def _get_new_params(params, scalar, eligibility):
        """Return the updated parameters."""
        return [(w + scalar * z_w, b + scalar * z_b) for (w, b), (z_w, z_b) in zip(params, eligibility)]

    @staticmethod
    def _random_layer_params(n_in, n_out, key, scale=1e-4):
        """Initialize parameters for a single layer.

        Args:
            n_in: Number of input units.
            n_out: Number of output units.
            key: A jax random key.
            scale: Optional, defaults to 1e-4, sets scale of gaussian.

        Returns:
            Tuple (weights, biases)
        """
        w_key, b_key = random.split(key)
        weights = scale * random.normal(w_key, (n_out, n_in))
        biases = jnp.zeros(shape=(n_out,))

        return weights, biases
