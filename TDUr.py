import jax as jx
import numpy as jnp
import haiku as hk

# TODO: take into account which player the agent is

class TDUr:

    def __init__(self, hidden_units=80, epsilon=0.01):
        self.input_units = 34
        self.hidden_units = hidden_units
        self.epsilon = epsilon
        self.W1 = jnp.random.randn(hidden_units, 34)
        self.W2 = jnp.random.randn(1, hidden_units)
        self.rescale = jnp.ones(shape=(2, 17))
        self.rescale[:, 0] = self.rescale[:, 15] = 7 * jnp.ones(shape=2)

    def value(self, board):
        rescaled = board / self.rescale
        input = rescaled.flatten()
        A1 = jnp.dot(self.W1, input)
        Z1 = self.sigma(A1)

        return self.sigma(jnp.dot(self.W2, Z1))

    def get_weights(self):
        return self.W1, self.W2

    def set_weights(self, W1, W2):
        self.W1 = W1
        self.W2 = W2

    def init_weights(self):
        self.W1 = jnp.random.randn(self.hidden_units, self.input_units)
        self.W2 = jnp.random.randn(1, self.hidden_units)

    def policy(self, game):
        moves = game.legal_moves()
        if moves == ['pass']:
            return 'pass'

        if jnp.random.uniform() < self.epsilon:
            return jnp.random.choice(moves)

        values = []
        rewards = []
        for move in moves:
            # possible optimization here: get features of all moves and multiply as matrix
            reward, board = game.simulate_move(move)
            values.append(self.value(board))
            rewards.append(reward)

        return moves[jnp.argmax(values)]

    def TD_error(self, game, move):
        v_current = self.value(game.board)
        game.play_move(self.policy(game))
        v_next = self.value(game.board)
        reward = game.reward()

        return reward + v_next - v_current

    @staticmethod
    def sigma(z):
        return 1 / (1 + jnp.exp(-z))
