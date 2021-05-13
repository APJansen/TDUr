import jax as jx
import numpy as jnp


class TDUr:

    def __init__(self, hidden_units=80):
        self.hidden_units = hidden_units
        self.W1 = jnp.random.randn(hidden_units, 34)
        self.W2 = jnp.random.randn(1, hidden_units)
        self.rescale = jnp.ones(shape=(2, 17))
        self.rescale[:, 0] = self.rescale[:, 15] = 7 * jnp.ones(shape=2)

    def value(self, game):
        board = game.board
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
        self.W1 = jnp.random.randn(hidden_units, 34)
        self.W2 = jnp.random.randn(1, hidden_units)

    @staticmethod
    def sigma(z):
        return 1 / (1 + jnp.exp(-z))
