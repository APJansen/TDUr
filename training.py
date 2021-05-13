def train(agent, game, episodes, learning_rate, epsilon, lmbda, discount=1):
    eligibility = np.zeros(shape=agent.weights)

    for episode in range(episodes):
        game.reset()
        new_value = agent.value(game)
        while game.winner == -1: # need to go one step beyond?
            value = new_value
            grad_value = jax.grad(agent.value(game))

            game.play_move(agent.policy(game, epsilon))

            reward = game.reward()
            new_value = agent.value(game)
            TD_error = reward + discount * new_value - value

            eligibility = discount * lmbda * eligibility + grad_value
            # need reshaping
            agent.weights += learning_rate * TD_error * eligibility
