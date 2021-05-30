import numpy as np


def compete_policies(game, pi_1, pi_2, episodes=1_000):
    scores = [0, 0]
    for episode in range(episodes):
        game.reset()
        player_1 = episode % 2
        player_2 = (player_1 + 1) % 2

        while game.winner == -1:
            if game.turn == player_1:
                move = pi_1.policy(game, epsilon=0)
            else:
                move = pi_2.policy(game, epsilon=0)
            game.play_move(move, checks=False)

        if game.winner == player_1:
            scores[0] += 1
        else:
            scores[1] += 1

    return scores
#
#
# def payoff_matrix(game, policies, episodes=1_000):
#     n_policies = len(policies)
#     payoffs = np.zeros(shape=(n_policies, n_policies))
#     for i, pi_1 in enumerate(policies):
#         for j, pi_2 in enumerate(policies):
#             if j > i:
#                 scores = compete_policies(game, pi_1, pi_2, episodes=episodes)[0]
#                 payoffs[i, j] = scores[0] / episodes
#     return payoffs
