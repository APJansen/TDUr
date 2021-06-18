def compete_policies(game, agent_1, agent_2, episodes=1_000, search_plies_1=1, search_plies_2=1):
    """Compete two agents against each other.

    Alternates the agent's colors after each game and agents use epsilon=0 (no exploration).

    Args:
        game: A game instance, needs `reset()` and `play_move(move)` methods and `turn` attribute.
        agent_1: The first agent, needs a `policy(game, epsilon, search_plies)` method.
        agent_2: The second agent, needs a `policy(game, epsilon, search_plies)` method.
        episodes: The number of games to play.
        search_plies_1: How many ply agent_1 searches (1 or 2).
        search_plies_2:How many ply agent_2 searches (1 or 2).

    Returns:
        Two element list of win fractions.
    """
    scores = [0, 0]
    for episode in range(episodes):
        game.reset()
        player_1 = episode % 2

        while not game.has_finished():
            if game.turn == player_1:
                move = agent_1.policy(game, epsilon=0, plies=search_plies_1)
            else:
                move = agent_2.policy(game, epsilon=0, plies=search_plies_2)
            game.play_move(move)

        if game.winner == player_1:
            scores[0] += 1
        else:
            scores[1] += 1

    return [score / episodes for score in scores]
