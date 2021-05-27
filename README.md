# The Royal Game of Ur

The Royal Game of Ur is an ancient Sumerian game, Ur being the Sumerian city in Mesapotamia, current Iraq, where the first game boards were found. 
It was played from around 2.500 BC until late antiquity, when it was replaced or evolved into Backgammon.


[![Alt text](https://img.youtube.com/vi/WZskjLq040I/0.jpg)](https://www.youtube.com/watch?v=WZskjLq040I)

For an introduction to both the history and the rules of the game, I highly recommend this very entertaining YouTube video.
In this video, dr. Irving Finkel, curator of the British Museum and the discoverer and translator of a clay tablet from 177 BC describing the rules of the game,
explains the rules and history of the game, and plays a game against popular youtuber Tom Scott.

## The rules
- There are two players, which each have 7 stones
- The goal is to move all of your stones along the route indicated and to the finish, before your opponent does so.
- A player throws 4 tetrahedral dice with 2 corners each marked, or equivalently a coin. The number of marked corners that point up, or the number of heads, 
is the number of squares a stone may move.
- The choice to make is which stone to move forward the rolled number of squares.
- You may not land on your own stones.
- You may land on your opponent's stone, which will capture it and return it to the start.
- If you land on a rosette, you get another turn and the stone on the rosette cannot be captured.
- To move a stone off the board, it must move exactly one square off the board, not more.
- If no move is possible, the turn goes back to the other player.

#TODO: add picture of board, mine and the original

# TD Ur

We present an AI that plays Ur at human level.
Ur's successor Backgammon was the first game in which human experts were surpassed by a learned AI, called TD-Gammon by Tesauro ---cite---.
Since Ur is essentially a simplified Backgammon, it seems appropriate and sufficient to use the techniques of TD-Gammon for Ur, resulting in the TD-Ur presented here. The essential techniques are using a neural network to parametrize a value function, which is learned using TD(lambda) and a 2-3 ply lookahead.

If this means nothing to you, what follows is a quick explanation.
For more detail I recommend the book Reinforcement Learning: An Introduction, by Sutton and Barto, and these video lectures by David Silver. ---cite 2---


## Reinforcement Learning

Reinforcement learning is a branch of machine learning that applies to situations in which an _agent_ interacts with and receives _rewards_ from an _environment_.
In the present case, the environment is the game board and its rules, the rewards only come at the end and is simply the outcome of the game, and the agent is the AI that we want to train.

This setting is formalized into a mathematical structure called a Markov Decision Process (MDP). An MDP consists of _states_, _actions_, _rewards_ and _transition probabilities_. 
In the present case, a state consists of the board configuration, the last die roll, and whose turn it is.
In each state, there are one or more available actions to choose from, here these are the legal moves.
Upon choosing a specific action in a given state, the game moves to a new state, according to the transition probabilities for this state, action pair.
In the present case, this is a three step process:
1. deterministic consequences of the move, i.e. when the roll is 2 and the chosen stone to move is at position 4, the stone is removed from position 4 and put back at position 6, the opponent's stone is removed and put back at the start, and the turn is transferred to the opponent. The intermediate "state" after the deterministic part of the transition but before the full transition is called an _after state_.
2. The opponent rolls the dice.
3. From the point of view of the opponent, this is now a state again. But viewing it from the point of view of our agent, we don't know and can't influence what the opponent will do, so this is still part of the environment dynamics, up until the point that it's our turn and we've thrown the dice.
Of course if we land on a rosette, it is our (i.e. the agent's) turn again, and there is no step 3.
On each transition to a new state, the agent receives a reward. But in this game, there are no intermediate rewards, only a final win or loss. So the reward is 1 on the final transition to a win, and 0 otherwise.

The goal of a reinforcement learning agent is to maximize the sum of its rewards, which is also called the return, but in this case simply is the chance of winning.

## Value Function

To maximize the return, or win probability, rather than engineering by hand, using our own knowledge of the game, complicated rules on what actions to take, in the reinforcement learning paradigm, we let the agent learn how to do it for itself.
This often involves having the agent learn a _value function_, an estimate of the expected return in the current state, or the win probability in our case.
This value function can be allowed to depend only on the current state, or also on the chosen action, when it's called an action value.
For games such as this where the transition to a new state can be decomposed into first a deterministic part and then a probabilistic part, it is convenient to choose a middle ground, namely the afterstate. So the input to our value function will be the board configuration and whose turn it is, but it does not include the dice roll, and it's not necessarily the agent's turn.

Given such an afterstate s_a obtained from the deterministic part of the transition after taking action a in state s, we want to learn to estimate a number  0 <= v(s_a) <= 1 representing the probability of winning the game.
We will use a neural network to represent this function. A simple fully connected network with one hidden layer will do. Using a sigmoid activation function on the final layer guarantees that the value will be between 0 and 1.
Initializing the weights as random small numbers, our initial estimate will have values near 0.5, with no structure or meaning to it.


---comment info used---

## Policy

The value function as presented above is not yet completely well defined, because the win probability in a given (after)state depends on the moves chosen subsequently. More formally, it depends on the _policy_ used, where a policy is a map from a state to an action, i.e. a way of choosing a move.

What we are after is the optimal policy, choosing the action that maximizes the win probability against an opponent also playing optimally.
Given the true value function assuming optimal play, we can easily obtain the optimal policy, simply by choosing the action that maximizes the afterstate value.

Of course we don't have this either, but we can work our way towards it. We start with some random value function as defined in the previous section. From this we derive a policy as above, this is called the _greedy policy_, because it maximizes the value in the next state, and does not take into consideration that to obtain the longer term maximum we might have to accept short term losses.
Then we update our estimate of the value function to more accurately reflect the win probability of this policy (see below how).
Then we use this updated value function to define a new policy in the same way.
Iterating this is called _policy iteration_, and will allow us to converge to the optimal policy and value function.


## TD Learning

Now how do we update our estimate of the value function?
We do this using _Temporal Difference (TD) Learning_, whose basic premise is that future states have more information than present states.
The logic is that a. they are closer to the final win/loss state, and b. they have experienced more environment dynamics.
So what we can do is compute the value of the current state, V(S_t), and then take a step (using the policy whose value we're estimating), and compute the value V(S_t+1) there.
Assuming that the latter is more accurate, we want to update our estimate to move V(S_t) closer to V(S_t+1). In other words we want to minimize (V(S_t+1) - V(S_t))^2, but only through V(S_t). Now V is a neural network, parametrized by weights W, so we can do this using gradient descent, resulting in the weight update
W_t+1 = W_t + alpha (V_(S_t+1) - V(S_t)) \nabla_W V(S_t)
This is called _semi-gradient descent_ because we keep the future estimate fixed. This is very important, if we do not do this we are not making use of the fact that future estimates are more accurate than present estimates.

## TD(lambda)

What is described above amounts to 1-step TD, where we look one step into the future, and only update the weights according to the current state's value.
This is a usually suboptimal solution to the _credit assignment problem_, which is the problem of identifying which of the choices in the past were most responsible for the current situation.

A more elegant and usually more efficient solution to this problem is called TD(lambda).
This uses an _eligibility trace_, which is an exponentially decaying average of the gradient of the value function,
z_t+1 = lambda z_t + nabla_w v,
where lambda in (0,1) specifies the decay rate, and thus the time scale over which previous moves are considered to have an impact on the current value.
When lambda=0 we recover the previous 1-step TD, but often, and indeed in our case too, lambda=0.9 is found to be more efficient.



## Lookahead?

## Implementation Details

- jax
- symmetry
- final TD error?
- 


# References

