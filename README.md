##### Table of Contents  
[The Royal Game of Ur](#game)  
- [Video Introduction by Irving Finkel](#vid)
- [Rules](#rules)
- [Play Game!](#play)
- [Strategy](#strat)

[TD-Ur](#tdur)
- [Reinforcement Learning](#rl)
- [Value Function](#value)
- [Policy](#plcy)
- [TD Learning](#td)
- [TD(lambda)](#eligibility)
- [Search](#srch)
- [Self-Play](#selfplay)
- [Implementation Details](#details)

<a name="game"/>

# The Royal Game of Ur

The Royal Game of Ur is an ancient Sumerian game, Ur being the Sumerian city in Mesapotamia, current Iraq, where the first game boards were found. 
It was played from around 2.500 BC until late antiquity, when it was replaced or evolved into Backgammon.

<a name="vid"/>

## Video Introduction by Irving Finkel

For an introduction to both the history and the rules of the game, I highly recommend this very entertaining YouTube video.
In this video, dr. Irving Finkel, curator of the British Museum and the translator of a clay tablet from 177 BC describing the rules of the game,
explains the rules and history of the game, and plays a game against popular youtuber Tom Scott.

[![Alt text](https://img.youtube.com/vi/WZskjLq040I/0.jpg)](https://www.youtube.com/watch?v=WZskjLq040I)


<a name="rules"/>

## The rules

![plot](./images/UrBoards.png)

The game is played by 2 players, on the board shown above, to the right an ancient Sumerian board and to the left my Mondriaan interpretation.
The yellow squares are not part of the board, but indicate the start (left) and finish (right) squares.
Each player starts with 7 stones, as indicated on the start square.
The crosses correspond to the "rosettes" on the original board, and have a special function as explained below.

The goal of the game is to be the first to bring all 7 stones from the start to the finish. Stones move along the route indicated in the image below.
The middle row is where the routes overlap and players can capture eachother's stones.

![plot](./images/UrBoardRoutes.png)

Players throw 4 tetrahedral dice with 2 corners each marked. The number of marked corners pointing up is the number of squares a single stone can be moved.
This is nothing but a fancy way of tossing 4 coins and counting the number of heads.

The choice to make at each turn is which of your stones to move, assuming multiple moves are possible, which is not always the case.
A stone may move the rolled number of squares, _unless_ the destination square:
- is occupied by a stone of the same player
- is a rosette (marked with a cross) occupied by the opponent
- is off the board, to finish one must land exactly on the finish

The movement of a stone has the following consequences:
- When moving to a square occupied by the opponent, the opponent's stone is captured and moved back to the start.
- When moving to a rosette square, the player gets another turn. Otherwise the turn transfers to the opponent.

Finally, when a player has no legal moves, or a 0 is thrown, they must pass. Passing is only allowed when there are no legal moves.

<a name="play"/>

## Play Game!
The easiest way to play the game against TD-Ur is to go to this link. __TODO: add link once it's public, changing the imports from drive to online__
This will run the code remotely (in a Colab notebook), which means you don't need to install anything, but also that there is a slight delay after making a move.

If you are familiar with Python you can download this repository and run the Jupyter notebook play_game.ipynb.
The dependencies are jax, numpy, matplotlib and ipywidgets.


<a name="strat"/>

## Strategy

To illustrate the types of strategic decisions the game presents, here are two game positions.

![plot](./images/StrategyExamples.png)

On the left board it's red's turn to move, with a throw of 2 (as indicated in the rightmost yellow squares).
Red can move 3 stones, which we label by the coordinate of the stone that is moved: 
- h3 moves that stone to the finish
- c3 moves to a rosette, giving another turn
- d2 captures the opponent's most advanced stone on f2

They all look good, but also all have downsides. Moving h3 to the finish might not be the most urgent as it is safe already. 
It also gives up the opportunity to land on the rosette with a future throw of 1.
Moving c3 to the rosette is risky, it will give the best result if the next roll of the dice is good, say another 2, but if it's bad it achieves the least.
Finally moving d2 also gives up the middle rosette, which is a strong outpost that can block the opponent's progress for the whole game if it is kept (as capture on the rosette is not allowed).

TD-Ur chose to capture with d2 here.

On the right board, it is again the red player's turn to move with a throw of 1.
There are 4 available moves
- b1 moves to a rosette and gives another turn
- b2 moves that stone closer to the finish and reduces its chance of being captured
- g3 moves that stone to the finish
- e3 puts another stone on the board
- 
Considerations are similar here to the other board, b3 could turn out very well if the next throw is good, but we could also throw a 0, or even a 2 would be bad.
Moving g3 to the finish might not seem most urgent, but only a rather rare throw of 1 can do this, so if we postpone it for too long we might end up with our last stone stuck there.

TD-Ur chose to move g4 to the finish.

<a name="tdur"/>

# TD-Ur

Ur's successor Backgammon was the first game in which human experts were surpassed by a learned AI, called [TD-Gammon](https://www.csd.uwo.ca/~xling/cs346a/extra/tdgammon.pdf) by Tesauro.
Since Ur is thought to be a simpler predecessor of Backgammon, it seems appropriate and sufficient to use the techniques of TD-Gammon for Ur, resulting in the TD-Ur presented here.

--TODO: add comment about performance

The essential techniques are using a neural network to parametrize a value function, which is learned using TD(lambda) and a 2-ply lookahead search.

If this means nothing to you, what follows is a basic explanation of all the techniques used.
For more detail I recommend the book [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html), by Sutton and Barto, and these [video lectures](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ) by David Silver.

<a name="rl"/>

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

<a name="value"/>

## Value Function

To maximize the return, or win probability, rather than engineering by hand, using our own knowledge of the game, complicated rules on what actions to take, in the reinforcement learning paradigm, we let the agent learn how to do it for itself.
This often involves having the agent learn a _value function_, an estimate of the expected return in the current state, or the win probability in our case.
This value function can be allowed to depend only on the current state, or also on the chosen action, when it's called an action value.
For games such as this where the transition to a new state can be decomposed into first a deterministic part and then a probabilistic part, it is convenient to choose a middle ground, namely the afterstate. So the input to our value function will be the board configuration and whose turn it is, but it does not include the next roll of the dice.

Given such an afterstate s_a obtained from the deterministic part of the transition after taking action a in state s, we want to learn to estimate a number
![equation](https://latex.codecogs.com/svg.latex?0%20%5Cleq%20v%28s_a%29%20%5Cleq%201%7B%5Ccolor%7BDarkOrange%7D%20%7D) <!-- 0 <= v(s_a) <= 1 -->
representing the probability of winning the game.

### Neural Network
We will use a neural network to represent this function. A simple fully connected network with one hidden layer will do. Using a sigmoid activation function on the final layer guarantees that the value will be between 0 and 1.
Initializing the weights as random small numbers, our initial estimate will have values near 0.5, with no structure or meaning to it.


To understand why we use a neural network for this, let's count the total number of afterstates in Ur. 
For both players each of the 7 stones can be in one of 4 groups of squares:
- the start: ![equation](https://latex.codecogs.com/gif.latex?s_r)
- the finish: ![equation](https://latex.codecogs.com/gif.latex?f_r)
- the home row: ![equation](https://latex.codecogs.com/gif.latex?h_r)
- the middle row: ![equation](https://latex.codecogs.com/gif.latex?m_r)

Naming these as indicated above, with the subscript r indicating the red player's stones and similarly with subscript b for blue, we obtain the total number of afterstates as follows.
We sum over the number of stones at the start for both players, from 0 to 7, over the number of stones at the finish, again for both players from 0 up to 7 - their respective start numbers. Then for both players over the number of stones in the home row from 0 to 6 - (stones at start or finish) (to a maximum of 6 here because there are only 6 squares in the home row. The remaining number of stones must be in the middle row.
For the home row and middle row there are multiple configurations, in each home row we have 6 squares over which to divide ![equation](https://latex.codecogs.com/gif.latex?h_i) stones and in the middle row we have 8 squares over which to divide ![equation](https://latex.codecogs.com/gif.latex?\small&space;m_r&space;&plus;&space;m_b) stones. 
This results in the number of afterstates:

![equation](https://latex.codecogs.com/gif.latex?\small&space;\sum_{s_r=0}^7\sum_{f_r=0}^{7-s_r}&space;\sum_{h_r=0}^{6&space;-&space;s_r&space;-&space;f_r}&space;\sum_{s_b=0}^7\sum_{f_b=0}^{7-s_b}&space;\sum_{h_b=0}^{6&space;-&space;s_b&space;-&space;f_b}&space;\binom{6}{h_r}&space;\binom{6}{h_b}&space;\binom{8}{m_r&space;&plus;&space;m_b}&space;=&space;21.342.488)
<!-- \text{\# states}  = \sum_{s_r=0}^7\sum_{f_r=0}^{7-s_r} \sum_{h_r=0}^{6 - s_r - f_r} 
\sum_{s_b=0}^7\sum_{f_b=0}^{7-s_b} \sum_{h_b=0}^{6 - s_b - f_b} \binom{6}{h_r} \binom{6}{h_b} \binom{8}{m_r + m_b} = 
21.342.488 -->

Unlike more complicated games like backgammon, or even worse chess or go, here this does still fit in memory.
So it _is_ possible to not use a neural network but explicitly tabulate all possible afterstates and estimate a value for each one separately.
Nevertheless, this is still not the best way to go, because there will be no generalization from states to other similar states whatsoever.
The agent will have to visit each of these states many times during training to obtain a good approximation of the value function.
In contrast with a neural network, states that are very similar will have similar values, and the network will learn which differences are important and which are not.

<a name="plcy"/>

## Policy

The value function as presented above is actually not yet completely well defined, because the win probability in a given (after)state depends on the moves chosen subsequently. 
More formally, it depends on the _policy_ used, where a policy is a map from a state to an action, i.e. a way of choosing a move.

What we are after is the optimal policy, choosing the action that maximizes the win probability against an opponent also playing optimally.
Given the true value function assuming optimal play, we can easily obtain the optimal policy, simply by choosing the action that maximizes the afterstate value.

Of course we don't have this either, but we can work our way towards it. 
We start with some random value function as defined in the previous section. 
From this we derive a policy as above, this is called the _greedy policy_, because it maximizes the value in the next state, 
and does not take into consideration that to obtain the longer term maximum we might have to accept short term losses.
Then we update our estimate of the value function to more accurately reflect the win probability of this policy (see below how).
Then we use this updated value function to define a new policy in the same way.
Iterating this is called _policy iteration_, and will allow us to converge to the optimal policy and value function.

<a name="td"/>

## TD Learning

Now how do we update our estimate of the value function?
We do this using _Temporal Difference (TD) Learning_, whose basic premise is that future states have more information than present states.
The logic is that a. they are closer to the final win/loss state, and b. they have experienced more environment dynamics.
In the extreme case, the game has finished and we know the outcome.

So what we can do is compute the value of the current state, ![equation](https://latex.codecogs.com/gif.latex?v(S_t)), and then take a step (using the policy whose value we're estimating), and compute the value ![equation](https://latex.codecogs.com/gif.latex?v(S_{t+1})) there.
Assuming that the latter is more accurate, we want to update our estimate to move ![equation](https://latex.codecogs.com/gif.latex?v(S_t)) closer to ![equation](https://latex.codecogs.com/gif.latex?v(S_{t+1})). In other words we want to minimize ![equation](https://latex.codecogs.com/gif.latex?(v(S_{t&plus;1})&space;-&space;v(S_t))^2), but only through ![equation](https://latex.codecogs.com/gif.latex?v(S_t)). Now v is a neural network, parametrized by weights w, so we can do this using gradient descent, resulting in the weight update

![equation](https://latex.codecogs.com/gif.latex?w_{t&plus;1}&space;=&space;w_t&space;&plus;&space;\alpha&space;(R_{t&plus;1}&space;&plus;&space;v(S_{t&plus;1})&space;-&space;v(S_t))&space;\nabla_w&space;v(S_t))
<!-- w_{t+1} = w_t + \alpha (R_{t+1} + v(S_{t+1}) - v(S_t)) \nabla_w v(S_t) -->
This is called _semi-gradient descent_ because we keep the future estimate fixed. This is very important, if we do not do this we are not making use of the fact that future estimates are more accurate than present estimates.

It is also an example of _boostrapping_: improving our estimate of the optimal value function based on earlier estimates.

<a name="eligibility" />

## TD(lambda)

What is described above amounts to 1-step TD, where we look one step into the future, and only update the weights according to the current state's value.
This is a usually suboptimal solution to the _credit assignment problem_, which is the problem of identifying which of the choices in the past were most responsible for the current situation.

A more elegant and usually more efficient solution to this problem is called TD(lambda).
This uses an _eligibility trace_, which is an exponentially decaying average of the gradient of the value function,

![equation](https://latex.codecogs.com/gif.latex?z_{t&plus;1}&space;=&space;\lambda&space;z_t&space;&plus;&space;\nabla_w&space;v)
<!-- z_{t+1} = \lambda z_t + \nabla_w v -->
where ![equation](https://latex.codecogs.com/gif.latex?\small&space;\lambda&space;\in&space;[0,&space;1]) pecifies the decay rate, and thus the time scale over which previous moves are considered to have an impact on the current value.

When lambda is 0 we recover the previous 1-step TD, where only the last actions is held responsible for the result.
At the other extreme when lambda is 1, all previous actions are held equally responsible, this is called Monte Carlo learning.
Moves clearly can have a longer term effect, but at the same time the random element in the game will wash out their influence over time. 
So the optimal value is somewhere in between.

It is important to re-initialize the eligibility trace to zero at the start of each training game, as no weights have any responsibility for creating the starting state.

<a name="srch"/>

## Search

Again following TD-Gammon, we use a 2-ply search. 
A ply is a single player's move, so a one ply search would be to look at all available moves, evaluate their afterstates, and choose the move with the highest value afterstate (or lowest for the blue player).
A two ply search, as it is done in TD-Gammon and as we do here, works as follows. The afterstates found after 1-ply are not full states, they need to be supplemented with a new roll of the dice. We complete them in all possible ways, that is, with die rolls from 0 to 4.
This then forms another full state, typically with the opponent to move (unless landing on a rosette). 
For each of these states, we do another 1-ply search to find the move that is best for whoever's turn it is, and compute the value of the resulting afterstate.
So now for a given initial move, we have the 5 possible die rolls combined with the value of the best following move's after state.
We sum these up, weighted with the probability of the corresponding die roll (these probabilities are (1, 4, 6, 4, 1)/16 for throws of 0, 1, 2, 3, 4 respectively).
This gives the expected 2-ply afterstate value.
The move we choose then is the one that maximizes this value.

We do this both during training and during play after training. 

<a name="selfplay"/>

## Self-Play

So far we have described the methods by which the agent can improve through playing the game. But who does it play against?
We want it to be able to play a lot of games quickly, so its opponent should also be an AI.
We also want its opponent to also grow in strength, as it won't learn much by playing against random moves for example.

The obvious answer is that it plays itself.
This brings the risk that it will learn a strategy that works well against the same strategy but fails against another strategy it does not know.
One could train it against an ensemble of previous versions or instantiations of itself to avoid this.
Here however, due to the simplicity of Ur and the random component that brings a degree of exploration, this is unlikely to be an issue.

It can be a bit confusing to have the agent play both sides, and update its weights on all moves done during training.
To make this clearer, we have set it up so that no matter whose turn it is, 
we always want to improve our estimate of the value function as seen from the point of view of the red player. 
The eligibility trace tracks the influence of older decisions on this value function.
So neither of these, the value function or the eligibility trace, or how the updates are done, depend on whose turn it is.
The only place where this enters is that if it is the red player's turn, they will choose the move that maximizes the value function, 
while if it is the blue player's turn they will choose the move that minimizes the value function.

<a name="details"/>

## Implementation Details
Finally we discuss a few implementation details.

### Board Representation

For the board itself, internally a different representation is used, where the board is unrolled so that movement is always from left to right.
The middle row is duplicated, with stones always staying on their own side but of course still capturing opponent stones on the other side of the region that maps to the middle row.

This simplifies the internal game logic and results in a 2 by 16 array.
The values in the array are simply zeros for absence of a stone and ones for presence, 
with the color of the stone determined by the row it is on.
The only exception are the first and last column indicating the number of stones at the start or finish, which range from 0 to 7.

Before we input this board to the value network, we flip it if necessary so that the network always gets fed boards where it is  the red player's turn.
Then we flatten the board into a 32-dimensional vector.
To always output the value function as seen from the red player's point of view, which seems conceptually the simplest,
we directly output the value if it actually was the red player's turn, and otherwise we output 1 - the computed value.
This also enforces a symmetry present in the game, where if we mirror the board along the middle row and change turn, we obtain the same state, only with the roles of red and blue reversed.

### Jax

To speed up the training, we made heavy use of jax, which has sped it up by a factor of 100, allowing play of about 1000 moves per second on a CPU.
This involves adding `jit` decorators around often used functions, which automatically compiles them the first time they're run.
It is actually a bit more involved, but only slightly. 
For jax to be able to do this, it traces the function with abstract inputs that have only a shape and a type, but not values.
If there are conditionals on values this will fail.
So to use jax to its full extent, these conditionals should be converted as much as possible to arithmetic operations, which is what we have done.
We have also used `grad` to compute the derivative of the value function in a single line, and `vmap` to compute the values of all legal moves in one batch, 
without having to explicitly add the batch dimension in the code.

### TD error

Finally, for the final TD error, when a game is finished, we use the reward minus the previous value, rather than the reward plus the next value minus the previous value. This is because we know the total return of the final state exactly, it is simply one if we have won and zero otherwise, so there is no need to bootstrap.

### Hyperparameters
There are a number of hyperparameters in this setup:
- hidden units: 40 (20, 40, 80)
- learning rate: 0.01 (0.1, 0.01, 0.001)
- lambda: 0.9 (0.8, 0.9, 0.99)
- epsilon: 0 (0, 0.0001)
- search depth: 2 (1, 2)
- training episodes: 8000 (500-20.000)

These need to be chosen manually, and this choice can dramatically affect the performance.
To find good hyperparameters, one needs to train an agent with them and then evaluate the trained agent against other agents trained with different hyperparameters. 
This quickly gets very time consuming. Due to the element of luck in Ur, one needs to compete two agents for a lot of games to be sure one is really better.
The values in brackets above are the ones I have tried, in most combinations. 
The value before that denotes the best values I have found, though I haven't done anything like an exhaustive search.

TD-Gamma used 80 hidden units, so since Ur is much simpler 40 seems reasonable.
It also uses a 2-ply search. Here it seems likely that searching for a bigger depth can yield more improvements, though with diminishing returns because of an increasing contribution of luck, and at the cost of slower training and playing.
The other hyperparameters, the learning rate, lambda and the exploration parameter epsilon, I haven't found TD-Gammon's values.
The value of 0 for epsilon also seems reasonable since the randomness of the die rolls already gives rise to exploration of different states.
Finally the values of the learning rate and the TD parameter lambda are commonly chosen values as well.

