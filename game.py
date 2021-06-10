import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from jax import jit, vmap
import jax.numpy as jnp
from jax.ops import index, index_update
from functools import partial


BOARD_ROSETTES = [4, 8, 14]
BOARD_WIDTH_INTERNAL = 16
BOARD_MID_START = 5
BOARD_MID_ENDED = 13
BOARD_START = 0
BOARD_FINISH = 15
BOARD_SAFE_SQUARE = 8
BOARD_PIECES = 7


class Ur:
    """
    The Ur board looks like this:
    ------------------------------------
    | 4 | 3 | 2 | 1 | s |  f | 14 | 13 |
    ------------------------------------
    | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
    ------------------------------------
    | 4 | 3 | 2 | 1 | s |  f | 14 | 13 |
    ------------------------------------
    where s and f are not part of the board, but can be seen as the places where stones that still
    have to go through (s) or have already finished (f) are located.
    To fully specify a game state, this needs to be supplemented with the last die throw and whose turn it is.

    we unroll this and copy the middle row that's shared between the two players, to give:
    --------------------------------------------------------------------
    s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | f |
    --------------------------------------------------------------------
    s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | f |
    --------------------------------------------------------------------
    """

    def __init__(self):
        # board
        self.start = 0
        self.finish = 15
        self.rosettes = [4, 8, 14]
        self.safe_square = 8
        self.mid_start = 5
        self.mid_ended = 13
        self.board_width_internal = 16

        # piece
        self.n_pieces = 7

        # die
        self.n_die = 4
        self.die_faces = 2
        self.rolls = np.arange(5)
        self.probabilities = np.array([1, 4, 6, 4, 1]) / 16

        # display
        self.display_width = 8

        # jitted function
        self._get_new_boards = jit(vmap(self._get_new_board, in_axes=(None, 0, None, None)))

        # state
        self.rolled = self.turn = self.winner = self.board = self.move_count = self.backup_state = None
        self.reset()

    def reset(self):
        """Reset the game to the initial state."""
        self.turn = 0
        self.winner = -1
        self.board = np.zeros(shape=(2, self.finish + 1), dtype=np.int8)
        self.board[0, self.start] = self.n_pieces
        self.board[1, self.start] = self.n_pieces

        self.move_count = 0
        self._roll()

    def _roll(self):
        """Roll the dice, store result in attribute `rolled`."""
        self.rolled = np.sum(np.random.randint(self.die_faces, size=self.n_die))

    def legal_moves(self):
        """Return a list of legal moves.

        Relies on jitted function `_legal_moves_array`.

        Returns:
            List of integers representing legal squares to move from, counted along the route.
        """
        if self.rolled == 0:
            moves = []
        else:
            moves_array = self._legal_moves_array(self.board, self.turn, self.rolled)
            moves = np.where(moves_array)[0].tolist()
        return moves if moves else ['pass']

    def play_move(self, move):
        """Play the given legal move.

        Plays the move on the board, changes turn (if appropriate) and rolls the dice again.
        Also increments `move_count` and sets `winner` to the player who moved if the game is won.
        Relies on the jitted function `_get_new_board`.


        Args:
            move: Integer representing the square to move from, as counted along the route.

        """
        self.move_count += 1

        if move == 'pass':
            self._change_turn()
            self._roll()
        else:
            new_board, new_turn, new_winner = self._get_new_board(self.board, move, self.rolled, self.turn)

            self.board = np.array(new_board)

            if new_winner != -1:
                self.winner = self.turn
            else:
                self.turn = int(new_turn)  # need to convert from DeviceArray
                self._roll()

    def _change_turn(self):
        """Change turn, stored in attribute `turn`."""
        self.turn = (self.turn + 1) % 2

    def other(self):
        """Return the number of the player whose turn it is not."""
        return (self.turn + 1) % 2

    def reward(self):
        """Return 1 if game was won by player 0, or 0 otherwise. So always seen from player 0's perspective!"""
        if not self.has_finished():
            return 0
        elif self.winner == 0:
            # note if there's a winner the turn no longer changes
            return 1
        else:
            return 0

    def has_finished(self):
        """Return True if the game has finished, False if not."""
        return self.winner != -1

    def get_state(self):
        """Return the current state of the game.

        Returns:
            A game state of the form `(board, turn, rolled, winner, move_count)`.
        """
        return self.board.copy(), self.turn, self.rolled, self.winner, self.move_count

    def set_state(self, state):
        """Set the game to the input state.

        Args:
            state: A game state of the form `(board, turn, rolled, winner, move_count)`.
        """
        self.board, self.turn, self.rolled, self.winner, self.move_count = state

    def backup(self):
        """Store the current state of the game in the attribute `backup_state`."""
        self.backup_state = self.get_state()

    def restore_backup(self):
        """Restore the current state of the game from the attribute `backup_state`."""
        self.board, self.turn, self.rolled, self.winner, self.move_count = self.backup_state

    def simulate_moves(self, moves):
        """Give afterstates resulting from moves.

        Relies on function `_get_new_boards`, a `vmap` of `get_new_board`

        Args:
            moves: A list of legal moves.

        Returns:
            A list of tuples (board, turn, winner), one for each move.
        """
        return self._get_new_boards(self.board, jnp.array(moves), self.rolled, self.turn)

    @staticmethod
    @partial(jit, static_argnums=(1, 2))
    def _legal_moves_array(board, turn, rolled):
        """Return a boolean array indicating which moves are legal.

        Jitted function.

        Args:
            board: The current board.
            turn: Whose turn it is.
            rolled: The die roll

        Returns:
            A jnp boolean vector with Trues for the legal moves.
        """
        # moves that don't move a stone beyond the finish, based only on the die roll
        start_squares = board[turn, 0:BOARD_FINISH + 1 - rolled]
        # the corresponding end squares
        end_squares = board[turn, rolled: BOARD_FINISH + 1]

        # start square contains a stone to move
        moves_with_legal_start = start_squares > 0

        # end square does not contain player stone (or is finish)
        moves_with_legal_end = end_squares == 0
        moves_with_legal_end = index_update(moves_with_legal_end, index[-1], True)

        # it's not a capture on the safe space
        safe_space = jnp.zeros(BOARD_FINISH + 1 - rolled, dtype='bool')
        safe_space = index_update(safe_space, index[BOARD_SAFE_SQUARE - rolled], True)
        opponent_present = board[(turn + 1) % 2, rolled: BOARD_FINISH + 1] > 0
        no_illegal_capture = ~(opponent_present & safe_space)

        legal_moves = moves_with_legal_start & moves_with_legal_end & no_illegal_capture

        return legal_moves

    @staticmethod
    @jit
    def _get_new_board(board, move, rolled, turn):
        """Return board after given move is played.

        Jitted function.

        Args:
            board: The board before the move is played.
            move: The move (integer indicating the square to move from as counted along the route).
            rolled: The die roll.
            turn: Whose turn it is (0 or 1).

        Returns:
            A tuple of the form `(board, turn, winner)` giving the resulting board and turn, and `winner` indicating
            whether the game has been won and by who.
        """
        end = move + rolled
        # move player's stone forward
        indices_x, indices_y, values = [turn, turn], [move, end], [board[turn, move] - 1, board[turn, end] + 1]

        # construct auxiliary boards to help with logic
        rosette_board = jnp.zeros(shape=BOARD_WIDTH_INTERNAL, dtype='int8')
        for i in BOARD_ROSETTES:
            rosette_board = index_update(rosette_board, i, 1)
        capture_board = jnp.zeros(shape=BOARD_WIDTH_INTERNAL, dtype='int8')
        capture_board = index_update(capture_board, (index[BOARD_MID_START:BOARD_MID_ENDED]), 1)

        # change turn, unless ending on a rosette
        other = (turn + 1) % 2
        new_turn = (turn + 1 + rosette_board[end]) % 2

        # capture, if opponent present and in capturable area
        indices_x, indices_y = indices_x + [other, other], indices_y + [end, BOARD_START]
        values = values + [(1 - capture_board[end]) * board[other, end],
                           board[other, BOARD_START] + capture_board[end] * board[other, end]]

        new_board = index_update(board, (tuple(indices_x), tuple(indices_y)), tuple(values))

        not_finished = jnp.sign(BOARD_PIECES - new_board[turn, BOARD_FINISH])
        new_winner = not_finished * -1 + (1 - not_finished) * turn

        return new_board, new_turn, new_winner

    def check_valid_board(self):
        """Return True if current board is valid, otherwise returns a string describing the first found violation."""
        board = self.board

        if board.dtype != 'int8':
            return 'not ints'

        on_board = board[:, self.start + 1:self.finish]
        if jnp.max(on_board) > 1:
            return 'more than one stone on square'
        if jnp.min(on_board) < 0:
            return 'less than 0 stones on square'

        for player in [0, 1]:
            if jnp.sum(board[player, self.start: self.finish + 1]) != self.n_pieces:
                return f'number of pieces not conserved (player {player})'
            if not (0 <= board[player, self.start] <= self.n_pieces):
                return f'illegal start pieces (player {player})'
            if not (0 <= board[player, self.finish] <= self.n_pieces):
                return f'illegal finish pieces (player {player})'

        overlap_board = board[:, self.mid_start:self.mid_ended]
        if not (jnp.sum(overlap_board, axis=0) <= jnp.ones(self.mid_ended - self.mid_start, dtype='int8')).all():
            return 'overlapping stones'

        if self.winner != -1:
            if board[self.winner, self.finish] != self.n_pieces:
                return "winner hasn't finished yet"
            if board[(self.winner + 1) % 2, self.finish] == self.n_pieces:
                return "loser has won before winner"
        return True

    # Last 3 functions only for display purposes
    def display(self):
        """Display the game board in the current state, in the conventional shape."""
        board_display = self._reshape_board()

        cmap = colors.ListedColormap(['b', 'w', 'r', 'y'])

        plt.imshow(board_display, cmap=cmap, extent=(0, self.display_width, 3, 0), vmin=-1, vmax=3)
        self._annotate_board()

    def _reshape_board(self):
        """Turn the internal 2x16 board into the conventional 3x8 shape."""
        reshaped_board = np.zeros(shape=(3, self.display_width), dtype=np.int8) + 3
        reshaped_board[1] = (self.board[0, self.mid_start:self.mid_ended] -
                             self.board[1, self.mid_start:self.mid_ended])
        for player in [0, 1]:
            sign = (1 - 2 * player)
            reshaped_board[2 * player, :4] = sign * np.flip(self.board[player, 1:self.mid_start])
            reshaped_board[2 * player, -(self.finish - self.mid_ended):] = sign * np.flip(
                self.board[player, self.mid_ended:-1])
        return reshaped_board

    def _annotate_board(self):
        """Add labels and decorations."""
        t_x, t_y = 4.2, 0.7
        # stones at start and finish
        stats = [self.board[ij] for ij in [(0, self.start), (0, self.finish), (1, self.start), (1, self.finish)]]
        player_colors = ['r', 'r', 'b', 'b']
        x_start = 4
        x_finish = self.display_width - (self.finish + 1 - self.mid_ended)
        positions = [(x_start, 0), (x_finish, 0), (x_start, 2), (x_finish, 2)]
        for s, c, (x, y) in zip(stats, player_colors, positions):
            plt.text(x + .3, y + .7, f'{s}', fontsize=24, color=c)

        # turn and throw
        plt.text(-0.8, 0.7 + 2 * self.turn, f'{self.rolled}', fontsize=24, color=('r' if self.turn == 0 else 'b'))

        # rosettes
        for (y, x) in [(0, 0), (2, 0), (1, 3), (0, 6), (2, 6)]:
            plt.text(t_x + x - 4 - .12, t_y + y + .28, 'X', fontsize=54, color='black')

        # make it pretty
        ax = plt.gca()
        ax.set_xticks(np.arange(0, self.display_width + 1, 1))
        ax.set_yticks(np.arange(0, 3 + 1, 1))
        ax.grid(color='black', linewidth=5, fillstyle='full')
        ax.tick_params(labelbottom=False, labelleft=False, color='w')
