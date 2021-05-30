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


@partial(jit, static_argnums=(1, 2))
def legal_moves_array(board, turn, rolled):
    # start square contains a stone to move, and won't move it beyond the finish
    moves_with_legal_start = board[turn, 0:BOARD_FINISH + 1 - rolled] > 0

    # end square is within range and does not contain player stone (or is finish)
    moves_with_legal_end = board[turn, rolled: BOARD_FINISH + 1] == 0
    moves_with_legal_end = index_update(moves_with_legal_end, index[-1], True)

    # it's not a capture on the safe space
    safe_space = jnp.zeros(BOARD_FINISH + 1 - rolled, dtype='bool')
    safe_space = index_update(safe_space, index[BOARD_SAFE_SQUARE - rolled], True)
    opponent_present = board[(turn + 1) % 2, rolled: BOARD_FINISH + 1] > 0
    no_illegal_capture = ~(opponent_present & safe_space)

    legal_moves = moves_with_legal_start & moves_with_legal_end & no_illegal_capture

    return legal_moves


@jit
def get_new_board(board, move, rolled, turn):
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

    board_new = index_update(board, (tuple(indices_x), tuple(indices_y)), tuple(values))

    return board_new, new_turn


get_new_boards = jit(vmap(get_new_board, in_axes=(None, 0, None, None)))


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
    ------------------------------------------------------------------------
    s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | f | t |
    ----------------------------------------------------------------------
    s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | f | t |
    ------------------------------------------------------------------------
    where t = 0 for the player whose turn it is not, and 1 for the other.
    """

    def __init__(self):
        # board
        self.start = BOARD_START
        self.finish = BOARD_FINISH
        self.rosettes = BOARD_ROSETTES
        self.safe_square = BOARD_SAFE_SQUARE
        self.mid_start = BOARD_MID_START
        self.mid_ended = BOARD_MID_ENDED

        # piece
        self.n_pieces = 7

        # die
        self.n_die = 4
        self.die_faces = 2

        # display
        self.display_width = 8

        # state
        self.rolled = self.turn = self.other = self.winner = self.board = self.move_count = self.backup_data = None
        self.reset()

    def reset(self):
        self.turn = 0
        self.other = 1
        self.winner = -1
        self.board = np.zeros(shape=(2, self.finish + 1), dtype=np.int8)
        self.board[0, self.start] = self.n_pieces
        self.board[1, self.start] = self.n_pieces

        self.move_count = 0
        self.roll()

    def roll(self):
        self.rolled = np.sum(np.random.randint(self.die_faces, size=self.n_die))

    def legal_moves_slow(self):
        if self.rolled == 0:
            return ['pass']

        moves = []
        for start in range(self.finish + 1 - self.rolled):
            if self.is_legal_move(start):
                moves.append(start)

        if not moves:
            moves = ['pass']

        return moves

    def legal_moves(self):
        if self.rolled == 0:
            moves = []
        else:
            moves_array = legal_moves_array(self.board, self.turn, self.rolled)
            moves = np.where(moves_array)[0].tolist()
        return moves if moves else ['pass']

    # Not used as moves are played by selecting from legal_moves
    def is_legal_move(self, move):
        # 0. a pass is only legal if there are no other moves
        if move == 'pass':
            return self.legal_moves() == ['pass']

        # Conditions under which it's false:
        # 1. either start or end square off the board
        if not self.start <= move <= self.finish - self.rolled:
            return False

        # 2. no player stone to move
        if self.board[self.turn, move] == 0:
            return False

        end = move + self.rolled

        # 3. player occupies end square, and it's not the finish square
        if self.board[self.turn, end] == 1 and end != self.finish:
            return False

        # 4. end square is the safe space and the opponent occupies it
        if end == self.safe_square and self.board[self.other, self.safe_square] == 1:
            return False

        # otherwise it's legal
        return True

    def play_move(self, move):
        self.move_count += 1
        turn_played = self.turn

        if move == 'pass':
            self.change_turn()
            self.roll()
        else:
            new_board, new_turn = get_new_board(self.board, move, self.rolled, self.turn)

            self.board = np.array(new_board)

            if self.board[turn_played, self.finish] == self.n_pieces:
                self.winner = self.turn
            else:
                self.turn = int(new_turn)  # need to convert from DeviceArray
                self.other = (self.turn + 1) % 2
                self.roll()

    def change_turn(self):
        self.turn, self.other = self.other, self.turn

    def reward(self):
        if self.winner == -1:
            return 0
        elif self.winner == 0:
            # note if there's a winner the turn no longer changes
            return 1
        else:
            return 0

    # to allow n-step methods and planning
    def get_state(self):
        return self.board.copy(), self.turn, self.rolled, self.winner, self.move_count

    def set_state(self, state):
        self.board, self.turn, self.rolled, self.winner, self.move_count = state

    def backup(self):
        self.backup_data = self.get_state()

    def restore_backup(self):
        self.board, self.turn, self.rolled, self.winner, self.move_count = self.backup_data

    def simulate_move(self, move, checks=True):
        self.backup()
        self.play_move(move)
        reward = self.reward()
        board = self.board.copy()
        self.restore_backup()
        return reward, board

    def simulate_moves(self, moves):
        return get_new_boards(self.board, jnp.array(moves), self.rolled, self.turn)

    # testing
    def check_valid_board(self):
        board = self.board

        if board.dtype != 'int8':
            return 'not ints'

        on_board = board[:, self.start + 1:self.finish]
        if jnp.max(on_board) > 1:
            return 'more than one stone on square'
        if jnp.min(on_board) < 0:
            return 'less than 0 stones on square'
        if jnp.sum(board[0, self.start: self.finish + 1]) != self.n_pieces:
            return 'number of pieces not conserved (player 0)'
        if jnp.sum(board[1, self.start: self.finish + 1]) != self.n_pieces:
            return 'number of pieces not conserved (player 1)'
        if not (0 <= board[0, self.start] <= self.n_pieces):
            return 'illegal start pieces (player 0)'
        if not (0 <= board[1, self.start] <= self.n_pieces):
            return 'illegal start pieces (player 1)'
        if not (0 <= board[0, self.finish] <= self.n_pieces):
            return 'illegal start pieces (player 0)'
        if not (0 <= board[1, self.finish] <= self.n_pieces):
            return 'illegal start pieces (player 1)'

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
        board_display = self.reshape_board()

        cmap = colors.ListedColormap(['b', 'w', 'r', 'y'])

        plt.imshow(board_display, cmap=cmap, extent=(0, self.display_width, 3, 0), vmin=-1, vmax=3)
        self.annotate_board()

    def reshape_board(self):
        reshaped_board = np.zeros(shape=(3, self.display_width), dtype=np.int8) + 3
        reshaped_board[1] = (self.board[0, self.mid_start:self.mid_ended] -
                             self.board[1, self.mid_start:self.mid_ended])
        for player in [0, 1]:
            sign = (1 - 2 * player)
            reshaped_board[2 * player, :4] = sign * np.flip(self.board[player, 1:self.mid_start])
            reshaped_board[2 * player, -(self.finish - self.mid_ended):] = sign * np.flip(
                self.board[player, self.mid_ended:-1])
        return reshaped_board

    def annotate_board(self):
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
