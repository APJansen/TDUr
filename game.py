import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

# TODO: add roll and turn info to raw state
# TODO: allow printing of raw state
# TODO: -1's as indicators, replace by None or False?
# TODO: hunt down numbers and remove


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
    we unroll this and copy the middle row that's shared between the two players, to give:
    ----------------------------------------------------------------------
    | s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | f |
    ----------------------------------------------------------------------
    | s | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | f |
    ----------------------------------------------------------------------
    """

    def __init__(self, mode='classic'):
        self.rosettes = [4, 8, 14]
        self.safe_space = 8
        self.n_pieces = 7
        self.width = 16
        self.n_die = 4
        self.die_faces = 2
        self.rolled = -1

        self.display_width = 8 if mode == 'classic' else 10
        self.safety_length = 2 if mode == 'classic' else 0

        self.turn = self.winner = self.board = self.count = self.finished = self.backup_data = None
        self.reset()

    def reset(self):
        self.turn = 0
        self.winner = -1
        self.board = np.zeros(shape=(2, self.width), dtype=np.int8)
        self.board[0, 0] = self.n_pieces
        self.board[1, 0] = self.n_pieces
        self.count = 0
        self.finished = False
        self.roll()

    def roll(self):
        self.rolled = np.sum(np.random.randint(self.die_faces, size=self.n_die))

    def legal_moves(self):
        if self.rolled == 0:
            return ['pass']

        moves = []
        for start in range(self.width - self.rolled):
            if self.is_legal_start(start):
                moves.append(start)

        if not moves:
            moves = ['pass']
        return moves

    def is_legal_start(self, start):
        # Conditions under which it's false:
        # 1. no player stone to move
        if self.board[self.turn, start] == 0:
            return False

        # 2. player occupies end square
        end = start + self.rolled
        end_val = self.board[self.turn, end]
        if end_val > 0 and end != self.width - 1:
            return False

        # 3. end square is the safe space and the opponent occupies it
        if end == self.safe_space and self.board[self.other(), self.safe_space] == 1:
            return False

        # otherwise it's legal
        return True

    def play_move(self, start):
        # will only be given legal moves, move is the square whose stone to move
        # returns the reward given to the played move
        self.count += 1
        if start == 'pass':
            self.turn = self.other()
            self.roll()
            return 0

        end = start + self.rolled
        self.board[self.turn, start] -= 1
        self.board[self.turn, end] += 1

        if 4 < end < self.width - 1 - self.safety_length and self.board[self.other(), end] == 1:
            self.board[self.other(), end] = 0
            self.board[self.other(), 0] += 1

        if not self.has_finished():
            self.roll()
            if end not in self.rosettes:
                self.turn = self.other()

    def has_finished(self):
        if self.board[0, -1] == self.n_pieces or self.board[1, -1] == self.n_pieces:
            self.finished = True
            self.winner = self.turn
            self.turn = -1
            return True
        return False

    def other(self):
        return (self.turn + 1) % 2

    # to allow n-step methods and planning
    def get_state(self):
        return self.board, self.turn, self.rolled, self.winner, self.count

    def set_state(self, state):
        self.board, self.turn, self.rolled, self.winner, self.count = state

    def backup(self):
        self.backup_data = self.get_state()

    def restore_backup(self):
        self.board, self.turn, self.rolled, self.winner, self.count = self.backup_data

    # Last 3 functions only for display purposes
    def display(self):
        board_display = self.reshape_board()

        cmap = colors.ListedColormap(['b', 'gray', 'r', 'w'])

        plt.imshow(board_display, cmap=cmap, extent=(0, self.display_width, 3, 0), vmin=-1, vmax=3)
        self.annotate_board()

    def reshape_board(self):
        reshaped_board = np.zeros(shape=(3, self.display_width), dtype=np.int8) + 3
        reshaped_board[1] = self.board[0, 5:-self.safety_length - 1] - self.board[1, 5:-self.safety_length - 1]
        for player in [0, 1]:
            sign = (1 - 2 * player)
            reshaped_board[2 * player, :4] = sign * np.flip(self.board[player, 1:5])
            if self.safety_length:
                reshaped_board[2 * player, -self.safety_length:] = sign * np.flip(
                    self.board[player, -self.safety_length - 1:-1])
        return reshaped_board

    def annotate_board(self):
        t_x, t_y = 4.2, 0.7
        # stones at start and finish
        stats = [self.board[ij] for ij in [(0, 0), (0, -1), (1, 0), (1, -1)]]
        colors = ['r', 'r', 'b', 'b']
        x_start = 4
        x_finish = self.display_width - self.safety_length - 1
        positions = [(x_start, 0), (x_finish, 0), (x_start, 2), (x_finish, 2)]
        for s, c, (x, y) in zip(stats, colors, positions):
            plt.text(x + .3, y + .7, f'{s}', fontsize=24, color=c)

        # turn and throw
        plt.text(-0.8, 0.7 + 2 * self.turn, f'{self.rolled}', fontsize=24, color=('r' if self.turn == 0 else 'b'))

        # rosettes
        for (y, x) in [(0, 0), (2, 0), (1, 3), (0, 6), (2, 6)]:
            plt.text(t_x + x - 4 - .1, t_y + y + .1, 'O', fontsize=36, color='black')

        # make it pretty
        ax = plt.gca()
        ax.set_xticks(np.arange(0, self.display_width, 1))
        ax.set_yticks(np.arange(0, 3, 1))
        ax.grid(color='black', linewidth=2)
        ax.tick_params(labelbottom=False, labelleft=False)
