import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors


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

    def __init__(self, mode='classic'):
        # board
        self.start = 0
        self.finish = 15
        self.turn_index = 16
        self.rosettes = [4, 8, 14]
        self.safe_square = 8
        self.safety_length_end = 2 if mode == 'classic' else 0
        self.safety_length_start = 4

        # piece
        self.n_pieces = 7

        # die
        self.n_die = 4
        self.die_faces = 2

        # display
        self.display_width = 8 if mode == 'classic' else 10

        # state
        self.rolled = self.turn = self.winner = self.board = self.move_count = self.backup_data = None
        self.reset()

    def reset(self):
        self.turn = 0
        self.winner = -1
        self.board = np.zeros(shape=(2, self.finish + 2), dtype=np.int8)
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
            return ['pass']

        # start square is within range and contains a stone to move
        moves_with_legal_start = self.board[self.turn, :self.finish + 1 - self.rolled] > 0

        # end square is within range and does not contain player stone (or is finish)
        moves_with_legal_end = self.board[self.turn, self.rolled: self.finish + 1] == 0
        moves_with_legal_end[-1] = True

        # almost legal
        moves = np.where(moves_with_legal_start & moves_with_legal_end)[0]

        # last check: moving to opponent occupied safe square
        if self.board[self.other(), self.safe_square] == 1:
            moves = moves[moves != self.safe_square - self.rolled]

        moves = moves.tolist()

        if not moves:
            moves = ['pass']

        return moves

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
        if end == self.safe_square and self.board[self.other(), self.safe_square] == 1:
            return False

        # otherwise it's legal
        return True

    def play_move(self, move, checks=True):
        # move is the square whose stone to move
        if checks:
            if self.winner != -1:
                print('game already finished.')
                return

            if not self.is_legal_move(move):
                # lose game
                self.winner = self.other()
                return

        self.move_count += 1

        if move == 'pass':
            self.change_turn()
            self.roll()
            return

        end = move + self.rolled
        self.board[self.turn, move] -= 1
        self.board[self.turn, end] += 1

        # captures
        if (self.board[self.other(), end] == 1 and  # opponent stone present
                self.safety_length_start < end < self.finish - self.safety_length_end):  # and in capturable zone
            self.board[self.other(), end] = 0
            self.board[self.other(), self.start] += 1

        # check if the game is finished
        if self.board[self.turn, self.finish] == self.n_pieces:
            self.winner = self.turn
        else:
            if end not in self.rosettes:
                self.change_turn()
            self.roll()

    def other(self):
        return (self.turn + 1) % 2

    def change_turn(self):
        self.turn = self.other()
        self.board[self.turn, self.turn_index] = 0
        self.board[self.other(), self.turn_index] = 1

    def reward(self):
        if self.winner == -1:
            return 0
        elif self.winner == 0:
            # note if there's a winner the turn no longer changes
            return 1
        else:
            return -1

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
        self.play_move(move, checks)
        reward = self.reward()
        board = self.board.copy()
        self.restore_backup()
        return reward, board

    # testing
    def check_valid_board(self):
        board = self.board

        if board.dtype != 'int8':
            return 'not ints'

        on_board = board[:, self.start + 1:self.finish]
        if np.max(on_board) > 1:
            return 'more than one stone on square'
        if np.min(on_board) < 0:
            return 'less than 0 stones on square'
        if np.sum(board[0][self.start: self.finish + 1]) != self.n_pieces:
            return 'number of pieces not conserved (player 0)'
        if np.sum(board[1][self.start: self.finish + 1]) != self.n_pieces:
            return 'number of pieces not conserved (player 1)'
        if not (0 <= board[0, self.start] <= self.n_pieces):
            return 'illegal start pieces (player 0)'
        if not (0 <= board[1, self.start] <= self.n_pieces):
            return 'illegal start pieces (player 1)'
        if not (0 <= board[0, self.finish] <= self.n_pieces):
            return 'illegal start pieces (player 0)'
        if not (0 <= board[1, self.finish] <= self.n_pieces):
            return 'illegal start pieces (player 1)'

        roll_info = board[:, self.turn_index]
        if not (min(roll_info) == 0 and 0 <= max(roll_info) <= (self.die_faces - 1) * self.n_die):
            return 'illegal roll/turn'

        overlap_board = board[:, self.safety_length_start + 1: -self.safety_length_end - 2]
        if not (np.sum(overlap_board, axis=0) <= np.ones(self.finish - self.safety_length_start
                                                         - self.safety_length_end - 1, dtype='int8')).all():
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

        cmap = colors.ListedColormap(['b', 'gray', 'r', 'w'])

        plt.imshow(board_display, cmap=cmap, extent=(0, self.display_width, 3, 0), vmin=-1, vmax=3)
        self.annotate_board()

    def reshape_board(self):
        reshaped_board = np.zeros(shape=(3, self.display_width), dtype=np.int8) + 3
        reshaped_board[1] = (self.board[0, self.safety_length_start + 1:-self.safety_length_end - 2] -
                             self.board[1, self.safety_length_start + 1:-self.safety_length_end - 2])
        for player in [0, 1]:
            sign = (1 - 2 * player)
            reshaped_board[2 * player, :4] = sign * np.flip(self.board[player, 1:self.safety_length_start + 1])
            if self.safety_length_end:
                reshaped_board[2 * player, -self.safety_length_end:] = sign * np.flip(
                    self.board[player, -self.safety_length_end - 2:-2])
        return reshaped_board

    def annotate_board(self):
        t_x, t_y = 4.2, 0.7
        # stones at start and finish
        stats = [self.board[ij] for ij in [(0, self.start), (0, self.finish), (1, self.start), (1, self.finish)]]
        player_colors = ['r', 'r', 'b', 'b']
        x_start = 4
        x_finish = self.display_width - self.safety_length_end - 1
        positions = [(x_start, 0), (x_finish, 0), (x_start, 2), (x_finish, 2)]
        for s, c, (x, y) in zip(stats, player_colors, positions):
            plt.text(x + .3, y + .7, f'{s}', fontsize=24, color=c)

        # turn and throw
        plt.text(-0.8, 0.7 + 2 * self.turn, f'{self.rolled}', fontsize=24, color=('r' if self.turn == 0 else 'b'))

        # rosettes
        for (y, x) in [(0, 0), (2, 0), (1, 3), (0, 6), (2, 6)]:
            plt.text(t_x + x - 4 - .08, t_y + y + .2, 'X', fontsize=48, color='black')

        # make it pretty
        ax = plt.gca()
        ax.set_xticks(np.arange(0, self.display_width, 1))
        ax.set_yticks(np.arange(0, 3, 1))
        ax.grid(color='black', linewidth=2)
        ax.tick_params(labelbottom=False, labelleft=False)
