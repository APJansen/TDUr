from td_ur.agent import TDUr
from td_ur.game import Ur
import ipywidgets as ipyw
from functools import partial
import time
from IPython.display import HTML, display

style_string = """
<style>
.box_style{
    width: auto;
    #border : 2px solid red;
    #height: auto;
    background-color:black;
}
.red_font{
    font-size:60px;
    color:rgb(221,1,0);
}
.blue_font{
    font-size:60px;
    color:rgb(34,80,149);
}
.red_font_small{
    font-size:32px;
    color:rgb(221,1,0);
}
.blue_font_small{
    font-size:32px;
    color:rgb(34,80,149);
}
.rosette_style {
  --size: 80px;
  --line-width: calc(var(--size) / 10);
  --line-horizontal: calc(var(--size) * 3/10);
  --line-vertical: calc(var(--size) * 5.5/10);
  width: var(--size);
  height: var(--size);
  position: relative;

}
.rosette_style::before,
.rosette_style::after {
  position: absolute;
  content: "";
  background: black;
}

.rosette_style::before {
  left: 0;
  bottom: var(--line-horizontal);
  width: 100%;
  height: var(--line-width);
}

.rosette_style::after {
  right: var(--line-vertical);
  top: 0;
  height: 100%;
  width: var(--line-width);
}
.label_font{
    font-size:32px;
    color:rgb(250,201,1);
    text-align:center;
}
.button_font{
    font-size:16px;
    color:black;
    text-align:center;
}
.message_style{
    font-size:32px;
    color:rgb(221,1,0);
    text-align:center;
}
.detailed_message_style{
    font-size:16px;
    color:rgb(221,1,0);
    text-align:center;
}
</style>"""


class InteractiveGame:
    """Class representing the game of Ur, interactively playable against an AI agent.

    To use, run `play` method in Jupyter/Colab notebook.
    """
    def __init__(self, agent_parameters=None, search_plies=2):
        """Construct an interactive Ur game.

        Args:
            agent_parameters: Optional, he parameters of the agent's value function network.
                            defaults to None, meaning `best_parameters` are chosen.
            search_plies: Optional, how many ply to search for, 1 or 2, defaults to 2.
        """
        self.game = Ur()
        self.agent = TDUr()
        if agent_parameters is not None:
            self.agent.set_params(agent_parameters)
        else:
            self.agent.set_best_params()
        self.search_plies = search_plies

        self.human = 0

        self.out = ipyw.Output()

        self.cell_width = 88
        self.cell_height = 0.4 * self.cell_width
        self.style = {'red': 'rgb(221,1,0)', 'blue': 'rgb(34,80,149)', 'yellow': 'rgb(250,201,1)',
                      'cell_width': self.cell_width, 'cell_height': self.cell_height}

        self.board = Board(self, self.game, style=self.style)
        self.roll = Roll(self.game, style=self.style)
        self.players = Players(self, style=self.style)
        self.messages = Messages(style=self.style)
        self.options = Options(self, style=self.style)
        self.scores = Scores(style=self.style)
        self.labels = Labels(style=self.style)

        self.game_interface = self._make_interface()

    def _make_interface(self):
        """Return the interface for the interactive game."""
        interface = ipyw.VBox(children=[
            self.labels.grid,
            ipyw.HBox(children=[self.board.grid, self.roll.grid, self.players.grid]),
            ipyw.HBox(children=[self.options.grid, self.messages.grid, self.scores.grid])])
        interface.add_class("box_style")
        return interface

    def play(self):
        """Show the interface to be able to play the game, in a Jupyter/Colab notebook."""
        display(HTML(style_string))
        return self.game_interface

    def play_move(self, button, move, h_display):
        """Play the given move in the game, if legal, and update the interface.

        Args:
            button: Dummy argument necessary for ipywidgets.
            move: The move, in internal coordinates.
            h_display: The height of the button pressed.
        """
        is_legal_move = self._handle_move_errors(move, h_display)
        if is_legal_move:
            self._play_and_update(move)

    def play_pass(self, button):
        """Pass, if legal, and update the interface.

        Args:
            button: Dummy argument necessary for ipywidgets.
        """
        if self._check_turn(is_human=True) and 'pass' in self.game.legal_moves():
            self._play_and_update('pass')
        else:
            self.messages.display_error("You shall not pass!", "You have a legal move")

    def play_agent(self, button):
        """Let the agent make a move, if it is its turn.

        Args:
            button: Dummy argument necessary for ipywidgets.
        """
        if self._check_turn(is_human=False):
            move = self.agent.policy(self.game, plies=self.search_plies)
            self._play_and_update(move)

    def start_new_game(self, button):
        """Reset the game and start a new one, if the current is finished.

        Args:
            button: Dummy argument necessary for ipywidgets.
        """
        if not self.game.has_finished():
            self.messages.display_error("Game not yet finished!", "Finish this one before starting next.")
        else:
            self.game.reset()
            self.human = (self.human + 1) % 2
            self.update()
            if self.options.auto_play:
                self._do_auto_play()

    def update(self):
        """Update everything in the interface."""
        self.roll.update()
        self.players.update()
        self.messages.clear()
        self.board.update()

    def _play_and_update(self, move):
        """Given a legal move, this plays it on the internal board, and updates all affected buttons.

        Args:
            move: The move to play.
            """
        game = self.game
        self.messages.clear()
        if move == 'pass':
            game.play_move(move)
        else:
            turn = game.turn
            rolled = game.rolled

            game.play_move(move)

            self.board.update_affected_squares(turn, move, rolled)

            if game.has_finished():
                self.scores.update(game.winner, self.human)
                self.messages.display_result(game.winner, self.human)

        self.roll.update()

        with self.out:
            self.out.clear_output(True)
            ipyw.widgets.interaction.show_inline_matplotlib_plots()

        self._do_auto_play()

    def _do_auto_play(self):
        """Let the agent make moves until it is no longer allowed to."""
        while self.options.auto_play and self.game.turn != self.human and not self.game.has_finished():
            time.sleep(1)
            # ipywidget actions have to take the button itself as argument, but none of the actions here depend on it
            # so need a dummy argument here
            dummy_button = 0
            self.play_agent(dummy_button)

    def _handle_move_errors(self, move, h_display):
        """Return boolean indicating the legality of the move, and if illegal print the reason.

        Args:
            move: The move being played.
            h_display: The height of the clicked button.

        Returns:
            Boolean indicating if the move is legal.
        """
        game = self.game

        if not self._check_turn(is_human=True):
            return False

        is_legal = True
        if h_display == 2 * ((self.game.turn + 1) % 2):
            is_legal = False
            self.messages.display_error("That's not your own row!", "See on the right which color you are.")
        elif move not in game.legal_moves():
            is_legal = False
            if game.board[game.turn, move] == 0:
                reason = "No own stone present to move."
            elif move + game.rolled > self.game.finish:
                reason = "This would move the stone off the board."
            elif game.board[game.turn, move + game.rolled] == 1:
                reason = "Your own stone is in the way."
            elif game.board[game.other(), move + game.rolled] == 1 and move + game.rolled == self.game.safe_square:
                reason = "Cannot capture opponent on rosette."
            elif game.rolled == 0:
                reason = 'Rolled 0, can only pass (click "you")'
            else:
                reason = "Not sure why..?"
            self.messages.display_error("Not a legal move!", reason)

        return is_legal

    def _check_turn(self, is_human):
        """Return True if it is the turn of the player indicated by `is_human`, and the game has not finished.

        Args:
            is_human: `True` to ask if it is the human's turn, `False` otherwise.

        Returns:
            Boolean.
        """
        if self.game.has_finished():
            self.messages.display_error("The game has finished!", "Click New Game to start a new one.")
            return False
        if is_human and self.game.turn != self.human:
            self.messages.display_error("It's TD-Ur's turn, not yours!", "Click its name to let it make a move.")
            return False
        if not is_human and self.game.turn == self.human:
            self.messages.display_error("It's your turn, not TD-Ur's!", "Click the square you want to move.")
            return False

        return True


class Board:
    def __init__(self, interface, game, style, cells_high=3, cells_wide=8):
        self.interface = interface
        self.game = game
        self.style = style

        self.board_height = cells_high
        self.board_width = cells_wide

        self.w_display_start = self.game.transform_to_display(0, game.start)[1]
        self.w_display_finish = self.game.transform_to_display(0, game.finish)[1]

        self.grid = make_empty_grid(style, cells_high, cells_wide, square=True)
        self._init_grid()

    def _init_grid(self):
        game = self.game
        grid = self.grid
        board_width = self.board_width
        board_height = self.board_height

        # put play move buttons on all squares
        for h_display in range(board_height):
            for w_display in range(board_width):
                grid[h_display, w_display] = self._make_play_move_button(h_display, w_display)

        # add stone count on start and finish squares
        for i in [0, 2]:
            for j in [self.w_display_start, self.w_display_finish]:
                grid[i, j].description = f'{game.board[self.game.transform_to_internal(i, j)]}'
                grid[i, j].style = {'button_color': self.style['yellow'], 'font_size': '20'}
                if i == 0:
                    grid[i, j].add_class("red_font")
                else:
                    grid[i, j].add_class("blue_font")

        # disable those on the finish
        for h_display in [0, 2]:
            grid[h_display, self.w_display_finish]._click_handlers.callbacks = []

        # add rosettes
        for h in [0, 1]:
            for w in game.rosettes:
                grid[self.game.transform_to_display(h, w)].add_class("rosette_style")

    def update(self):
        """Update all buttons in the board grid."""
        for h_display in range(self.board_height):
            for w_display in range(self.board_width):
                self._update_square(h_display, w_display)

    def update_affected_squares(self, turn, move, rolled):
        """Update only those buttons that could have been affected by the move.

        Called after the internal board has been updated.

        Args:
            turn: Integer, 0 or 1, indicating which player made the move.
            move: The move played.
            rolled: The die roll.
        """
        _, w_display_before = self.game.transform_to_display(turn, move)
        _, w_display = self.game.transform_to_display(turn, move + rolled)

        for h_display in range(self.board_height):
            self._update_square(h_display, w_display)
            self._update_square(h_display, w_display_before)
        self._update_starts()

    def _update_starts(self):
        """Update starting squares."""
        for h_display in [0, 2]:
            self._update_square(h_display, self.w_display_start)

    def _update_square(self, h_display, w_display):
        """Update the board square specified.

        Args:
            h_display: The display board height coordinate.
            w_display: The display board width coordinate.
        """
        game = self.game

        button = self.grid[h_display, w_display]

        h, w = button.coords
        board_val = game.board[h, w]

        # for start and finish square, updated displayed number
        if w == game.start or w == game.finish:
            button.description = f'{board_val}'

        # for squares on the board, update color
        else:
            red, blue = self.style['red'], self.style['blue']
            if h_display == 1:
                if game.board[0, w]:
                    color = red
                elif game.board[1, w]:
                    color = blue
                else:
                    color = 'white'
            else:
                if board_val == 0:
                    color = 'white'
                else:
                    color = [red, blue][h_display // 2]

            button.style = {'button_color': color}

    def _make_play_move_button(self, h_display, w_display):
        """Return ipywidgets Button that moves the stone on the button when clicked.

        The button is given its display and internal coordinates as attributes.

        Args:
            h_display: The display board height coordinate of the button.
            w_display: The display board width coordinate of the button.

        Returns:
            An instance of the ipywidgets.Button class.
        """
        h, w = self.game.transform_to_internal(h_display, w_display)
        btn_play = make_button('', 'white', action=partial(self.interface.play_move, move=w, h_display=h_display))
        btn_play.display_coords = (h_display, w_display)
        btn_play.coords = h, w
        return btn_play


class Roll:
    def __init__(self, game, style, cells_high=3, cells_wide=1):
        self.game = game
        self.style = style
        self.grid = make_empty_grid(style, cells_high, cells_wide, square=True)
        self._init_grid()

    def _init_grid(self):
        self.grid[0, 0] = make_button(f'{self.game.rolled}', self.style['yellow'], css_style="red_font")
        self.grid[2, 0] = make_button('', self.style['yellow'], css_style="blue_font")

    def update(self):
        """Update roll and turn information."""
        self.grid[2 * self.game.other(), 0].description = ' '
        self.grid[2 * self.game.turn, 0].description = f'{self.game.rolled}'


class Players:
    def __init__(self, interface, style, cells_high=3, cells_wide=2):
        self.interface = interface
        self.style = style
        self.grid = make_empty_grid(style, cells_high, cells_wide, square=True)
        self._init_grid()

    def _init_grid(self):
        self.grid[0, :] = make_button('You', self.style['yellow'], css_style="red_font_small",
                                      action=self.interface.play_pass)
        self.grid[2, :] = make_button('TD-Ur', self.style['yellow'], css_style="blue_font_small",
                                      action=self.interface.play_agent)

    def update(self):
        """Update player information."""
        human = self.interface.human
        ai = (human + 1) % 2

        self.grid[2 * human, :] = make_button('You', self.style['yellow'], action=self.interface.play_pass)
        self.grid[2 * ai, :] = make_button('TD-Ur', self.style['yellow'], action=self.interface.play_agent)
        self.grid[0, :].add_class("red_font_small")
        self.grid[2, :].add_class("blue_font_small")


class Options:
    def __init__(self, interface, style, cells_high=4, cells_wide=3):
        self.style = style
        self.interface = interface
        self.new_game_index = 1
        self.auto_play_index = 3

        self.auto_play = False

        self.grid = make_empty_grid(style, cells_high, cells_wide)
        self._init_grid()

    def _init_grid(self):
        self.grid[self.new_game_index, :] = make_button('New Game', self.style['yellow'],
                                                        css_style="button_font", action=self.interface.start_new_game)
        self.grid[self.auto_play_index, :] = make_button('auto-play: off', self.style['yellow'],
                                                         css_style="button_font", action=self.toggle_auto_play)

    def toggle_auto_play(self, button):
        """Toggle the auto-play option.

        Args:
            button: Dummy argment necessary for ipywidgets.
        """
        if self.auto_play:
            self.auto_play = False
            self.grid[self.auto_play_index, :].description = 'auto-play: off'
        else:
            self.auto_play = True
            self.grid[self.auto_play_index, :].description = 'auto-play: on'
            self.interface._do_auto_play()


class Scores:
    def __init__(self, style, cells_high=4, cells_wide=3):
        self.header_h = 1
        self.player_h = 2
        self.agent_h = 3

        self.values = [0, 0]

        self.grid = make_empty_grid(style, cells_high, cells_wide)
        self._init_grid()

    def _init_grid(self):
        self.grid[self.header_h, :] = make_label('scores')
        self.grid[self.player_h, :-1] = make_label('You')
        self.grid[self.agent_h, :-1] = make_label('TD-Ur')
        self.grid[self.player_h, -1] = make_label('0')
        self.grid[self.agent_h, -1] = make_label('0')

    def update(self, winner, human):
        """Update the score information."""
        if winner == human:
            self.values[0] += 1
        else:
            self.values[1] += 1

        self.grid[self.player_h, -1].value = f'{self.values[0]}'
        self.grid[self.agent_h, -1].value = f'{self.values[1]}'


class Messages:
    def __init__(self, style, cells_high=4, cells_wide=5):
        self.main_index = 1
        self.detail_index = 2
        self.grid = make_empty_grid(style, cells_high, cells_wide)
        self._init_grid()

    def _init_grid(self):
        self.grid[self.main_index, :] = make_label(' ', css_style="message_style")
        self.grid[self.detail_index, :] = make_label(' ', css_style="detailed_message_style")

    def display_error(self, message, details):
        """Display the given error message.

        Args:
            message: The main error.
            details: Possible further explanation.
        """
        self.grid[self.main_index, :].value = message
        self.grid[self.detail_index, :].value = details

    def clear(self):
        """Clear any error messages."""
        self.grid[self.main_index, :].value = ''
        self.grid[self.detail_index, :].value = ''

    def display_result(self, winner, human):
        """Display the winner of the game."""
        if winner == human:
            self.display_error("You won this game!", "Congratulations!")
        else:
            self.display_error("You lost this game!", "Better luck next time!")


class Labels:
    def __init__(self, style, cells_high=1, cells_wide=11):
        self.start_index = 4
        self.finish_index = 5
        self.roll_index = 8

        self.grid = make_empty_grid(style, cells_high, cells_wide)
        self._init_grid()

    def _init_grid(self):
        self.grid[0, 9::] = make_label('players')

        self.grid[0, self.start_index] = make_label('start')
        self.grid[0, self.finish_index] = make_label('finish')
        self.grid[0, self.roll_index] = make_label('roll')


def make_empty_grid(style, cells_high, cells_wide, square=False):
    """Return an ipywidgets grid.

    Args:
        style: Dictionary containing keys `'cell_width'` and `'cell_height'`.
        cells_high: How many rows of cells to create.
        cells_wide: How many columns of cells to create.
        square: Optional, will set `'cell_height'` to `'cell_width'` if True, defaults to False.

    Returns:
        An instance of ipywidgets.GridspecLayout.
    """
    cell_width = style['cell_width']
    cell_height = cell_width if square else style['cell_height']
    return ipyw.GridspecLayout(cells_high, cells_wide,
                               width=f'{cells_wide * cell_width}px',
                               height=f'{cells_high * cell_height}px')


def make_button(description, color, css_style="", action=None):
    """Return an ipywidgets Button.

    Args:
        description: The text on the button.
        color: The color of the button (rgb string).
        css_style: The css style to use for the button.
        action: The action for the button, a function.

    Returns:
        An instance of ipywidgets.Button.
    """
    button = ipyw.Button(description=description, style={'button_color': color},
                         layout=ipyw.Layout(height='auto', width='auto'))
    if css_style:
        button.add_class(css_style)
    if action:
        button.on_click(action)
    return button


def make_label(description, css_style="label_font"):
    """Return an ipywidgets Label.

    Args:
        description: The text to put in the label.
        css_style: The css style to use.

    Returns:
        An instance of ipywidgets.Label.
    """
    label = ipyw.Label(description, layout=ipyw.Layout(height='auto', width='auto'))
    if css_style:
        label.add_class(css_style)
    return label
