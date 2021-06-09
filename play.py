from TDUr import TDUr
from game import Ur
import ipywidgets as ipyw
from functools import partial
import time

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

color_yellow = 'rgb(250,201,1)'
color_blue = 'rgb(34,80,149)'
color_red = 'rgb(221,1,0)'


class InteractiveGame:

    def __init__(self, agent_parameters, search_plies=2):
        self.game = Ur()
        self.agent = TDUr(hidden_units=agent_parameters[0][0].shape[0])
        self.agent.set_params(agent_parameters)
        self.search_plies = search_plies

        self.human = 0

        self.out = ipyw.Output()

        self.square_size = 88
        self.cell_height = 0.4 * self.square_size
        self.board_width = 8
        self.grid = self.create_interactive_board()
        self.message_grid = self.create_message_grid()
        self.options = Options(self, cell_height=self.cell_height, cell_width=self.square_size, cells_high=4, cells_wide=3)
        self.scores = Scores(cell_height=self.cell_height, cell_width=1.5 * self.square_size, cells_high=4, cells_wide=2)
        self.labels = Labels(cell_height=self.cell_height, cell_width=self.square_size, cells_high=1, cells_wide=11)
        self.game_interface = self.create_interface()

    def create_interactive_board(self, grid_height=3, grid_width=11):
        game = self.game
        board_width = self.board_width

        grid = ipyw.GridspecLayout(grid_height, grid_width,
                                   height=f'{grid_height * self.square_size}px',
                                   width=f'{grid_width * self.square_size}px')

        # put play move buttons on all squares
        for h_display in range(grid_height):
            for w_display in range(board_width):
                grid[h_display, w_display] = self.play_move_button(h_display, w_display)

        # add stone count on start and finish squares
        w_display_start, w_display_finish = 4, 5
        for i in [0, 2]:
            for j in [w_display_start, w_display_finish]:
                grid[i, j].description = f'{game.board[self.transform_to_internal(i, j)]}'
                grid[i, j].style = {'button_color': color_yellow, 'font_size': '20'}
                if i == 0:
                    grid[i, j].add_class("red_font")
                else:
                    grid[i, j].add_class("blue_font")

        # disable those on the finish
        for i in [0, 2]:
            grid[i, w_display_finish]._click_handlers.callbacks = []

        # add rosettes
        for i in [0, 1]:
            for j in game.rosettes:
                grid[self.transform_to_display(i, j)].add_class("rosette_style")

        # add roll and turn info
        grid[0, board_width] = make_button(f'{game.rolled}', color_yellow, css_style="red_font")
        grid[2, board_width] = make_button('', color_yellow, css_style="blue_font")

        # add player info
        grid[0, board_width + 1:] = make_button('You', color_yellow, css_style="red_font_small",
                                                     action=self.play_pass)
        grid[2, board_width + 1:] = make_button('TD-Ur', color_yellow, css_style="blue_font_small",
                                                     action=self.play_agent)

        return grid

    def create_message_grid(self, message_width=5, message_height=4):
        message_grid = ipyw.GridspecLayout(3, 1, width=f'{5 * self.square_size}px',
                                           height=f'{4 * 0.4 * self.square_size}px')
        message_grid[1, 0] = make_label(' ', css_style="message_style")
        message_grid[2, 0] = make_label(' ', css_style="detailed_message_style")

        return message_grid

    def create_interface(self):
        interface = ipyw.VBox(children=[
            self.labels.grid,
            self.grid,
            ipyw.HBox(children=[self.options.grid, self.message_grid, self.scores.grid])])
        interface.add_class("box_style")
        return interface

    def play_move(self, button, move):
        is_legal_move = self.handle_move_errors(move)
        if is_legal_move:
            self.play_and_update(move)

    def play_pass(self, button):
        if 'pass' in self.game.legal_moves():
            self.play_and_update('pass')
        else:
            self.display_error("You shall not pass!", "You have a legal move")

    def play_agent(self, button):
        if self.game.has_finished():
            self.display_error("The game has finished!", "Click New Game to start a new one.")
        elif self.game.turn == self.human:
            self.display_error("It's your turn, not TD-Ur's!", "Click the square you want to move.")
        else:
            move = self.agent.policy(self.game, plies=self.search_plies)
            self.play_and_update(move)

    def play_and_update(self, move):
        """Given a legal move, this plays it on the internal board, and updates all affected buttons."""
        game = self.game
        self.remove_error()
        if move == 'pass':
            game.play_move(move)
        else:
            turn = game.turn
            rolled = game.rolled
            _, w_display_before = self.transform_to_display(turn, move)
            _, w_display = self.transform_to_display(turn, move + rolled)
            game.play_move(move)

            for h_display in range(3):
                self.update_board_square(h_display, w_display, turn)
                self.update_board_square(h_display, w_display_before, turn)
            self.update_starts()

            if game.has_finished():
                self.scores.update(game.winner, self.human)
                self.print_result()

        self.update_turn()

        with self.out:
            self.out.clear_output(True)
            ipyw.widgets.interaction.show_inline_matplotlib_plots()

        self.do_auto_play()

    def update_turn(self):
        self.grid[2 * self.game.other(), 8].description = ' '
        self.grid[2 * self.game.turn, 8].description = f'{self.game.rolled}'

    def update_starts(self):
        for h_display in [0, 2]:
            self.update_board_square(h_display, 4, 0)

    def update_players(self):
        human = self.human
        ai = (human + 1) % 2
        w_player = self.game.display_width + 1

        self.grid[2 * human, w_player:] = make_button('You', color_yellow, action=self.play_pass)
        self.grid[2 * ai, w_player:] = make_button('TD-Ur', color_yellow, action=self.play_agent)
        self.grid[0, w_player:].add_class("red_font_small")
        self.grid[2, w_player:].add_class("blue_font_small")

    def update_board_square(self, h_display, w_display, turn):
        """Updates button color/number, after the internal board has been updated, but with the turn argument
        being the turn before it was updated, so the identity of the player who last moved.
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
            if self.game.mid_start <= w < self.game.mid_ended:
                if game.board[0, w]:
                    color = color_red
                elif game.board[1, w]:
                    color = color_blue
                else:
                    color = 'white'
            else:
                if board_val == 0:
                    color = 'white'
                elif h_display == 1:
                    color = [color_red, color_blue][turn]
                else:
                    color = [color_red, color_blue][h]

            button.style = {'button_color': color}

    def handle_move_errors(self, move):
        game = self.game
        is_legal = True

        if game.has_finished():
            is_legal = False
            self.display_error("The game has finished!", "Click New Game to start a new one.")
        elif game.turn != self.human:
            is_legal = False
            self.display_error("It's TD-Ur's turn!", "Click its name to let it make a move.")
        elif move not in game.legal_moves():
            is_legal = False
            if game.board[game.turn, move] == 0:
                reason = "No stone present to move."
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
            self.display_error("Not a legal move!", reason)

        return is_legal

    def display_error(self, message, details):
        self.message_grid[1, 0].value = message
        self.message_grid[2, 0].value = details

    def remove_error(self):
        self.message_grid[1, 0].value = ''
        self.message_grid[2, 0].value = ''

    def print_result(self):
        if self.game.winner == self.human:
            self.display_error("You won this game!", "Congratulations!")
        else:
            self.display_error("You lost this game!", "Better luck next time!")

    def start_new_game(self, button):
        if self.game.winner == -1:
            self.display_error("Game not yet finished!", "Finish this one before starting next.")
        else:
            self.game.reset()
            self.human = (self.human + 1) % 2
            self.update_all_buttons()
            if self.options.auto_play:
                self.do_auto_play()

    def update_all_buttons(self):
        self.update_turn()
        self.update_players()
        self.remove_error()
        for i in range(3):
            for j in range(self.game.display_width):
                self.update_board_square(i, j, 0)

    def transform_to_display(self, i, j):
        """Go from internal board representation coordinates (2x16) to displayed board (3x8) coordinates"""
        if j < self.game.mid_start:
            j_display = self.game.mid_start - 1 - j
            i_display = 2 * i
        elif j >= self.game.mid_ended:
            j_display = (self.game.display_width - 1) - (j - self.game.mid_ended)
            i_display = 2 * i
        else:
            j_display = j - self.game.mid_start
            i_display = 1

        return i_display, j_display

    def transform_to_internal(self, i, j):
        """Go from to displayed board (3x8) coordinates to internal board representation coordinates (2x16)"""
        if i == 1:  # middle row
            i_internal = self.game.turn
            j_internal = j + self.game.mid_start
        else:
            i_internal = i // 2
            if j < self.game.mid_start:
                j_internal = self.game.mid_start - 1 - j
            else:
                j_internal = self.game.mid_ended - (j - (self.game.display_width - 1))

        return i_internal, j_internal

    def do_auto_play(self):
        while self.options.auto_play and self.game.turn != self.human and not self.game.has_finished():
            time.sleep(1)
            self.play_agent(self.grid[2, -1])

    def play_move_button(self, h_display, w_display):
        h, w = self.transform_to_internal(h_display, w_display)
        btn_play = make_button('', 'white', action=partial(self.play_move, move=w))
        btn_play.display_coords = (h_display, w_display)
        btn_play.coords = h, w
        return btn_play


class Options:
    def __init__(self, interface, cell_height, cell_width, cells_high, cells_wide):
        self.new_game_index = 1
        self.auto_play_index = 3
        self.interface = interface

        self.auto_play = False

        self.grid = self.make_grid(interface, cell_height, cell_width, cells_high, cells_wide)

    def make_grid(self, interface, cell_height, cell_width, cells_high, cells_wide):
        grid = make_empty_grid(cell_height, cell_width, cells_high, cells_wide)

        grid[self.new_game_index, :] = make_button('New Game', color_yellow, css_style="button_font",
                                                   action=interface.start_new_game)
        grid[self.auto_play_index, :] = make_button('auto-play: off', color_yellow, css_style="button_font",
                                                    action=self.toggle_auto_play)

        return grid

    def toggle_auto_play(self, button):
        if self.auto_play:
            self.auto_play = False
            self.grid[self.auto_play_index, :].description = 'auto-play: off'
        else:
            self.auto_play = True
            self.grid[self.auto_play_index, :].description = 'auto-play: on'
            self.interface.do_auto_play()


class Scores:
    def __init__(self, cell_height, cell_width, cells_high, cells_wide):
        self.header_indices = 1, -1
        self.player_name_indices = 2, -2
        self.agent_name_indices = 3, -2
        self.player_score_indices = 2, -1
        self.agent_score_indices = 3, -1

        self.values = [0, 0]

        self.grid = self.make_grid(cell_height, cell_width, cells_high, cells_wide)

    def make_grid(self, cell_height, cell_width, cells_high, cells_wide):
        grid = make_empty_grid(cell_height, cell_width, cells_high, cells_wide)

        grid[self.header_indices] = make_label('scores')
        grid[self.player_name_indices] = make_label('You')
        grid[self.agent_name_indices] = make_label('TD-Ur')
        grid[self.player_score_indices] = make_label('0')
        grid[self.agent_score_indices] = make_label('0')

        return grid

    def update(self, winner, human):
        if winner == human:
            self.values[0] += 1
        else:
            self.values[1] += 1

        self.grid[self.player_score_indices].value = f'{self.values[0]}'
        self.grid[self.player_score_indices].value = f'{self.values[1]}'


class Labels:
    def __init__(self, cell_height, cell_width, cells_high=1, cells_wide=11):
        self.start_index = 4
        self.finish_index = 5
        self.roll_index = 8

        self.grid = self.make_grid(cell_height, cell_width, cells_high, cells_wide)

    def make_grid(self, cell_height, cell_width, cells_high, cells_wide):
        grid = make_empty_grid(cell_height, cell_width, cells_high, cells_wide)

        grid[0, 9::] = make_label('players')

        grid[0, self.start_index] = make_label('start')
        grid[0, self.finish_index] = make_label('finish')
        grid[0, self.roll_index] = make_label('roll')

        return grid


def make_empty_grid(cell_height, cell_width, cells_high, cells_wide):
    return ipyw.GridspecLayout(cells_high, cells_wide,
                               width=f'{cells_wide * cell_width}px',
                               height=f'{cells_high * cell_height}px')


def make_button(description, color, css_style="", action=None):
    button = ipyw.Button(description=description, style={'button_color': color},
                         layout=ipyw.Layout(height='auto', width='auto'))
    if css_style:
        button.add_class(css_style)
    if action:
        button.on_click(action)
    return button


def make_label(description, css_style="label_font"):
    label = ipyw.Label(description, layout=ipyw.Layout(height='auto', width='auto'))
    if css_style:
        label.add_class(css_style)
    return label
