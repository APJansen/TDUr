import random

 #
 #  Policies
 #
def pi_random(game):
    game.play_move(random.choice(game.legal_moves()))


def pi_closest(game):
    # play stone closest to start
    game.play_move(game.legal_moves()[0])


def pi_furthest(game):
    # play stone closest to finish
    game.play_move(game.legal_moves()[-1])


def pi_capture(game):
    # prefer captures, closest to finish
    moves = game.legal_moves()
    if moves == ['pass']:
        game.play_move('pass')
    else:
        for move in reversed(moves):
            if is_capture(game, move):
                game.play_move(move)
                return

        pi_furthest(game)

#
# def pi_save(game):
#     # prefer move to safe finish area
#     moves = game.legal_moves()
#     if moves == ['pass']:
#         game.play_move('pass')
#     else:
#         for move in reversed(moves):

#
#  Features
#
def is_capture(game, move):
    end = move + game.rolled
    opponent_present = game.board[game.other(), end] == 1
    capturable = game.safety_length_start < end < game.finish - game.safety_length_end
    return opponent_present and capturable

#
# def is_save(game, move):
