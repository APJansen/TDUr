import random
import numpy as np


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
#  ----- Features -----
# These are some hand made features to test hand-made policies with
#


def is_new(game, move):
    return move == game.start


def is_finished(game, move):
    end = move + game.rolled
    return end == game.finish


def is_rosette(game, move):
    end = move + game.rolled
    return end in game.rosettes


def is_save(game, move):
    # moves to the safe area near finish
    end = move + game.rolled
    return move < game.finish - game.safety_length_end <= end


def is_risk(game, move):
    was_safe = move <= game.safety_length_start or move == game.safe_square
    end = move + game.rolled
    is_not = end > game.safety_length_start and end != game.safe_square
    return was_safe and is_not


def is_capture(game, move):
    end = move + game.rolled
    opponent_present = game.board[game.other(), end] == 1
    capturable = game.safety_length_start < end < game.finish - game.safety_length_end
    return opponent_present and capturable


def is_escape(game, move):
    was_capturable = game.safety_length_start < move < game.finish - game.safety_length_end and \
                     move is not game.safe_square and \
                     not (game.board[game.other(), move - 4:move] == np.zeros(4)).all()
    if not was_capturable:
        return False

    end = move + game.rolled
    is_safe = end >= game.finish - game.safety_length_end or end == game.safe_square or \
              (game.board[game.other(), end - 4:end] == np.zeros(4)).all()

    return was_capturable and is_safe


# compute all above at once, and include the move itself normalized
def all_features(game, move):
    end = move + game.rolled

    features = [move == game.start, end == game.finish, end in game.rosettes,
                move < game.finish - game.safety_length_end <= end]

    is_risk = (move <= game.safety_length_start or move == game.safe_square) and \
              (end > game.safety_length_start and end != game.safe_square)

    opponent_present = game.board[game.other(), end] == 1
    capturable = game.safety_length_start < end < game.finish - game.safety_length_end
    is_capture = opponent_present and capturable

    was_capturable = game.safety_length_start < move < game.finish - game.safety_length_end and \
                     move is not game.safe_square and \
                     not (game.board[game.other(), move - 4:move] == np.zeros(4)).all()
    is_escape = was_capturable and (end >= game.finish - game.safety_length_end or end == game.safe_square or
                                    (game.board[game.other(), end - 4:end] == np.zeros(4)).all())

    features += [is_risk, is_capture, is_escape, end / game.finish]
    return features
