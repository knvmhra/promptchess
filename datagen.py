import chess.pgn
import chess.engine
import chess
from typing import Tuple, List
import pickle

STOCKFISH_PATH = '/Users/kmwork/stockfish/stockfish-macos-m1-apple-silicon'
LICHESS_GAMES = '/Users/kmwork/lichess_elite_2020-06.pgn'


def generate_examples(engine_path: str, pgn: str, max_games: int = 3) -> List[Tuple[str, str]]:
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

    results: List[Tuple[str, str]] = []
    with open(LICHESS_GAMES) as games:
        for i in range(max_games):
            game = chess.pgn.read_game(games)
            if game is None:
                raise ValueError('None game found')
            board = game.board()
            for move in game.mainline_moves():
                position = stringify_board(board)
                result = engine.play(board, chess.engine.Limit(time= 1))
                if result.move is None:
                    continue
                best_move = board.san(result.move)
                results.append((position, best_move))
                board.push(move)

    engine.quit()
    return results

def save_lichess(results: List[Tuple[str, str]]) -> None:
    with open('lichess_data.pkl', 'wb') as file:
        pickle.dump(results, file)
    return

def stringify_board(board: chess.Board) -> str:
    ascii_board = str(board)

    turn = 'White to move' if board.turn else 'Black to move'

    castling = []
    if board.has_kingside_castling_rights(chess.WHITE): castling.append("K")
    if board.has_queenside_castling_rights(chess.WHITE): castling.append("Q")
    if board.has_kingside_castling_rights(chess.BLACK): castling.append("k")
    if board.has_queenside_castling_rights(chess.BLACK): castling.append("q")
    castling_str = "Castling rights (not availablity): " + ("".join(castling) if castling else "none")

    ep_square = board.ep_square

    ep_str = f'En passant square: {chess.square_name(ep_square) if ep_square else "None"}'

    return f'{ascii_board}\n{turn}\n{castling_str}\n{ep_str}'

if __name__ == '__main__':
    board = chess.Board()
    print(stringify_board(board))
