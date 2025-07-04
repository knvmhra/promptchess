import chess.pgn
import chess.engine
from chess import Board
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
                position = str(board)
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


if __name__ == '__main__':
    save_lichess(generate_examples(STOCKFISH_PATH, LICHESS_GAMES, 1000)
