from typing import Tuple, List, Dict, Optional
from model_player import ModelPlayer, ModelConfig
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import chess
import chess.pgn
import random
import json
from itertools import combinations


class EloCalculator:
    @staticmethod
    def calculate_rating_change(player_rating: float, opponent_rating: float, actual_score: float, k_factor: int = 32) -> float:
        expected_score = 1 / (1 + 10 ** ((opponent_rating - player_rating) / 400))
        return k_factor * (actual_score - expected_score)

    @staticmethod
    def update_ratings(white_rating: float, black_rating: float, result_score: float, k_factor: int = 32) -> Tuple[float, float]:

        white_score = result_score
        black_score = 1.0 - result_score

        white_change = EloCalculator.calculate_rating_change(
            white_rating, black_rating, white_score, k_factor)
        black_change = EloCalculator.calculate_rating_change(
            black_rating, white_rating, black_score, k_factor)

        return white_rating + white_change, black_rating + black_change

@dataclass
class Game:
    white: ModelConfig
    black: ModelConfig
    moves: List[str] = field(default_factory=list)
    reasonings: List[str] = field(default_factory=list)
    result: float = 0.5  # 1.0 = white win, 0.0 = black win, 0.5 = draw

    def to_pgn(self) -> chess.pgn.Game:
        pgn = chess.pgn.Game()
        pgn.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        pgn.headers["White"] = self.white.label
        pgn.headers["Black"] = self.black.label
        pgn.headers["WhiteElo"] = str(int(self.white.elo))
        pgn.headers["BlackElo"] = str(int(self.black.elo))
        pgn.headers["Result"] = ["0-1", "1/2-1/2", "1-0"][int(self.result * 2)]

        board = chess.Board()
        node = pgn
        for san, reasoning in zip(self.moves, self.reasonings):
            move = board.parse_san(san)
            node = node.add_variation(move)
            if reasoning and reasoning != 'No reasoning':
                node.comment = reasoning
            board.push(move)

        return pgn

class League:
    def __init__(self, players: List[ModelConfig], max_retries: int = 3, stringifier = None):
        self.players = players
        self.max_retries = max_retries
        self.stringifier = stringifier or (lambda b: f"{b}\nLegal: {', '.join(b.san(m) for m in b.legal_moves)}")
        self.games: List[Game] = []

    def play_game(self, white: ModelConfig, black: ModelConfig) -> Game:
        board = chess.Board()
        game = Game(white, black)

        white_player = ModelPlayer(white, self.max_retries, self.stringifier)
        black_player = ModelPlayer(black, self.max_retries, self.stringifier)

        move_history = ""
        while not board.is_game_over():
            player = white_player if board.turn else black_player
            move, reasoning = player.get_move(board, move_history)
            san = board.san(move)
            game.moves.append(san)
            game.reasonings.append(reasoning)

            move_num = (len(game.moves) + 1) // 2
            move_history += f"{move_num}. {san} " if board.turn else f"{san} "

            board.push(move)

        if board.is_checkmate():
            game.result = 0.0 if board.turn else 1.0

        return game

    def save_configs(self, path: Path = Path("model_configs.json")):
        configs = []
        for player in self.players:
            configs.append({
                'provider': player.provider.value,
                'api_name': player.api_name,
                'label': player.label,
                'is_reasoning': player.is_reasoning,
                'is_COT': player.is_COT,
                'instructions': player.instructions,
                'elo': player.elo
            })
        with open(path, 'w') as f:
            json.dump(configs, f, indent=2)

    def run(self):
        for p1, p2 in combinations(self.players, 2):
            print(f"\n{p1.label} vs {p2.label}")

            if random.random() < 0.5:
                white1, black1 = p1, p2
            else:
                white1, black1 = p2, p1

            game1 = self.play_game(white1, black1)
            self.games.append(game1)
            white1.elo, black1.elo = EloCalculator.update_ratings(
                white1.elo, black1.elo, game1.result
            )
            print(f"  {white1.label}-{black1.label}: {['0-1', '1/2-1/2', '1-0'][int(game1.result * 2)]}")

            game2 = self.play_game(black1, white1)
            self.games.append(game2)
            black1.elo, white1.elo = EloCalculator.update_ratings(
                black1.elo, white1.elo, game2.result
            )
            print(f"  {black1.label}-{white1.label}: {['0-1', '1/2-1/2', '1-0'][int(game2.result * 2)]}")

        print("\nFinal ELO Rankings:")
        for player in sorted(self.players, key=lambda p: -p.elo):
            print(f"{player.label}: {player.elo:.0f}")

        self.save_configs()

    def export_pgns(self, dir: Path = Path("pgns")):
        dir.mkdir(exist_ok=True)
        for i, game in enumerate(self.games):
            with open(dir / f"game_{i+1}.pgn", 'w') as f:
                print(game.to_pgn(), file=f)
