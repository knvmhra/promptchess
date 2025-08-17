from typing import Tuple, List, Callable, Set, Dict, Optional
from model_player import ModelPlayer, ModelConfig
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import chess
import chess.pgn
import random
import json
from itertools import combinations

from models import ProviderType


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

    def to_dict(self) -> dict:
            return {
                'white_label': self.white.label,
                'black_label': self.black.label,
                'moves': self.moves,
                'reasonings': self.reasonings,
                'result': self.result
            }

    @classmethod
    def from_dict(cls, data: dict, player_map: dict) -> 'Game':
        return cls(
            white=player_map[data['white_label']],
            black=player_map[data['black_label']],
            moves=data['moves'],
            reasonings=data['reasonings'],
            result=data['result']
        )

class League:
    def __init__(self, players: List[ModelConfig], max_retries: int = 3, stringifier = None):
        self.players = players
        self.max_retries = max_retries
        self.stringifier = stringifier or (lambda b: f"{b}\nLegal: {', '.join(b.san(m) for m in b.legal_moves)}")
        self.games: List[Game] = []
        self.completed_games: Set[Tuple[str, str]] = set()

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
            print(san + '\n')

        if board.is_checkmate():
            game.result = 0.0 if board.turn else 1.0

        return game

    def run(self):
        for p1, p2 in combinations(self.players, 2):
            print(f"\n{p1.label} vs {p2.label}")

            if random.random() < 0.5:
                white1, black1 = p1, p2
            else:
                white1, black1 = p2, p1

            if (white1.label, black1.label) not in self.completed_games:
                game1 = self.play_game(white1, black1)
                self.games.append(game1)
                white1.elo, black1.elo = EloCalculator.update_ratings(
                    white1.elo, black1.elo, game1.result
                )
                print(f"  {white1.label} vs. {black1.label}: {['0-1', '1/2-1/2', '1-0'][int(game1.result * 2)]}")
                self.completed_games.add((white1.label, black1.label))
                self.save_state()

            if (black1.label, white1.label) not in self.completed_games:
                game2 = self.play_game(black1, white1)
                self.games.append(game2)
                black1.elo, white1.elo = EloCalculator.update_ratings(
                    black1.elo, white1.elo, game2.result
                )
                print(f"  {black1.label} vs. {white1.label}: {['0-1', '1/2-1/2', '1-0'][int(game2.result * 2)]}")
                self.completed_games.add((black1.label, white1.label))
                self.save_state()

        print("\nFinal ELO Rankings:")
        for player in sorted(self.players, key=lambda p: -p.elo):
            print(f"{player.label}: {player.elo:.0f}")

    def save_configs(self, path: Path = Path("model_configs.json")):
        with open(path, 'w') as f:
            json.dump([p.to_dict() for p in self.players], f, indent=2)

    def save_state(self, path: Path = Path("league_state.json")):
            state = {
                'players': [p.to_dict() for p in self.players],
                'games': [g.to_dict() for g in self.games],
                'completed_games': list(self.completed_games),
                'max_retries' : self.max_retries
            }
            with open(path, 'w') as f:
                json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: Path = Path("league_state.json"), stringifier: Optional[Callable] = None):
        with open(path, 'r') as f:
            state = json.load(f)

        players: List[ModelConfig] = []
        player_map: Dict[str, ModelConfig] = {}
        for cfg in state['players']:
            player = ModelConfig(
                provider=ProviderType(cfg['provider']),
                api_name=cfg['api_name'],
                label=cfg['label'],
                instructions=cfg['instructions'],
                is_reasoning=cfg.get('is_reasoning', False),
                is_COT=cfg.get('is_COT', False),
                elo=cfg.get('elo', 400)
            )
            players.append(player)
            player_map[player.label] = player

        league = cls(players=players, max_retries=state['max_retries'], stringifier=stringifier)

        for game_data in state['games']:
            game = Game.from_dict(game_data, player_map)
            league.games.append(game)

        league.completed_games = {tuple(m) for m in state['completed_games']}

        return league

    def export_latest_pgn(self, path: Path = Path("pgn")):
        path.mkdir(exist_ok= True)
        game = self.games[-1]
        game_idx = len(self.games)
        with open(path / f"game_{game_idx}.pgn", 'w') as f:
            print(game.to_pgn(), file=f)


if __name__ == '__main__':
    stringifier: Callable[[chess.Board], str] = lambda x: x.fen()

    INSTRUCTIONS = 'Analyse the chess position and provide the best move'

    gemini_thinking = ModelConfig(
        provider= ProviderType.GEMINI,
        api_name= 'gemini-2.5-pro',
        label= 'gemini-2.5-pro-thinking',
        is_reasoning= True,
        instructions= INSTRUCTIONS
    )

    o3 = ModelConfig(
        provider= ProviderType.OPENAI,
        api_name='gpt-5',
        label='gpt-5',
        is_reasoning= True,
        instructions=INSTRUCTIONS
    )

    opus = ModelConfig(
        provider = ProviderType.ANTHROPIC,
        api_name= 'claude-opus-4-1-20250805',
        label= 'claude-opus-4.1-thinking',
        instructions= INSTRUCTIONS
    )

    arena = League(
        players= [gemini_thinking, o3, opus],
        stringifier=stringifier
    )

    arena.run()
