import chess
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
import random
from model_player import ChessPredictor
import dspy
from datagen import stringify_board
from openai import OpenAI
import json
from datetime import datetime

class ChessPlayer(ABC):
    @abstractmethod
    def get_move(self, board: chess.Board, reasoning: bool = False) -> chess.Move |Tuple[chess.Move, str]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

class RandomPlayer(ChessPlayer):
    def get_move(self, board: chess.Board, reasoning: bool = False) -> chess.Move | Tuple[chess.Move, str]:
        return random.choice(list(board.legal_moves))

    @property
    def name(self) -> str:
        return "Random"

class DSPyPlayer(ChessPlayer):
    def __init__(self, predictor_path: str, name: str, max_retries: int):
        self.predictor = dspy.ChainOfThought(ChessPredictor)
        self.predictor.load(predictor_path)
        self._name = name
        self.max_retries = max_retries


    def get_move(self, board: chess.Board, reasoning: bool = False) -> chess.Move | Tuple[chess.Move, str]:
        board_str = stringify_board(board)
        result = self.predictor(board_str)
        failed_attempts: List[str] = []

        for i in range(self.max_retries):
            if failed_attempts:
                context = board_str + f"\nIllegal moves in this position:\n" + '\n'.join(failed_attempts)
            else:
                context = board_str
            result = self.predictor(board= context)

            try:
                move = board.parse_san(result.move)
                if reasoning:
                    return move, result.reasoning
                return move
            except ValueError:
                failed_attempts.append(result.move)

        #return random fallback move after max_retries
        fallback_move = random.choice(list(board.legal_moves))
        if reasoning:
            return fallback_move, f"All {self.max_retries} attempts incorrect: {failed_attempts}."
        return fallback_move

    @property
    def name(self) -> str:
        return self._name

class SimpleModelPlayer(ChessPlayer):
    def __init__(self, model: str, instructions: str, max_retries: int):
        self.client = OpenAI()
        self.model = model
        self.instructions = instructions
        self.max_retries = max_retries

    def get_move(self, board: chess.Board, reasoning: bool = False) -> chess.Move | Tuple[chess.Move, str]:
        schema = {
            "type": "object",
            "properties": {
                "chess_move_SAN": {"type": "string"},
                **({"reasoning": {"type": "string"}} if reasoning else {})
            },
            "required": ["chess_move_SAN"] + (["reasoning"] if reasoning else []),
            "additionalProperties": False
        }

        board_str = stringify_board(board)
        failed_attempts = []

        for i in range(self.max_retries):
            if failed_attempts:
                context = board_str + f"\nIllegal moves in this position:\n" + '\n'.join(failed_attempts)
            else:
                context = board_str


            response = self.client.responses.create(
                model = self.model,
                input = context,
                instructions = self.instructions,
                text = {
                    'format' : {
                        'type': 'json_schema',
                        'name': 'chess_player_schema',
                        'schema' : schema,
                        'strict' : True
                    }
                }
            )
            parsed_response = json.loads(response.output_text)

            try:
                move = board.parse_san(parsed_response['chess_move_SAN'])
                if reasoning:
                    return move, parsed_response['reasoning']
                return move
            except ValueError:
                failed_attempts.append(parsed_response['chess_move_SAN'])

        fallback_move = random.choice(list(board.legal_moves))
        if reasoning:
            return fallback_move, f"All {self.max_retries} attempts incorrect: {failed_attempts}."
        return fallback_move

    @property
    def name(self) -> str:
        return self.model

class Arena:
    def __init__(self, max_games: int, player_1: ChessPlayer, player_2: ChessPlayer):
        assert(max_games % 2 == 0)
        self.max_games = max_games
        self.player_1 = player_1
        self.player_2 = player_2
        self.games = []
        self.results = {player_1.name: 0, 'draw': 0, player_2.name: 0}

    def play_game(self, white: ChessPlayer, black: ChessPlayer, reasoning: bool = False):
        board = chess.Board()
        move_history = []

        while not board.is_game_over(claim_draw=True):
            if board.turn:
                result = white.get_move(board, reasoning)
            else:
                result = black.get_move(board, reasoning)

            if isinstance(result, tuple):
                move, rationale = result
            else:
                move = result
                rationale = 'Reasoning is false'

            move_history.append({
                'move': board.san(move),
                'reasoning': rationale
            })

            board.push(move)

        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            raise ValueError('Outcome is None but game is over')

        if outcome.winner is None:
            self.results['draw'] += 1
        elif outcome.winner == chess.WHITE:
            self.results[white.name] += 1
        else:
            self.results[black.name] += 1

        game_info = {
            'white': white.name,
            'black': black.name,
            'result': outcome.winner,
            'move_history': move_history
        }
        self.games.append(game_info)

    def play_match(self, reasoning: bool = True):
        for i in range(self.max_games):
            if i % 2 == 0:
                white, black = self.player_1, self.player_2
            else:
                white, black = self.player_2, self.player_1

            self.play_game(white, black, reasoning)


def export_to_pgn(game_data: Dict, filename: str, event: str = "LLM Arena", site: str = "Local") -> None:
    """
    Export game data to PGN format with reasoning as commentary.

    Args:
        game_data: Dict with keys 'white', 'black', 'result', 'move_history'
        filename: Output PGN filename
        event: Tournament/event name
        site: Location where game was played

    Thanks Claude!
    """

    # Convert result to PGN format
    result_map = {
        "white": "1-0",
        "black": "0-1",
        "draw": "1/2-1/2",
        None: "*"
    }
    result = result_map.get(game_data.get("result"), "*")

    # Generate headers
    headers = [
        f'[Event "{event}"]',
        f'[Site "{site}"]',
        f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]',
        f'[Round "1"]',
        f'[White "{game_data["white"]}"]',
        f'[Black "{game_data["black"]}"]',
        f'[Result "{result}"]',
        ""  # Empty line after headers
    ]

    # Process moves
    move_history = game_data.get("move_history", [])
    pgn_moves = []

    for i, move_data in enumerate(move_history):
        move = move_data["move"]
        reasoning = move_data.get("reasoning", "")

        # Add move number for white moves
        if i % 2 == 0:
            move_num = (i // 2) + 1
            move_text = f"{move_num}. {move}"
        else:
            move_text = move

        # Add reasoning as comment if available and not default
        if reasoning and reasoning != "Reasoning is false":
            move_text += f" {{{reasoning}}}"

        pgn_moves.append(move_text)

    # Add result at the end
    pgn_moves.append(result)

    # Format moves with proper line wrapping (80 chars)
    formatted_moves = format_pgn_moves(pgn_moves)

    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(headers))
        f.write(formatted_moves)
        f.write('\n\n')  # Double newline at end

def format_pgn_moves(moves: List[str], line_length: int = 80) -> str:
    """Format moves with proper line wrapping for PGN."""
    lines = []
    current_line = ""

    for move in moves:
        # Check if adding this move would exceed line length
        if current_line and len(current_line + " " + move) > line_length:
            lines.append(current_line)
            current_line = move
        else:
            if current_line:
                current_line += " " + move
            else:
                current_line = move

    # Add the last line
    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)

def export_games_to_pgn(games_file: str, output_file: str) -> None:
    """
    Export all games from JSON file to a single PGN file.

    Args:
        games_file: Path to JSON file containing games
        output_file: Path to output PGN file
    """
    with open(games_file, 'r') as f:
        games = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, game in enumerate(games, 1):
            # Convert result to PGN format
            result_map = {
                "white": "1-0",
                "black": "0-1",
                "draw": "1/2-1/2",
                None: "*"
            }
            result = result_map.get(game.get("result"), "*")

            # Write headers
            headers = [
                f'[Event "LLM Arena"]',
                f'[Site "Local"]',
                f'[Date "{datetime.now().strftime("%Y.%m.%d")}"]',
                f'[Round "{i}"]',
                f'[White "{game["white"]}"]',
                f'[Black "{game["black"]}"]',
                f'[Result "{result}"]',
                ""
            ]

            f.write('\n'.join(headers))

            # Process moves
            move_history = game.get("move_history", [])
            pgn_moves = []

            for j, move_data in enumerate(move_history):
                move = move_data["move"]
                reasoning = move_data.get("reasoning", "")

                # Add move number for white moves
                if j % 2 == 0:
                    move_num = (j // 2) + 1
                    move_text = f"{move_num}. {move}"
                else:
                    move_text = move

                # Add reasoning as comment if available and not default
                if reasoning and reasoning != "Reasoning is false":
                    move_text += f" {{{reasoning}}}"

                pgn_moves.append(move_text)

            # Add result at the end
            pgn_moves.append(result)

            # Format and write moves
            formatted_moves = format_pgn_moves(pgn_moves)
            f.write(formatted_moves)
            f.write('\n\n')  # Double newline between games
