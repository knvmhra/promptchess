import chess
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
import random
from model_player import ChessPredictor
import dspy
from datagen import stringify_board
from openai import OpenAI, models
import json

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
    def __init__(self, predictor_path: str, name: str, max_retries: int) --> None:
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
