import chess
from abc import ABC, abstractmethod
from typing import Tuple, Dict, List
import random
from model_player import ChessPredictor
import dspy
from datagen import stringify_board

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
