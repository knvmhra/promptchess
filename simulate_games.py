import chess
from abc import ABC, abstractmethod
from typing import Tuple, Dict
import random
import dspy

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
