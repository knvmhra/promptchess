import random
from models import ProviderType, ModelConfig, build_provider
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable
from chess import Board, Move

class ModelPlayer():
    def __init__(self, config: ModelConfig,  max_retries: int, stringifier: Callable[[Board], str], instructions: str):
        self.max_retries = max_retries
        self.stringifier = stringifier
        self.cfg = config
        self.client = build_provider(config)

    def get_move(self, board: Board, move_history: str) -> Tuple[Move, str]
        failed_attempts: List[str] = []

        for i in range(self.max_retries):
            if failed_attempts:
                ctx = self.stringifier(board) + '\n' + f"\nIllegal moves in this position:\n" + '\n'.join(failed_attempts) + 'Move history: \n' + move_history
            else:
                ctx = self.stringifier(board) + '\n' + 'Move history: \n' + move_history

            move_str, reasoning_str = self.client.call(context= ctx, instructions=self.cfg.instructions)

            try:
                legal_move = board.parse_san(move_str)
                return legal_move, reasoning_str

            except ValueError:
                failed_attempts.append(move_str)

        fallback_move = random.choice(list(board.legal_moves))

        return fallback_move, 'Fallback move, ' + f'Failed attempts: {' '.join(failed_attempts)}'
