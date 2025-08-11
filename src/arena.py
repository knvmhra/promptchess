from typing import Tuple, List, Dict
from model_player import ModelPlayer, ModelConfig

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
