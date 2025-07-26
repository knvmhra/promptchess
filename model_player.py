import chess
import dspy
import pickle


class ChessPredictor(dspy.Signature):
    board: str = dspy.InputField(
        desc= "Chess position with ranks 1-8 (bottom to top) and files a-h (left to right).\
        Uppercase pieces are white, Lowercase pieces are black. Castling rights, available en passant, and turn are provided.\
        It is always your turn to play.")
    move: str = dspy.OutputField(desc= "Best move in standard algebraic notation (SAN).")


def metric(example: dspy.Example, prediction: dspy.Prediction) -> bool:
    return example.move == prediction.move


if __name__ == '__main__':
    with open('lichess_data.pkl', 'rb') as f:
        data = pickle.load(f)

    lm = dspy.LM(model= 'gpt-4.1')
    dspy.configure(lm=lm)

    examples = [dspy.Example(board= pos, move= move) for pos, move, in data[80]]

    predictor = dspy.ChainOfThought(ChessPredictor)

    optimizer = dspy.BootstrapFewShot(metric = metric, max_bootstrapped_demos= 90)

    predictor = optimizer.compile(predictor, trainset= examples)

    predictor.save('chess_predictor_optimized.json')
