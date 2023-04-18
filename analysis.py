from qlearning import FeatExtractor, Locator, TennisState
from typing import Any, Dict, List
import matplotlib.image as image
import csv
import numpy as np


def saveGameRecording(stateSequence: List[TennisState], fileNamePrefix: str,
                      actionsAndProbs: Dict[Any, float], extractor: FeatExtractor):
    """Save a sequence of images for the game, along with a picked list holding
    the features for each.
    """
    for index, state in enumerate(stateSequence):
        obsPlus = state.obs.copy()
        if state.opponentClosestApproach < 25:
            x, y = (state.opponentBallHit + 1.0) / 2.0
            obsPlus[int(y * Locator.screen_shape[0]), int(x * Locator.screen_shape[1]), :] = (255, 0, 0)
        if not np.isnan(state.ballLastKnownPos[0]):
            if not np.isnan(state.expectedBallVelocity[0]):
                ballT1 = state.ballLastKnownPos + state.expectedBallVelocity
                x, y = (ballT1 + 1.0) / 2.0
                obsPlus[int(y * Locator.screen_shape[0]), int(x * Locator.screen_shape[1]), :] = (0, 0, 255)
            x, y = (state.ballLastKnownPos + 1.0) / 2.0
            obsPlus[int(y * Locator.screen_shape[0]), int(x * Locator.screen_shape[1]), :] = (0, 0, 255)
        image.imsave(f'{fileNamePrefix}_{str(index).zfill(4)}.png', obsPlus)


def analyzeGame(stateSequence: List[TennisState], reward: float) -> List[float]:
    """Determines how the agent played in one game (run each time a point is scored).
    
    Return fields:
    index 0 = side of agent (0 for top, 1 for bottom)
    index 1 = whether agent served
    index 2 = number of times hitting the ball
    index 3 = number of times the opponent hit the ball
    index 4 = whether agent won
    """
    firstState = stateSequence[0]
    # 1 = bottom, -1 = top
    playerSide = np.sign(firstState.player.y)
    opponentSide = -playerSide
    didServe = None
    lastBallSide = np.nan
    playerHits = 0
    opponentHits = 0
    playerWon = 1 if (reward > 0) else 0
    for state in stateSequence:
        if not np.isnan(state.ball.x):
            # 1 = bottom, -1 = top
            ballSide = np.sign(state.ball.y)
            if didServe is None:
                # set only on first apperance of the ball
                didServe = 1 if (ballSide == playerSide) else 0
            if lastBallSide != ballSide:
                if lastBallSide == playerSide:
                    # the ball switched sides to the opponent's side, so our
                    # player definitely hit it
                    playerHits += 1
                elif lastBallSide == opponentSide:
                    opponentHits += 1
                # if the last ball side was unknown (np.nan), no hits are
                # registered
                lastBallSide = ballSide
    return [max(0, playerSide), didServe, playerHits, opponentHits, playerWon]


def countPreviousScores() -> int:
    try:
        with open('score.csv', 'r') as scoreFile:
            return len(scoreFile.readlines()) - 1
    except FileNotFoundError:
        # already has column headers
        return 0


def saveScore(score_rows: List[List[float]]) -> None:
    try:
        with open('score.csv', 'x') as scoreFile:
            # if open succeeds with mode 'x', the file was just created and is empty
            scoreFile.write(','.join(['game', 'agent_side', 'agent_served',
                                      'agent_hit', 'opponent_hit', 'won']))
            scoreFile.write('\n')
    except FileExistsError:
        # already has column headers
        pass
    with open('score.csv', 'a+', newline='') as scoreFile:
        csvwriter = csv.writer(scoreFile)
        rows = []
        for index, each_row in enumerate(score_rows):
            row = [index, *each_row]
            rows.append(row)
        csvwriter.writerows(rows)
