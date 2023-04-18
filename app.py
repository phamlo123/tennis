import gym
import numpy as np
import sys
from analysis import countPreviousScores, saveGameRecording, analyzeGame, saveScore
from qlearning import (
    ApproximateQAgent, FeatExtractor, Locator, RewardShaping, TennisState, getActionDistribution
)

class TennisExtractor(FeatExtractor):
    F_POSITION_BALL = 0
    F_POSITION_OPPONENT = 2
    F_BALL_DISTANCE = 4
    F_BALL_DISTANCE_T1 = 6
    F_BALL_DISTANCE_T3 = 8
    F_BALL_PLAYER_GRID = 10
    FEATS_PER_ACT_PER_SIDE = 91
    FEATS_PER_ACT = 2 * FEATS_PER_ACT_PER_SIDE

    def __init__(self, actionsAndProbs):
        super().__init__(1 + len(actionsAndProbs) * TennisExtractor.FEATS_PER_ACT)

    def setFeatures(self, result, side: int, action: int, whichFeat: int, ndarray):
        # action -> a group of 16
        #  side -> a group of 8
        #   whichFeat -> a single feature
        idx = 1 + whichFeat + action * self.FEATS_PER_ACT + side * self.FEATS_PER_ACT_PER_SIDE
        setSize = ndarray.shape[0]
        result[idx:idx + setSize] = ndarray

    def getGrid(self, pos):
        """
        Returns a 3x3 Numpy array representing the 9 cells of the screen.

        The values of the array represent the probability that the cell contains the given position.
        They are always 0, 1, or 1/9 (if the location is None or [np.nan, np.nan])
        """
        if pos is None or np.isnan(pos[0]):
            return np.ones((3, 3)) / 9.0
        # gridX, gridY = np.meshgrid([-0.5, 0.0, 0.5], [-0.5, 0.0, 0.5])
        # gridX = np.exp(-10 * np.abs(gridX - pos[0]))
        # gridY = np.exp(-10 * np.abs(gridY - pos[1]))
        # grid = gridX * gridY
        grid = np.zeros((3, 3))
        # real number -1 to 1 --> real number 0 to 3 --> [0, 1, 2]
        gridCoords = np.min([[2, 2], (pos + 1.0) * 1.5], axis=0).astype(int)
        grid[gridCoords[0], gridCoords[1]] = 1.0
        return grid
        

    def getFeatures(self, state: TennisState, action: int):
        """
        Returns a 1-D Numpy array of N numbers for the feature set.
        """
        result = np.zeros(self.numFeatures)
        result[0] = 1.0
        # 1 = bottom, -1 = top
        side = np.sign(state.player.y)
        # becomes 1 = bottom, 0 = top
        side = int(max(0, side))
        ballPos = state.ball.lastKnownPos
        ballGrid = self.getGrid(ballPos)
        playerGrid = self.getGrid(state.player.pos)
        ballPlayerGrid = np.dot(ballGrid.reshape(9, 1), playerGrid.reshape(1, 9))
        self.setFeatures(result, side, action, self.F_BALL_PLAYER_GRID, ballPlayerGrid.flatten())
        self.setFeatures(result, side, action, self.F_POSITION_OPPONENT, state.opponent.pos)
        if np.isnan(ballPos[0]):
            return result
        self.setFeatures(result, side, action, self.F_POSITION_BALL, ballPos)
        self.setFeatures(result, side, action, self.F_BALL_DISTANCE, ballPos - state.player.pos)
        if np.isnan(state.expectedBallVelocity[0]):
            return result
        ballT1 = ballPos + state.expectedBallVelocity
        ballT3 = ballPos + 3.0 * state.expectedBallVelocity
        self.setFeatures(result, side, action, self.F_BALL_DISTANCE_T1, ballT1 - state.player.pos)
        self.setFeatures(result, side, action, self.F_BALL_DISTANCE_T3, ballT3 - state.player.pos)
        # gridlines
        return result
    
    def format(self, agent: ApproximateQAgent):
        rep = ''
        w = agent.getWeights()
        for action in range(self.numFeatures // self.FEATS_PER_ACT):
            for feat in range(self.FEATS_PER_ACT):
                idx = 1 + self.FEATS_PER_ACT * action + feat
                rep += '{0:>16.6g}'.format(w[idx])
            rep += '\n'
        return rep


class DistanceToLineRewardShaping(RewardShaping):
    @property
    def name(self) -> str:
        return 'distance-to-line'
    
    def intersection_times(self, pos1, vel1, pos2, vel2):
        """Find the times when each object will cross their velocities' intersection point.
        
        [x1, y1] + a[vx1, vy1] = [x2, y2] + b[vx2, vy2]
        a(vx1) - b(vx2) = -x1 + x2
        a(vy1) - b(vy2) = -y1 + y2
        -> find and return (a, b)
        -> or return None if the lines won't intersect
        """
        matrix = np.array([vel1, -vel2]).T
        if np.linalg.det(matrix) == 0:
            return None
        x1, y1 = pos1
        x2, y2 = pos2
        # solve the system of equations:
        #  AX = Y
        #  -> A^-1 AX = A^-1Y
        #  -> X = A^-1 Y
        solution = np.dot(np.linalg.inv(matrix), np.array([[-x1 + x2], [-y1 + y2]]))
        # number of steps until impact for player and ball, respectively
        return tuple(solution.flatten())
    
    def state_to_reward(self, state: TennisState):
        if np.sign(state.ball.y) == np.sign(state.player.y):
            ballVelocity = state.expectedBallVelocity
            ballPos = state.ball.lastKnownPos
            if np.isnan(ballVelocity[0]) or np.isnan(ballPos[0]):
                return 0.0
            ballPathAngle = np.arctan2(ballVelocity[1], ballVelocity[0])
            playerToPathDist = (
                np.cos(ballPathAngle) * (ballPos[1] - state.player.y)
                - np.sin(ballPathAngle) * (ballPos[0] - state.player.x)
            )
            return -1.0 * (playerToPathDist ** 2)
        return 0.0


class IntersectionRewardShaping(RewardShaping):
    @property
    def name(self) -> str:
        return 'intersection'
    
    def intersection_times(self, pos1, vel1, pos2, vel2):
        """Find the times when each object will cross their velocities' intersection point.
        
        [x1, y1] + a[vx1, vy1] = [x2, y2] + b[vx2, vy2]
        a(vx1) - b(vx2) = -x1 + x2
        a(vy1) - b(vy2) = -y1 + y2
        -> find and return (a, b)
        -> or return None if the lines won't intersect
        """
        matrix = np.array([vel1, -vel2]).T
        if np.linalg.det(matrix) == 0:
            return None
        x1, y1 = pos1
        x2, y2 = pos2
        # solve the system of equations:
        #  AX = Y
        #  -> A^-1 AX = A^-1Y
        #  -> X = A^-1 Y
        solution = np.dot(np.linalg.inv(matrix), np.array([[-x1 + x2], [-y1 + y2]]))
        # number of steps until impact for player and ball, respectively
        return tuple(solution.flatten())
    
    def state_to_reward(self, state: TennisState):
        ballPos = state.ball.lastKnownPos
        if np.isnan(ballPos[0]):
            return 0.0
        if np.isnan(state.player.velocity[0]):
            return 0.0
        ballVelocity = state.expectedBallVelocity
        if np.isnan(ballVelocity[0]):
            return 0.0
        playerToBall = ballPos - state.player.pos
        opponentToBall = ballPos - state.opponent.pos
        if np.dot(playerToBall, ballVelocity) > np.dot(opponentToBall, ballVelocity):
            # ball direction is closer to "towards player" than "towards opponent"
            return 0.0
        crossTimes = self.intersection_times(state.player.pos, state.player.velocity,
                                             ballPos, ballVelocity)
        if crossTimes is None:
            return 0.0
        a, b = crossTimes
        if a < 0 or b < 0:
            return -0.1
        return 0.1 * (b / (a + 1.0))
    

ALPHA = 0.015
DISCOUNT = 0.8
def run():
    env = gym.make('Tennis-v0')
    if '--explain' in sys.argv:
        print(env.unwrapped.get_action_meanings())
        return
    # initialize agent
    actionsAndProbs = getActionDistribution(env.action_space)
    agent = ApproximateQAgent(
        extractor=TennisExtractor(actionsAndProbs),
        actionSpace=actionsAndProbs,
        alpha=ALPHA,
        discount=DISCOUNT,
        rewardShaping=DistanceToLineRewardShaping()
    )
    scoreRows = []
    gamesPlayed = countPreviousScores()
    print('\n\n{} games already played'.format(gamesPlayed))

    try:
        while gamesPlayed <= 5000:
            obs = env.reset()
            Locator.firstImage(obs)
            state = TennisState(obs, None, False, None)
            gameSequence = [state]
            frozenAtPointEnd = False
            while True:
                probExplore = np.interp(gamesPlayed, [0, 5000], [0.8, 0.1])
                if np.random.rand() < probExplore:
                    action = env.action_space.sample()
                else:
                    action = agent.computeActionFromQValues(state)
                if action is None:
                    break
                nextObs, reward, done, info = env.step(action)
                if frozenAtPointEnd:
                    # the ball and players stay frozen for a while when a point is
                    # won or lost, so we detect when it's back to start analyzing the game.
                    if not (nextObs == state.obs).all():
                        frozenAtPointEnd = False

                firstPointOfGame = (len(gameSequence) == 0) and not frozenAtPointEnd
                nextState = TennisState(nextObs, state, done, info, reset=firstPointOfGame)
                if not frozenAtPointEnd:
                    gameSequence.append(nextState)
                if reward != 0:
                    gamesPlayed += 1
                    row = analyzeGame(gameSequence, reward)
                    if row[0] is None:
                        print('Warning: could not tell which player served')
                    else:
                        scoreRows.append(row)
                    if gamesPlayed % 50 == 1:
                        saveGameRecording(gameSequence, f'GAME{str(gamesPlayed).zfill(5)}', actionsAndProbs,
                                          agent.featExtractor)
                    gameSequence.clear()
                    frozenAtPointEnd = True
                    
                # print("State: \n", state, "Action: \n", action, "Nextstate: \n", nextState, "reward: \n", reward)
                agent.update(state, action, nextState, reward)
                # if step % 10 == 9:
                #     print(agent)
                state = nextState
                
                env.render('human')
    except KeyboardInterrupt:
        print('Exiting and saving weights')
    finally:
        agent.saveWeights()
        saveScore(scoreRows)


if __name__ == '__main__':
    run()