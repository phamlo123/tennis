# Skeleton Code taken from HW3
from typing import Any, Dict, Optional
from gym.spaces import Space
from constants import Colors
from datetime import datetime
from collections import deque
import numpy as np
import pickle

class Locator:
    @classmethod
    def firstImage(cls, obs):
        blue = Colors.BLUE
        blue_mask = (
            (obs[:, :, 0] == blue[0])
            & (obs[:, :, 1] == blue[1])
            & (obs[:, :, 2] == blue[2])
        )
        pink = Colors.PINK
        pink_mask = (
            (obs[:, :, 0] == pink[0])
            & (obs[:, :, 1] == pink[1])
            & (obs[:, :, 2] == pink[2])
        )
        # left and right side of screen have some blue and pink
        # lines which are not part of the player/opponent.
        height, width = obs.shape[:2]
        sideline_amount = int(width * 0.3)
        left_mask = np.zeros((height, width), dtype=bool)
        # rectangle insets: top=20px bottom=30px; left=0 width=30%
        left_mask[20:-30, :sideline_amount] = True
        right_mask = np.zeros((height, width), dtype=bool)
        # rectangle insets: top=20px bottom=30px; right=0 width=30%
        right_mask[20:-30, -sideline_amount:] = True
        
        sideline_left = left_mask & (blue_mask | pink_mask)
        sideline_right = right_mask & (blue_mask | pink_mask)
        sideline_leftpts = np.argwhere(sideline_left)
        sideline_rightpts = np.argwhere(sideline_right)
        lline = np.polyfit(sideline_leftpts[:, 0], sideline_leftpts[:, 1], 1)
        rline = np.polyfit(sideline_rightpts[:, 0], sideline_rightpts[:, 1], 1)
        cls.sideline_x = []
        for y in range(height):
            # x = my + b to get best guess range of tennis court
            cls.sideline_x.append((np.polyval(lline, y), np.polyval(rline, y)))

        # also block out the top 20 pixels which might have the score text,
        # and the bottom 30 pixels where the activision logo is.
        score_mask = np.zeros((height, width), dtype=bool)
        score_mask[:20, :] = True
        score_mask[-30:, :] = pink_mask[-30:, :]
        # screen-sized matrix with 1 for pixels with ignored blue or pink, 0 elsewhere
        sideline_mask = (sideline_left | sideline_right) & (blue_mask | pink_mask)
        cls.sideline_remover = score_mask | sideline_mask
        cls.screen_shape = blue_mask.shape

    @classmethod
    def meanPositionOfColor(cls, obs, color):
        """
        Returns the mean position of the color in the given pixel array, scaling both
        coordinates to a roughly [-1, 1] range.
        """
        # 210x160 boolean grid, true only if color matches
        color_mask = (
            (obs[:, :, 0] == color[0])
            & (obs[:, :, 1] == color[1])
            & (obs[:, :, 2] == color[2])
        )
        color_locs = np.argwhere(color_mask & ~cls.sideline_remover)
        # reverse indices for usual (x, y) order
        xy = color_locs.mean(axis=0)[::-1]
        if np.isnan(xy[0]):
            return xy
        xy[0] /= cls.screen_shape[1]
        xy[1] /= cls.screen_shape[0]
        return xy * 2.0 - 1.0


def findLastIndex(lst, predicate):
    for j in reversed(range(len(lst))):
        if predicate(lst[j]):
            return j
    return -1


NAN_ARRAY = np.ones(2) * np.nan
class TennisObject:
    def __init__(self):
        self.pos = NAN_ARRAY
        self.history = deque()
    
    def copy(self):
        theCopy = TennisObject()
        theCopy.pos = self.pos
        theCopy.history = self.history
        return theCopy
    
    def update(self, pos):
        if len(self.history) > 50:
            self.history.popleft()
        if len(self.history) > 0 and (not np.isnan(pos[0])) and np.isnan(self.history[-1][0]):
            # gap detected in readings, fill it linearly
            index = self.lastGoodPositionIndex()
            if index >= 0:
                lastAccurate = self.history[index]
                delta = (pos - lastAccurate) / (len(self.history) - index)
                for j in range(index + 1, len(self.history)):
                    self.history[j] = lastAccurate + delta * (j - index)
        self.history.append(pos)
        self.pos = pos
    
    @property
    def x(self): return self.pos[0]

    @property
    def y(self): return self.pos[1]

    def lastGoodPositionIndex(self):
        return findLastIndex(self.history, lambda p: not np.isnan(p[0]))

    @property
    def lastKnownPos(self):
        index = self.lastGoodPositionIndex()
        if index >= 0:
            return self.history[index]
        return NAN_ARRAY

    @property
    def velocity(self):
        if len(self.history) < 2:
            return NAN_ARRAY
        return self.history[-1] - self.history[-2]
    
    @property
    def isMoving(self):
        v = self.velocity
        return np.isfinite(v).all() and (v != 0).all()
    
    def squaredDist(self, other: 'TennisObject'):
        if np.isnan(self.pos[0]) or np.isnan(other.pos[0]):
            return np.inf
        return np.sum((self.pos - other.pos) ** 2)
    
    def lastNPositions(self, n):
        # slice indexing doesn't work on deque
        end = len(self.history) 
        start = max(0, end - n)
        return [self.history[j] for j in range(start, end)]

class TennisState:
    def __init__(self, obs, prevState: Optional['TennisState'], done: bool, info: Any,
                 reset: bool = False):
        self.obs = obs
        self.done = done
        self.info = info
        self.prevState = prevState
        if reset or prevState is None:
            self.player = TennisObject()
            self.opponent = TennisObject()
            self.ball = TennisObject()
            self.shadow = TennisObject()
        else:
            self.player = prevState.player.copy()
            self.opponent = prevState.opponent.copy()
            self.ball = prevState.ball.copy()
            self.shadow = prevState.shadow.copy()
        
        self.player.update(Locator.meanPositionOfColor(obs, Colors.PINK))
        self.opponent.update(Locator.meanPositionOfColor(obs, Colors.BLUE))
        self.ball.update(Locator.meanPositionOfColor(obs, Colors.WHITE))
        self.shadow.update(Locator.meanPositionOfColor(obs, Colors.DARK_GRAY))
        self.ballLastKnownPos = self.ball.lastKnownPos
        # track last place opponent hit the ball
        self.opponentBallHit = self.ball.pos
        self.timeSinceOpponentApproach = 0
        self.opponentClosestApproach = self.opponent.squaredDist(self.ball)
        self.serveDetected = False
        if len(self.ball.history) >= 20:
            minX, minY = np.nanmin(self.ball.lastNPositions(20), axis=0)
            maxX, maxY = np.nanmax(self.ball.lastNPositions(20), axis=0)
            self.serveDetected = (minX == maxX) and (maxY < minY + 0.3)
        if prevState is not None:
            # prevent infinite reference chain
            prevState.prevState = None
            # last place opponent hit the ball might have been before now
            if not reset:
                keepBallHitThreshold = prevState.opponentClosestApproach + prevState.timeSinceOpponentApproach * 0.01
                if keepBallHitThreshold < self.opponentClosestApproach:
                    self.opponentBallHit = prevState.opponentBallHit
                    self.opponentClosestApproach = prevState.opponentClosestApproach
                    self.timeSinceOpponentApproach = prevState.timeSinceOpponentApproach + 1
        # use last opponent ball hit to better predict future ball velocity
        self.expectedBallVelocity = np.nanmean([self.ball.velocity, self.shadow.velocity], axis=0)
        if not np.isnan(self.expectedBallVelocity[0]):
            if self.timeSinceOpponentApproach >= 3:
                ballTraveled = self.ball.pos - self.opponentBallHit
                self.expectedBallVelocity = (
                    0.5 * self.expectedBallVelocity
                    + 0.5 * ballTraveled / float(self.timeSinceOpponentApproach + 1)
                )
                


def getActionDistribution(actionSpace: Space) -> Dict[Any, float]:
    """Samples the given action space, and returns a dictionary of action to frequency.
    
    The order of the dictionary iteration is guaranteed to stay the same between
    runs, which enables caching.
    """
    actionsAndProbs = {}
    actionSamples = 1000
    for _ in range(actionSamples):
        sample = actionSpace.sample()
        if sample in actionsAndProbs:
            actionsAndProbs[sample] += 1.0 / actionSamples
        else:
            actionsAndProbs[sample] = 1.0 / actionSamples
    # reorder
    result = {}
    for act in sorted(actionsAndProbs.keys()):
        result[act] = actionsAndProbs[act]
    return result

class FeatExtractor:
    def __init__(self, numFeatures):
        self.numFeatures = numFeatures
        
    def getFeatures(self, state: TennisState, action: int):
        """
        Returns a 1-D Numpy array of N numbers for the feature set.
        """
        raise NotImplementedError()

    def format(self, agent: 'ApproximateQAgent'):
        """
        Returns a string representation of the agent's weights,
        in an understandable form.
        """
        raise NotImplementedError()


class RewardShaping:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def state_to_reward(self, state: TennisState) -> float:
        raise NotImplementedError()

class ApproximateQAgent:
    """
    Uses Q-learning with features and linear Q-value estimation.
    """
    def __init__(self, extractor: FeatExtractor, actionSpace: Dict[Any, float], alpha: float,
                 discount: float, rewardShaping: RewardShaping):
        self.featExtractor = extractor
        self.alpha = alpha
        self.discount = discount
        self.actionsAndProbs = actionSpace
        self.rewardShaping = rewardShaping
        self.weightsFilename = f'weights-{rewardShaping.name}.pkl'
        self.weights = np.zeros(extractor.numFeatures)
        self.readWeights()

    def saveWeights(self) -> None:
        with open(self.weightsFilename, 'wb') as weightFile:
            pickle.dump(self.weights, weightFile)

    def readWeights(self):
        try:
            with open(self.weightsFilename, 'rb') as weightFile:
                self.weights = pickle.load(weightFile)
            print('Weights loaded')
        except FileNotFoundError:
            print('No weights found')
        print(self)
            
    def getLegalActions(self, state):
        if state.done:
            return []
        return list(self.actionsAndProbs.keys())

    def getQValue(self, state, action) -> float:
        """
        Computes the Q-value (expected future points from a state given the selected
        action).
        """
        feats = self.featExtractor.getFeatures(state, action)
        return np.dot(feats, self.getWeights())

    def getWeights(self):
        return self.weights

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action) where the max is over legal actions.
        """
        actions = self.getLegalActions(state)
        return max([self.getQValue(state, a) for a in actions], default=0.0)

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state. If there
        are no legal actions, return None.
        """
        actions = self.getLegalActions(state)
        return max(actions, key=lambda a: self.getQValue(state, a), default=None)

    def update(self, state, action, nextState, reward):
        prevVal = self.getQValue(state, action)
        rPrime = (reward + self.rewardShaping.state_to_reward(nextState)
                  - self.rewardShaping.state_to_reward(state))
        stepVal = rPrime + self.discount * self.computeValueFromQValues(nextState)
        stepFeats = self.featExtractor.getFeatures(state, action)
        # updates correlations - when higher feature values appear
        # with positive expected scores, the weight increases, and
        # the opposite happens with negative expected scores
        self.weights += stepFeats * self.alpha * (stepVal - prevVal)
        # if stepVal - prevVal != 0:
        #     print('update')
        #     print(prevVal, stepVal, reward)
        #     print(self)
        # self.weights += stepFeats * self.alpha * (stepVal - prevVal)
    
    def updatev2(self, state, action, nextState, reward):
        features = self.featExtractor.getFeatures(state, action)
        difference = (reward + (self.discount*self.computeValueFromQValues(nextState))) - self.getQValue(state, action)
        for feature in features:
            newWeight = self.weights[feature] + (self.alpha*difference*features[feature])
            self.weights[feature] = newWeight

    def __str__(self):
        return self.featExtractor.format(self)