# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        pacman_pos = state.getPacmanPosition()
        ghost_positions = state.getGhostPositions()

        # --- Feature 1: Closest ghost distance, bucketed ---
        # Rather than a single boolean, we bucket the nearest ghost distance
        # into 3 bands: DANGER (≤2), CLOSE (≤5), SAFE (>5).
        # This gives the agent enough warning to flee without exploding the
        # state space the way a full distance value would.
        if ghost_positions:
            min_ghost_dist = min(
                util.manhattanDistance(pacman_pos, gp) for gp in ghost_positions
            )
        else:
            min_ghost_dist = 999

        if min_ghost_dist <= 2:
            ghost_threat = "DANGER"
        elif min_ghost_dist <= 5:
            ghost_threat = "CLOSE"
        else:
            ghost_threat = "SAFE"

        # --- Feature 2: Direction *and* distance band to the nearest food ---
        # Using only direction lost whether food was 1 step away or 10 steps
        # away, which made the reward signal very noisy.  Adding a distance
        # band (NEAR / FAR) doubles the food states but dramatically sharpens
        # the signal.
        food_list = state.getFood().asList()

        if food_list:
            dists = [util.manhattanDistance(pacman_pos, f) for f in food_list]
            min_food_dist = min(dists)
            closest_food = food_list[dists.index(min_food_dist)]

            dx = closest_food[0] - pacman_pos[0]
            dy = closest_food[1] - pacman_pos[1]

            if abs(dx) > abs(dy):
                food_dir = Directions.EAST if dx > 0 else Directions.WEST
            else:
                food_dir = Directions.NORTH if dy > 0 else Directions.SOUTH

            food_dist_band = "NEAR" if min_food_dist <= 2 else "FAR"
        else:
            food_dir = None
            food_dist_band = "NONE"

        # --- Feature 3: Legal actions (encodes wall layout around Pacman) ---
        self.legalActions = state.getLegalActions()
        if Directions.STOP in self.legalActions:
            self.legalActions.remove(Directions.STOP)

        self.features = (
            ghost_threat,
            food_dir,
            food_dist_band,
            tuple(sorted(self.legalActions)),
        )

    def __hash__(self):
        return hash(self.features)

    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return self.features == other.features


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.9,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # q_values & counts
        self.q_values = {}
        self.counts = {}

        # previous actions
        self.lastState = None
        self.lastAction = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        # Use score delta as the base reward.  The game engine already
        # encodes eating food (+10), dying (-500), and winning (+500) inside
        # the score, so we must NOT add those again — that was causing the
        # agent to receive ±1000 for terminal states instead of ±500 and
        # confused the Q-table badly.
        reward = endState.getScore() - startState.getScore()

        # Small living penalty on non-terminal steps to discourage looping.
        # Terminal penalties/bonuses are already captured by the score delta
        # above, so we only add the living penalty on ordinary moves.
        if not endState.isWin() and not endState.isLose():
            reward -= 1

        return reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.q_values.get((state, action), 0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        if not state.legalActions:
            return 0.0

        return max(self.getQValue(state, a) for a in state.legalActions)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        old_q = self.getQValue(state, action)
        future_q = self.maxQValue(nextState)

        # Standard Bellman update
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.q_values[(state, action)] = new_q

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.counts[(state, action)] = self.counts.get((state, action), 0) + 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.counts.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        # Optimistic initialisation: prefer under-explored (state, action)
        # pairs, but scale the bonus DOWN as counts grow rather than keeping
        # it at a flat 1000.  A flat 1000 made the agent keep re-visiting
        # pairs it had already identified as harmful.
        if counts < self.maxAttempts:
            # Decaying bonus: the more we've tried it the less curious we are.
            return utility + (self.maxAttempts - counts) * 10
        return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Learn from the previous transition before choosing the next action
        if self.lastState is not None:
            reward = self.computeReward(self.lastState, state)
            self.learn(
                GameStateFeatures(self.lastState),
                self.lastAction,
                reward,
                GameStateFeatures(state),
            )

        print("Legal moves: ", legal)
        print("Pacman position: ", state.getPacmanPosition())
        print("Ghost positions:", state.getGhostPositions())
        print("Food locations: ")
        print(state.getFood())
        print("Score: ", state.getScore())

        stateFeatures = GameStateFeatures(state)

        # Epsilon-greedy exploration
        if util.flipCoin(self.epsilon):
            action = random.choice(legal)
        else:
            best_actions = []
            max_val = float('-inf')

            for a in legal:
                q_val = self.getQValue(stateFeatures, a)
                count = self.getCount(stateFeatures, a)
                val = self.explorationFn(q_val, count)

                if val > max_val:
                    max_val = val
                    best_actions = [a]
                elif val == max_val:
                    best_actions.append(a)

            action = random.choice(best_actions)

        self.updateCount(stateFeatures, action)
        self.lastState = state
        self.lastAction = action

        return action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        reward = self.computeReward(self.lastState, state)
        self.learn(
            GameStateFeatures(self.lastState),
            self.lastAction,
            reward,
            GameStateFeatures(state),
        )

        self.lastState = None
        self.lastAction = None

        print(f"Game {self.getEpisodesSoFar()} just ended!")

        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)