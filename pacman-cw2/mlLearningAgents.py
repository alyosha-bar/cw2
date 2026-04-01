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
       #using board coordinates for pacman and ghosts, and a tuple of food coordinates for the food grid
        pacman_pos = state.getPacmanPosition()
        ghost_pos = tuple(state.getGhostPositions()) 
        food_grid = tuple(state.getFood().asList())  
        
        # get legal actions and remove STOP to simplify the action space
        self.legalActions = state.getLegalActions()

        # find the best state representation that captures all relevant information for decision-making using a tuple of all the information extracted from the top three lines. 
        self.features = (
            pacman_pos,
            ghost_pos,
            food_grid,
            tuple(sorted(self.legalActions))
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
                 epsilon: float = 0.2,
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
        
        # count the number of games we have played
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
       
        
        return endState.getScore() - startState.getScore()

       
       


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
        
        # optimistic pick --> reward under-explored actions but decay the reward 
        if counts < self.maxAttempts:
            return utility + 1000.0 / (1 + counts)
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

        # get legal moves
        legal = state.getLegalPacmanActions()
        
        # remove STOP because it is not a useful action in almost all scenarios
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # learn from the previous transition before choosing the next action
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

        # epsilon-greedy exploration
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

        # udpates counts and stores the last state and action for learning in the next step
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

        # compute final reward
        reward = self.computeReward(self.lastState, state)

        # learn from the final transition
        self.learn(
            GameStateFeatures(self.lastState),
            self.lastAction,
            reward,
            GameStateFeatures(state),
        )

        # reset actions and states for the next episode
        self.lastState = None
        self.lastAction = None

        print(f"Game {self.getEpisodesSoFar()} just ended!")

        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)