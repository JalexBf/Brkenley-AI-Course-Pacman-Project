# multiAgents.py
# --------------
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


import util
from game import Agent
import random
from typing import Optional, Tuple
import math
import time


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (exercise 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getPossibleActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateNextState(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWinningState():
        Returns whether or not the game state is a winning state

        gameState.isLosingState():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def value(state, depth, agentIndex):
            # Check if the state is a terminal state or if depth limit is reached
            if state.isWinningState() or state.isLosingState() or depth == 0:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (MAX agent)
                return max_value(state, depth, agentIndex)
            else:  # Ghost's turn (MIN agent)
                return min_value(state, depth, agentIndex)

        def max_value(state, depth, agentIndex):
            best_score = float('-inf')  # Initialize best_score as negative infinity
            actions = state.getPossibleActions(agentIndex)

            # Find the action that maximizes the value recursively
            return max(value(state.generateNextState(agentIndex, action), depth, agentIndex + 1) for action in actions)

        def min_value(state, depth, agentIndex):
            best_score = float('inf')  # Initialize best_score as positive infinity
            actions = state.getPossibleActions(agentIndex)

            if agentIndex == state.getNumAgents() - 1:
                # Last ghost's turn, start from Pacman again
                return min(value(state.generateNextState(agentIndex, action), depth - 1, 0) for action in actions)
            else:
                # Next ghost's turn
                return min(value(state.generateNextState(agentIndex, action), depth, agentIndex + 1) for action in actions)

        best_action = None
        best_value = float('-inf')  # Initialize best_value as negative infinity

        # Evaluate each possible action for Pacman
        for action in gameState.getPossibleActions(0):  # Pacman's turn (MAX agent)
            next_state = gameState.generateNextState(0, action)
            # Calculate the value of the action using the minimax algorithm
            action_value = value(next_state, self.depth, 1)  # Start from the first ghost
            if action_value > best_value:
                # Update the best action and value if a better action is found
                best_value = action_value
                best_action = action
                if best_value >= float('inf'):
                    break  # Early termination if the best value is a winning state

        return best_action

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    ALPHA_INFINITY = float('-inf')
    BETA_INFINITY = float('inf')

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction

        Parameters:
        - gameState: The current game state

        Returns:
        - The best action to take according to the minimax algorithm with alpha-beta pruning
        """
        def maxLevel(gameState, depth, alpha, beta):
            """
            Performs the max level of the minimax algorithm

            Parameters:
            - gameState: The current game state
            - depth: The current depth in the search tree
            - alpha: The best value found for the maximizing player
            - beta: The best value found for the minimizing player

            Returns:
            - The maximum value found at this level
            """
            currDepth = depth + 1
            if gameState.isWinningState() or gameState.isLosingState() or currDepth == self.depth:
                return self.evaluationFunction(gameState)

            # Initialize the maximum value as negative infinity
            max_value = self.ALPHA_INFINITY

            actions = gameState.getPossibleActions(0)
            alpha1 = alpha
            for action in actions:
                successor = gameState.generateNextState(0, action)
                # Recursive call to the min level
                max_value = max(max_value, minLevel(successor, currDepth, 1, alpha1, beta))

                # Perform alpha-beta pruning
                if max_value > beta:
                    return max_value
                alpha1 = max(alpha1, max_value)

            return max_value

        def minLevel(gameState, depth, agentIndex, alpha, beta):
            """
            Performs the min level of the minimax algorithm

            Parameters:
            - gameState: The current game state
            - depth: The current depth in the search tree
            - agentIndex: The index of the current agent
            - alpha: The best value found for the maximizing player
            - beta: The best value found for the minimizing player

            Returns:
            - The minimum value found at this level
            """
            min_value = self.BETA_INFINITY
            if gameState.isWinningState() or gameState.isLosingState():
                return self.evaluationFunction(gameState)

            actions = gameState.getPossibleActions(agentIndex)
            beta1 = beta
            for action in actions:
                successor = gameState.generateNextState(agentIndex, action)
                if agentIndex == (gameState.getNumAgents() - 1):
                    # Last ghost's turn, so we need to go back to the max level
                    min_value = min(min_value, maxLevel(successor, depth, alpha, beta1))

                    # Perform alpha-beta pruning
                    if min_value < alpha:
                        return min_value
                    beta1 = min(beta1, min_value)
                else:
                    # Recursive call to the min level
                    min_value = min(min_value, minLevel(successor, depth, agentIndex + 1, alpha, beta1))

                    # Perform alpha-beta pruning
                    if min_value < alpha:
                        return min_value
                    beta1 = min(beta1, min_value)

            return min_value

        actions = gameState.getPossibleActions(0)
        currentScore = self.ALPHA_INFINITY
        returnAction = ''
        alpha = self.ALPHA_INFINITY
        beta = self.BETA_INFINITY
        for action in actions:
            nextState = gameState.generateNextState(0, action)
            # Recursive call to the min level
            score = minLevel(nextState, 0, 1, alpha, beta)

            # Choose the action with the maximum score
            if score > currentScore:
                returnAction = action
                currentScore = score

            # Perform alpha-beta pruning
            if score > beta:
                return returnAction
            alpha = max(alpha, score)

        return returnAction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action, _ = self.expectimax(0, 0, gameState)  # Get the action from the expectimax algorithm
        return action

    def expectimax(self, depth, agent_idx, gameState):
        """
        Returns the best action and score for an agent using the expectimax algorithm.

        Args:
            depth (int): The current depth of the search tree.
            agent_idx (int): The index of the current agent.
            gameState (GameState): The current game state.

        Returns:
            tuple: A tuple containing the best action and score.
        """
        if agent_idx >= gameState.getNumAgents():
            # If all agents have been evaluated, reset the agent index and increment the depth
            agent_idx = 0
            depth += 1

        if depth == self.depth or gameState.isWinningState() or gameState.isLosingState():
            # If the maximum depth is reached or the game is in a terminal state, return the evaluation
            return None, self.evaluationFunction(gameState)

        if agent_idx == 0:  # Max player (Pacman)
            return self.max_value(depth, agent_idx, gameState)
        else:  # Min players (Ghosts)
            return self.exp_value(depth, agent_idx, gameState)

    def max_value(self, depth, agent_idx, gameState):
        """
        Returns the best action and score for the max player (Pacman).

        Args:
            depth (int): The current depth of the search tree.
            agent_idx (int): The index of the current agent.
            gameState (GameState): The current game state.

        Returns:
            tuple: A tuple containing the best action and score.
        """
        best_score = float('-inf')
        best_action = None

        for action in gameState.getPossibleActions(agent_idx):
            next_state = gameState.generateNextState(agent_idx, action)
            _, child_score = self.expectimax(depth, agent_idx + 1, next_state)

            if child_score > best_score:
                # Update the best action and score if a higher score is found
                best_score = child_score
                best_action = action

        return best_action, best_score

    def exp_value(self, depth, agent_idx, gameState):
        """
        Returns the average score for the min players (Ghosts).

        Args:
            depth (int): The current depth of the search tree.
            agent_idx (int): The index of the current agent.
            gameState (GameState): The current game state.

        Returns:
            tuple: A tuple containing None (no action) and the average score.
        """
        ghost_actions = gameState.getPossibleActions(agent_idx)
        num_actions = len(ghost_actions)
        total_score = 0

        for action in ghost_actions:
            next_state = gameState.generateNextState(agent_idx, action)
            _, child_score = self.expectimax(depth, agent_idx + 1, next_state)
            total_score += child_score

        return None, total_score / num_actions
        util.raiseNotDefined()



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (exercise 4).

    DESCRIPTION: <write something here so we know what you did>
    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    """ Manhattan distance to the foods from the current state """
    foodList = newFood.asList()
    from util import manhattanDistance
    foodDistance = [0]
    for pos in foodList:
        foodDistance.append(manhattanDistance(newPos, pos))

    """ Manhattan distance to each ghost from the current state"""
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghostDistance = [0]
    for pos in ghostPos:
        ghostDistance.append(manhattanDistance(newPos, pos))

    numberofPowerPellets = len(currentGameState.getCapsules())

    score = 0
    numberOfNoFoods = len(newFood.asList(False))
    sumScaredTimes = sum(newScaredTimes)
    sumGhostDistance = sum(ghostDistance)
    reciprocalfoodDistance = 0
    if sum(foodDistance) > 0:
        reciprocalfoodDistance = 1.0 / sum(foodDistance)

    score += currentGameState.getScore() + reciprocalfoodDistance + numberOfNoFoods

    if sumScaredTimes > 0:
        score += sumScaredTimes + (-1 * numberofPowerPellets) + (-1 * sumGhostDistance)
    else:
        score += sumGhostDistance + numberofPowerPellets
    return score


# Abbreviation
better = betterEvaluationFunction
