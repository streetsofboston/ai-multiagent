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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newScore = successorGameState.getScore()

        import util

        newFoodList = newFood.asList()
        
        # Get the closest distance to a food pellet.
        newFoodDistances = [util.manhattanDistance(foodPos, newPos) for foodPos in newFoodList]
        minDistance = min(newFoodDistances) if len(newFoodDistances) > 0 else None
        
        # Get the closest distance to any ghost (won't remember which gost if more than one ghost)
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        newGhostDistances = [util.manhattanDistance(ghostPos, newPos) for ghostPos in newGhostPositions]
        minGhostDistance = min(newGhostDistances) if len(newGhostDistances) > 0 else None
        
        # Get closet distance to a capsule (i.e. food-pellet make ghosts scared)
        capsuleList = currentGameState.getCapsules()
        newCapsuleList = successorGameState.getCapsules()
        capsuleDistances = [util.manhattanDistance(pos, newPos) for pos in newCapsuleList]
        minCapsuleDistance = min(capsuleDistances) if len(capsuleDistances) > 0 else None
        if len(capsuleList) > len(newCapsuleList):
            newScore += 1
        
        # Get the minimum amount of time that any ghost is still scared (just to be cautious).
        minScaredTime = min(newScaredTimes)
        
        # Scoring
        minDistanceScore = 1.0 / float(minDistance + 1) if minDistance != None else 0
        ghostDistanceScore = 1.0 / float(minGhostDistance + 1) if minGhostDistance != None else 0
        minCapsuleDistanceScore = 1.0 / float(minCapsuleDistance + 1) if minCapsuleDistance != None else 0

        if (minGhostDistance != None) and (minGhostDistance <= 1):
            # If any ghost is normal, run from them if one of them is really close.
            # Only if all of them scared, go to them.
            return -1 if minScaredTime == 0 else newScore + 1
        else:
            # If any ghost is normal, get the closest food-pellet or capsule
            # IF all ghost scared, go to them.
            score = max(minCapsuleDistanceScore, minDistanceScore) if minScaredTime == 0 else ghostDistanceScore
            return newScore + score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        bestAction = minimax(self, gameState, self.index, self.depth, None)
        return bestAction[1][0]

def minimax(agent, gameState, agentIndex, ply, alphaBeta):
    import sys
    isMaximizingPlayer = (agentIndex == 0)

    if (isMaximizingPlayer and ply == 0) or gameState.isWin() or gameState.isLose():
        return agent.evaluationFunction(gameState), []
  
    actions = gameState.getLegalActions(agentIndex)
        
    if isMaximizingPlayer:
        return max_value(agent, gameState, agentIndex, ply, actions, alphaBeta)
    else:
        return min_value(agent, gameState, agentIndex, ply, actions, alphaBeta)
    
def max_value(agent, gameState, agentIndex, ply, actions, alphaBeta):
    import sys
    nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

    # Save and restore the alphaBeta so that it only propagates from path to root, not over the entire tree.
    savedAlphaBeta = [] + alphaBeta if alphaBeta != None else None
    
    bestValue = -(sys.maxint-1), []
    for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        minimaxResult = minimax(agent, successorState, nextAgentIndex, ply - 1, alphaBeta)
           
        value = minimaxResult[0], [action] + minimaxResult[1]
        if value[0] > bestValue[0]:
            bestValue = value
        
        if alphaBeta != None:    
            if (bestValue[0] > alphaBeta[1]):
                alphaBeta[0] = savedAlphaBeta[0]
                alphaBeta[1] = savedAlphaBeta[1]
                return bestValue
            
            alphaBeta[0] = max(alphaBeta[0], bestValue[0])
        
    if savedAlphaBeta != None:
        alphaBeta[0] = savedAlphaBeta[0]
        alphaBeta[1] = savedAlphaBeta[1]

    return bestValue
    
def min_value(agent, gameState, agentIndex, ply, actions, alphaBeta):
    import sys
    nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

    # Save and restore the alphaBeta so that it only propagates from path to root, not over the entire tree.
    savedAlphaBeta = [] + alphaBeta if alphaBeta != None else None

    bestValue = sys.maxint, []
    for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        minimaxResult = minimax(agent, successorState, nextAgentIndex, ply, alphaBeta)
            
        value = minimaxResult[0], [action] + minimaxResult[1]
        if value[0] < bestValue[0]:
            bestValue = value
           
        if alphaBeta != None:    
            if (bestValue[0] < alphaBeta[0]):
                alphaBeta[0] = savedAlphaBeta[0]
                alphaBeta[1] = savedAlphaBeta[1]
                return bestValue
            
            alphaBeta[1] = min(alphaBeta[1], bestValue[0])
            
    if savedAlphaBeta != None:
        alphaBeta[0] = savedAlphaBeta[0]
        alphaBeta[1] = savedAlphaBeta[1]

    return bestValue

    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import sys

        alphaBeta = [-(sys.maxint-1), sys.maxint]
        bestAction = minimax(self, gameState, self.index, self.depth, alphaBeta)
        return bestAction[1][0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        bestAction = expectimax(self, gameState, self.index, self.depth)
        return bestAction[1][0]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def expectimax(agent, gameState, agentIndex, ply):
    import sys
    isMaximizingPlayer = (agentIndex == 0)

    if (isMaximizingPlayer and ply == 0) or gameState.isWin() or gameState.isLose():
        return agent.evaluationFunction(gameState), []
  
    actions = gameState.getLegalActions(agentIndex)
        
    if isMaximizingPlayer:
        return max_value2(agent, gameState, agentIndex, ply, actions)
    else:
        return expected_value(agent, gameState, agentIndex, ply, actions)
    
def max_value2(agent, gameState, agentIndex, ply, actions):
    import sys
    nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

    bestValue = -(sys.maxint-1), []
    for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        minimaxResult = expectimax(agent, successorState, nextAgentIndex, ply - 1)
           
        value = minimaxResult[0], [action] + minimaxResult[1]
        if value[0] > bestValue[0]:
            bestValue = value
        
    return bestValue
    
def expected_value(agent, gameState, agentIndex, ply, actions):
    nextAgentIndex = (agentIndex + 1) % gameState.getNumAgents()

    bestValue = 0, []
    chance = 1.0 / float(len(actions))
    
    for action in actions:
        successorState = gameState.generateSuccessor(agentIndex, action)
        minimaxResult = expectimax(agent, successorState, nextAgentIndex, ply)
            
        expectedValue = minimaxResult[0]
        bestValue = bestValue[0] + chance * expectedValue, minimaxResult[1]
           
    return bestValue

# Abbreviation
better = betterEvaluationFunction

