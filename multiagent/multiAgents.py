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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # previous_count = gameState.getNumFood()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # previous_food_count = currentGameState.getNumFood()
        # print("previous_food_count = " , previous_food_count)

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # successor_food_count = successorGameState.getNumFood()
        # print("successor_food_count = " , successor_food_count)

        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodnumber = len(newFood.asList())

        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        # visit state with smallest dist
        if len(foodDistances) == 0:
            minFoodDistance = 0
        else:
            minFoodDistance = min(foodDistances)

        GhostPosition = [Ghost.getPosition() for Ghost in newGhostStates]
        ghost_scores = [manhattanDistance(newPos , ghost) for ghost in GhostPosition]
        ghost_distance = min(ghost_scores)
        if ghost_distance != 0:
            ghost_distance = 1 / ghost_distance
        else:
            ghost_distance = 0

        return successorGameState.getScore() - 0.2 * minFoodDistance - ghost_distance
       
        return successorGameState.getScore()


#        a , b , c , d  = - 300 , - 5 , 0 , 100000

        # 1/9 = 0.1111
        # 1/10 = 0.1000

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        
        # print("self_depth = " , self.depth)
        currentGameState = gameState
        agent_number = gameState.getNumAgents()

          
        # print(self.index)

        current_depth = self.index # ???

        current_type = current_depth % agent_number

        # print("current_depth = " , current_depth)
        # if currentGameState.isWin() or currentGameState.isLose() or current_depth == self.depth:
        if currentGameState.isWin() or currentGameState.isLose() or current_depth == (self.depth * agent_number) :
            
            # print("current_depth = " , current_depth)
            self.evaluation = self.evaluationFunction(currentGameState)
            # print("self.index = " , self.index)
            # print("value = " , self.evaluation)
            return self.evaluation


      
        legalMoves = gameState.getLegalActions(current_type)
       
        successor_list = []
        values_list = []

        for action in legalMoves:
            successor  = currentGameState.generateSuccessor(current_type, action)
          
            self.index = self.index + 1
            next_action = self.getAction(successor)
            temp_value = self.evaluation
            self.index = self.index - 1
            next_value = temp_value

            successor_list.append(successor)
            values_list.append(next_value)
        
        # print("self.index = " , self.index)
        # print("value_list = " , values_list)
        if current_type == 0:
            # print("sfsdf")
            # print("value_list = " , values_list)
            self.evaluation = max(values_list)
            # print("self.evaluation = " , self.evaluation)
            bestIndices = [index for index in range(len(values_list)) if values_list[index] == self.evaluation]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return legalMoves[chosenIndex]
            # return self.evaluation
        
        else:
            self.evaluation = min(values_list)
            bestIndices = [index for index in range(len(values_list)) if values_list[index] == self.evaluation]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return legalMoves[chosenIndex]
            # return self.evaluation
        
inf = 1e+5


    
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def helper_action(self , gameState:GameState , p_alpha , p_beta):

        alpha = p_alpha
        beta = p_beta

        currentGameState = gameState
        agent_number = gameState.getNumAgents()

        # print(self.index)

        current_depth = self.index # ???

        current_type = current_depth % agent_number

        # print("current_depth = " , current_depth)
        # if currentGameState.isWin() or currentGameState.isLose() or current_depth == self.depth:
        if currentGameState.isWin() or currentGameState.isLose() or current_depth == (self.depth * agent_number) :
            
            # print("current_depth = " , current_depth)]
            
            self.evaluation = self.evaluationFunction(currentGameState)
      
            return self.evaluation
      
        legalMoves = gameState.getLegalActions(current_type)
       
        if current_type == 0:
            v = -inf
        else:
            v = +inf

        value_list = []

        for action in legalMoves:


            successor  = currentGameState.generateSuccessor(current_type, action)
            self.index = self.index + 1

            # self.evalution = temp_parent_value
            next_action = self.helper_action(successor , alpha , beta)  #self.evalution will change

            # self.evaluation = temp_parent_value

            temp_value = self.evaluation
            value_list.append(temp_value)
                   
            print("")
            print("current_depth = " , self.index)
            print("child_value = " , temp_value)
            print("before_alpha = " , alpha)
            print("before_beta = " , beta)
            # print("current_type  = " , current_type)
            print("v = " , v)

            self.index = self.index - 1

            if current_type == 0: # max_value

                if v < temp_value:
                    v = temp_value
                    best_action = action 

                    # self.evaluation = temp_value
                    print("current_depth = " , self.index)
                    print("parent_value = " , self.evaluation)

                    self.index = self.index - 1
                    print("test_evalution = " , self.evaluation)
                    self.index = self.index + 1

                if v > beta:
                    return best_action
                
                alpha = max(alpha , v)

            else: # min_value

                if v > temp_value:
                    v = temp_value
                    best_action = action 
                    # self.evaluation = temp_value
                    print("current_depth = " , self.index)
                    print("parent_value = " , self.evaluation)

                    self.index = self.index - 1
                    print("test_evalution = " , self.evaluation)
                    self.index = self.index + 1

                if v < alpha:
                    return best_action
                
                beta = min(beta , v)

            print("after_alpha = " , alpha)
            print("after_beta = " , beta)

        if current_type == 0:
            self.evaluation = max(value_list)
        else:
            self.evaluation = min(value_list)

        return best_action


    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = -inf
        beta = +inf

        currentGameState = gameState
        agent_number = gameState.getNumAgents()

        # print(self.index)

        current_depth = self.index # ???

        current_type = current_depth % agent_number
      
        legalMoves = gameState.getLegalActions(current_type)
       
        
        v = -inf
        for action in legalMoves:


            successor  = currentGameState.generateSuccessor(current_type, action)
            self.index = self.index + 1


            next_action = self.helper_action(successor , alpha , beta)  #self.evalution will change

            temp_value = self.evaluation

            print("")
            print("current_depth = " , self.index)
            print("child_value = " , temp_value)
            print("before_alpha = " , alpha)
            print("before_beta = " , beta)
            # print("current_type  = " , current_type)
            print("v = " , v)

            self.index = self.index - 1

            if v < temp_value:
                v = temp_value
                best_action = action 
                self.evaluation = temp_value

            if v > beta:
                return best_action
            
            alpha = max(alpha , v)

         
            print("alpha = " , alpha)
            print("beta = " , beta)


        return best_action




class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        currentGameState = gameState
        agent_number = gameState.getNumAgents()

          
        # print(self.index)

        current_depth = self.index # ???

        current_type = current_depth % agent_number

        # print("current_depth = " , current_depth)
        # if currentGameState.isWin() or currentGameState.isLose() or current_depth == self.depth:
        if currentGameState.isWin() or currentGameState.isLose() or current_depth == (self.depth * agent_number) :
            
            # print("current_depth = " , current_depth)
            self.evaluation = self.evaluationFunction(currentGameState)
            # print("self.index = " , self.index)
            # print("value = " , self.evaluation)
            return self.evaluation


      
        legalMoves = gameState.getLegalActions(current_type)
       
        successor_list = []
        values_list = []

        for action in legalMoves:
            successor  = currentGameState.generateSuccessor(current_type, action)
          
            self.index = self.index + 1
            next_action = self.getAction(successor)
            temp_value = self.evaluation
            self.index = self.index - 1
            next_value = temp_value

            successor_list.append(successor)
            values_list.append(next_value)
        
        # print("self.index = " , self.index)
        # print("value_list = " , values_list)
        if current_type == 0:
            # print("sfsdf")
            # print("value_list = " , values_list)
            self.evaluation = max(values_list)
            # print("self.evaluation = " , self.evaluation)
            bestIndices = [index for index in range(len(values_list)) if values_list[index] == self.evaluation]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return legalMoves[chosenIndex]
            # return self.evaluation
        
        else:
            self.evaluation = sum(values_list) / len(values_list)
            bestIndices = [index for index in range(len(values_list))]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return legalMoves[chosenIndex]
  


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # successorGameState = currentGameState.generatePacmanSuccessor(action)

        # successor_food_count = successorGameState.getNumFood()
        # print("successor_food_count = " , successor_food_count)
    successorGameState = currentGameState

    # successor_food_count = successorGameState.getNumFood()
    # print("successor_food_count = " , successor_food_count)

    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodnumber = len(newFood.asList())

    foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
    # visit state with smallest dist
    if len(foodDistances) == 0:
        minFoodDistance = 0
    else:
        minFoodDistance = min(foodDistances)

    GhostPosition = [Ghost.getPosition() for Ghost in newGhostStates]
    ghost_scores = [manhattanDistance(newPos , ghost) for ghost in GhostPosition]
    ghost_distance = min(ghost_scores)
    if ghost_distance != 0:
        ghost_distance = 1 / ghost_distance
    else:
        ghost_distance = 0

    return 1.5 * successorGameState.getScore() - 0.4 * minFoodDistance - 1.9 * ghost_distance
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
