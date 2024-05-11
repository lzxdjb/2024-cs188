# searchAgents.py
# ---------------
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman
import layout

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs



    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState: pacman.GameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = [(1,1), (1,top), (right, 1), (right, top)]
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded

    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        "*** YOUR CODE HERE ***"
        startingPosition = self.startingPosition
        x = startingPosition[0]
        y = startingPosition[1]
        start = (x , y , 0 , 0)
        return start

        util.raiseNotDefined()

    def isGoalState(self, state: Any):
        """
        Returns whether this search state is a goal state of the problem.
        """
        "*** YOUR CODE HERE ***"
        
        # if state in self.corners:
        #     return True
            # print("fuck!")
            # self.corners.remove(state)


        # if len(self.corners) == 0:
        #     return True
        # else:
        #     return False
        # print("tuple = " , state)
        if state[2] == 4:
            return True

        return False

        util.raiseNotDefined()

    def is_corner(self , state: Any):

        x = state[0]
        y = state[1]
        position = (x , y)
        if position in self.corners:
            return True


    def getSuccessors(self, state: Any):

        a = self.corners[0]
        b = self.corners[1]
        c = self.corners[2]
        d = self.corners[3]

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0] , state[1]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                step_goal = state[2]

             
                cost = state[3]
                # cost = cost + 1
                if self.is_corner(nextState):

                    if nextState == a:
                        if cost % 10 <= 0:
                            cost = cost + 1
                            step_goal = step_goal + 1
                            
                    elif nextState == b:
                        if cost % 100 < 10:
                            cost = cost + 10
                            step_goal = step_goal + 1

                    elif nextState == c:
                        if cost % 1000 < 100:
                            cost = cost + 100
                            step_goal = step_goal + 1

                    elif nextState == d:
                        if cost % 10000 < 1000:
                            cost = cost + 1000
                            step_goal = step_goal + 1

                    # self.corners.remove(nextState)
                    # cost = cost + 100
                # else:
                #     cost = cost + 1
                next_successor = (nextx , nexty , step_goal , cost)
                successors.append(( next_successor, action, 1) )

        # Bookkeeping for display purposes
        # self._expanded += 1 # DO NOT CHANGE
        # if state not in self._visited:
        #     self._visited[state] = True
        #     self._visitedlist.append(state)

        return successors
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            "*** YOUR CODE HERE ***"

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)


from itertools import permutations

def jj(state, goal):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = (state[0] , state[1])
    xy2 = goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])




def cornersHeuristic(state: Any, problem: CornersProblem ):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    # game_state = pacman.GameState()
    # ll = layout.Layout()
    # game_state.initialize(ll)
    corners = []
    # corners = problem.corners # These are the corner coordinates
    # walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    # print("state" , state)
    a = state[3] // 1000
    b = state[3] // 100 % 10
    c = state[3] // 10 % 10
    d = state[3] // 1 % 10

    # print("a = " , a)
    # print("b = " , b)
    # print("c = " , c)
    # print("d = " , d)

    if(a == 0):
        corners.append(problem.corners[0])
    if(b == 0):
        corners.append(problem.corners[1])
    if(c == 0):
        corners.append(problem.corners[2])
    if(d == 0):
        corners.append(problem.corners[3])

    if(a == b == c == d == 1):
        return 0
        
#####
    # temp_permutation = list(range(0 , len(corners)))
    # # print("temp_permutation = " , temp_permutation)
    # permutation_list = list(permutations(temp_permutation))
    # # print(permutation_list)

    # cost_list = []
    # for permutation in permutation_list:
    #     cost = 0
    #     # print("permutation = " , corners[permutation[0]])
    #     # print("corner = " , corners)
    #     # print("permutation = " ,permutation)
    #     position = (state[0] , state[1])
    #     cost = cost + mazeDistance(position,corners[permutation[0]]  , game_state)
    #     for i in range(0 , len(permutation) - 1):

    #         xy1 = corners[permutation[i]]
    #         xy2 = corners[permutation[i + 1]]
    #         cost = cost + abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
    #         # cost = ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5
    #         # print(cost)
    #     cost_list.append(cost)

    # print("permutation = " , permutation)
    # print("problem_corner = " , problem.corners)
    # print("cost = " , min(cost_list))
    # return min(cost_list)
####
    # cost = 0
    # for corner in corners:
    #     cost = cost + abs(state[0] - corner[0]) + abs(state[1] - corner[1])

    return 1
    
    return cost

    "*** YOUR CODE HERE ***"
    return 0 # Default to trivial solution




class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem(pacman.GameState):
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, startingGameState: pacman.GameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append((((nextx, nexty), nextFood), direction, 2) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))


# def foodHeuristic(state: Tuple[Tuple, List[List]], problem: pacman.GameState):
def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):

    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristsicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """
   
    if state[1].count() == 0:
        return 0
    
    startState = problem.startingGameState 

    position, foodGrid = state
    food_list = foodGrid.asList()
   
    temp_permutation = list(range(0 , len(food_list)))
 
    total_cost = 0
  
    distance = []

    current_position = position

    cost1_list = []
    cost2_list = []

    cost1 = 0
    cost2 = 0

    # while len(food_list) > 0: 
    for food in food_list:
        cost = mazeDistance(current_position , food , startState)
        cost1_list.append(cost)
    cost1 = min(cost1_list)
    

    for i in range(0 , len(food_list)):
        for j in range(i + 1 , len(food_list)):
            if(i == j):
                cost = 0
            else:
                cost = mazeDistance(food_list[i] , food_list[j] , startState)

            # print("cost = " , cost)
            cost2_list.append(cost)
            # print(cost2_list)

    if len(cost2_list) == 0:
        cost2 == 0
    else:
        cost2 = min(cost2_list)
    # for food1 in food_list:
    #     for food2 in food_list:
    #         if(food1 == food2):
    #             continue

    #         cost = mazeDistance(food1 , food2 , startState)
    #         cost2_list.append(cost)
    #     cost2 = min(cost2_list)

    return max(cost1 , cost2)
        
    
    


#####
    # while len(food_list) > 0:

    #     distance = []
    #     # cost = 0
    #     for food in food_list:
    #         cost = mazeDistance(current_position ,food ,  startState)
    #         item = (food[0] , food[1] , cost)
    #         distance.append(item)

    #     min_tuple = min(distance, key=lambda x: x[2])
    #     current_position = (min_tuple[0] , min_tuple[1])
    #     food_list.remove(current_position)
    #     total_cost = total_cost + min_tuple[2]

    # total_cost = total_cost ** 0.76

####  
    return total_cost


    "*** YOUR CODE HERE ***"
    return 0

















class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print('Path found with cost %d.' % len(self.actions))

    def findPathToClosestDot(self, gameState: pacman.GameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        return problem.breadthFirstSearch(problem)


        # print("food = " , food)
        # print("wall = " , walls)
        # print("startPosition" , startPosition)

        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()


from search import SearchProblem
from util import Stack , Queue

class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state: Tuple[int, int]):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        if state in self.food:
            return True
        return False

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def change(self , visited_path : list , problem : PositionSearchProblem , true_answer):
        answer = Stack()
        visited_path.reverse()

        # print("visited_path = " , visited_path)

        i = 0

        while 1:

            # currentFood = state.getFood()
            # if currentFood[x][y] == True: 
            
            # if self.food[visited_path[i][1][0]][visited_path[i][1][1]] == False:
            if true_answer != visited_path[i][1]:
                i = i + 1
            else:
                j = i + 1
                break
        # self.food.remove(visited_path[i][1])
        answer.push(visited_path[i][2])

        while 1:
            if j == len(visited_path):
                break     

            
            if(visited_path[i][0] == visited_path[j][1]):
                i = j
                j = j + 1
                # print("visited_path = " ,visited_path[i])
                answer.push(visited_path[i][2])  
            
            # if flag == False:
            else:
                j = j + 1
                    

        # print(("fuck = " ,answer.list))
        answer.list.reverse()

        # print("answer = " , answer.list)
        # exit()
        return answer.list     

    def breadthFirstSearch(self , problem: PositionSearchProblem):

        fringe = Queue()# store all the path
        visited_path = []
        # path_stack = Stack()

        current = self.startState

        # print("current = " , current)

        # exit()

        fringe.push(current)
        visited_path.append(current)

        link = []
        # path_stack.push(current)

        while 1:

            current = fringe.pop()
            print("current = " , current)
            print("fringe = " , fringe.list)


            # visited_path.append(current)
            
            # if problem.isGoalState(current):
            # if current in self.food:
            if self.food[current[0]][current[1]] == True:
                # return [n , n]
                # print("visited_path = " , visited_path)
                # print("link = " , link)
                # exit()
                true_answer = current
                return self.change(link , problem , true_answer)

            for successor in problem.getSuccessors(current):
                if successor[0] in visited_path:
                    continue
                # flag = True
                temp_link_element = []
                temp_link_element.append(current)
                temp_link_element.append(successor[0])
                temp_link_element.append(successor[1])

                link.append(temp_link_element)

                fringe.push(successor[0])
                if  problem.isGoalState(successor[0]) == False:
                    visited_path.append(successor[0])
                # visited_path.append(successor[0])
                
