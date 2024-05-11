# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST

    test = Directions.EAST
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    return []
    return  [s, s, w, s, w, w, s]
    # return [test , test, test]
from util import Stack
from util import Queue
from util import PriorityQueue

import time

def change(visited_path : list , problem : SearchProblem):
    answer = Stack()
    visited_path.reverse()

    # print("visited_path = " , visited_path)
  
    i = 0

    while 1:
        if problem.isGoalState(visited_path[i][1]) == False:
            i = i + 1
        else:
            j = i + 1
            break
    
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
        

     
def depthFirstSearch(problem: SearchProblem):
   

    fringe = Stack()# store all the path
    visited_path = []
    # path_stack = Stack()

    current = problem.getStartState()

    fringe.push(current)
    # visited_path.append(current)

    link = []
    # path_stack.push(current)

    while 1:

        current = fringe.pop()
        visited_path.append(current)

        if problem.isGoalState(current):

            # return [n , n]
            print("visited_path = " , visited_path)
            print("link = " , link)
            # exit()
            return change(link , problem)

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

        

def breadthFirstSearch(problem: SearchProblem):

    fringe = Queue()# store all the path
    visited_path = []
    # path_stack = Stack()

    current = problem.getStartState()

    # print("current = " , current)

    # exit()

    fringe.push(current)
    visited_path.append(current)

    link = []
    # path_stack.push(current)

    while 1:

        current = fringe.pop()
        # print("current = " , current)

        # visited_path.append(current)
        
        if problem.isGoalState(current):

            # return [n , n]
            # print("visited_path = " , visited_path)
            # print("link = " , link)
            # exit()
            return change(link , problem)

        for successor in problem.getSuccessors(current):
            # print("fringe = " , fringe.list)
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

        

def uniformCostSearch(problem: SearchProblem):
  
    fringe = PriorityQueue()# store all the path
    visited_path = []
    # path_stack = Stack()

    current = problem.getStartState()

    item = []
    item.append(current)
    item.append(0)

    cc = 0

    fringe.push(item , 0) #    def push(self, item, priority ):

    # fringe.push(current , 0)
    visited_path.append(current)

    link = []
    # path_stack.push(current)

    while 1:

        item = fringe.pop() #        return priority  , item
        current = item[0]
        cost = item[1]        # print("current = " , current)
        # visited_path.append(current)
        
        if problem.isGoalState(current):

            # return [n , n]
            # if link[len(link) - 1][]
            # print("visited_path = " , visited_path)
            # print("link = " , link)
            # exit()
            return change(link , problem)

        for successor in problem.getSuccessors(current):
            if successor[0] in visited_path:
                continue
            # flag = True
            temp_link_element = []
            temp_link_element.append(current)
            temp_link_element.append(successor[0])
            temp_link_element.append(successor[1])

            link.append(temp_link_element)

            temp = cost
            temp = temp + successor[2]
            temp_current = successor[0]

            item = []
            item.append(temp_current)
            item.append(temp)

            priority = temp + 0
            fringe.push(item , priority)

            if  problem.isGoalState(successor[0]) == False:
                visited_path.append(successor[0])

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

# from util import astar_PriorityQueue

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):

    fringe = PriorityQueue()# store all the path
    visited_path = []
    # path_stack = Stack()

    current = problem.getStartState()
    item = []
    item.append(current)
    item.append(0)
    # item = list[current , 0]
    # from searchAgents import manhattanHeuristic
    
    cc = heuristic(current , problem)

    fringe.push(item , cc) #    def push(self, item, priority ):

    visited_path.append(current)

    link = []

    while 1:

        # priority , cost , current = fringe.pop()

        item = fringe.pop() #        return priority  , item
        current = item[0]
        cost = item[1]
        # print("current = " , current)
        # visited_path.append(current)
        
        if problem.isGoalState(current):

            # return [n , n]
            # if link[len(link) - 1][]
            # print("visited_path = " , visited_path)
            # print("link = " , link)
            # exit()
            return change(link , problem)

        for successor in problem.getSuccessors(current):
            if successor[0] in visited_path:
                continue
            # flag = True
            temp_link_element = []
            temp_link_element.append(current)
            temp_link_element.append(successor[0])
            temp_link_element.append(successor[1])

            link.append(temp_link_element)

            temp = cost
            temp = temp + successor[2]
            temp_current = successor[0]

            # print("nulherusitc = " , manhattanHeuristic(current, problem))

            # print("total_cost = " , temp + manhattanHeuristic(current , problem))
            # fringe.push(successor[0] ,temp + manhattanHeuristic(current , problem) , temp) #position , cost , priority
            # print("nulherusitc = " , heuristic(current, problem))

            # print("total_cost = " , temp + heuristic(current , problem))
            # fringe.push(successor[0] ,temp + heuristic(temp_current , problem) , temp) #position ,  priority , cost 
            item = []
            item.append(temp_current)
            item.append(temp)
            # item = [temp_current ,  temp]
            # print("heuristic = " , heuristic(temp_current , problem))
            priority = temp + heuristic(temp_current , problem)
            fringe.push(item , priority)

            # fringe.push(successor[0] ,temp + heuristic(temp_current , problem) , temp) #position ,  priority , cost 

            
            if  problem.isGoalState(successor[0]) == False:
                visited_path.append(successor[0])



# bfs = j_aStarSearch

# bfs = j_aStarSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

bfs = breadthFirstSearch
# dfs = depthFirstSearch
# astar = aStarSearch
# ucs = uniformCostSearch

# j_bfs = j_breadthFirstSearch