# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

from typing import Dict, List, Tuple, Callable, Generator, Any
import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr, pl_true

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}


#______________________________________________________________________________
# QUESTION 1

def sentence1() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** BEGIN YOUR CODE HERE ***"

    A = Expr('A')   
    B = Expr('B')   
    C = Expr('C')   
    D = Expr('D')  

    sentence1 = A | B
    sentence2 = (~ A ) % (~ B | C)
    sentence3 = disjoin(~A , ~B , C)

    ss = conjoin(sentence1 , sentence2 , sentence3)


  
    return ss


    "*** END YOUR CODE HERE ***"


def sentence2() -> Expr:
   
    A = Expr('A')   
    B = Expr('B')   
    C = Expr('C')   
    D = Expr('D')  

    sentence1 = C % (B | D)
    sentence2 = A >> (~ B & ~ D)
    sentence3 = ~(B & ~C)>>A
    sentence4 = ~D >> C

    ss = conjoin(sentence1 , sentence2 , sentence3 , sentence4)
    return ss

    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def sentence3() -> Expr:
    """Using the symbols PacmanAlive_1 PacmanAlive_0, PacmanBorn_0, and PacmanKilled_0,
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    """
    "*** BEGIN YOUR CODE HERE ***"

    PacmanAlive_0 = PropSymbolExpr('PacmanAlive' , time = 0)
    PacmanAlive_1 = PropSymbolExpr('PacmanAlive' , time = 1)
    PacmanBorn_0 = PropSymbolExpr('PacmanBorn' , time = 0)
    PacmanKilled_0 = PropSymbolExpr('PacmanKilled' , time = 0)
    
    s1 = PacmanAlive_1 % disjoin( PacmanAlive_0 & (~ PacmanKilled_0) ,  (~PacmanAlive_0) & PacmanBorn_0 ) 
    s2 = ~(PacmanAlive_0 & PacmanBorn_0)
    s3 = PacmanBorn_0
    ss = conjoin(s1 , s2 , s3)
    return ss

 
    return PacmanAlive_0

    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def findModel(sentence: Expr) -> Dict[Expr, bool]:
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """

    # A = Expr('A')   
    # B = Expr('B')   
    # C = Expr('C')   
    # D = Expr('D')  

    # sentence1 = A | B
    # sentence2 = (~ A ) % (~ B | C)
    # sentence3 = disjoin(~A , ~B , C)

    # sentence = conjoin(sentence1 , sentence2 , sentence3)

    cnf_sentence = to_cnf(sentence)
    # print(pycoSAT(cnf_sentence))

    return pycoSAT(cnf_sentence)

def findModelUnderstandingCheck() -> Dict[Expr, bool]:
    """Returns the result of findModel(Expr('a')) if lower cased expressions were allowed.
    You should not use findModel or Expr in this method.
    """
    # a = Expr('A')
    # a
    a = Expr('A')
    # dic = a.__dict__
    # a = 'a'

    # print(dic)
    
    "*** BEGIN YOUR CODE HERE ***"
    # answer = {a , True}
    # a = 1
    a.op = 'a'
    an = {a: True}
    # print("an = "  ,an)
    # print(an)
    # return dic
    # return 
    # print("a.__dict__ is:", a.__dict__) # might be helpful for getting ideas
    return an
    # util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def entails(premise: Expr, conclusion: Expr) -> bool:
    """Returns True if the premise entails the conclusion and False otherwise.
    """
    "*** BEGIN YOUR CODE HERE ***"
    
    judge = findModel(premise & ~conclusion)

    if judge == False:
        return True
    else:
        return False
    # util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

def plTrueInverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    """Returns True if the (not inverse_statement) is True given assignments and False otherwise.
    pl_true may be useful here; see logic.py for its description.
    """
    "*** BEGIN YOUR CODE HERE ***"
    judge = pl_true(inverse_statement , assignments)
    if judge == False:
        return True
    else:
        return False
        
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 2

def atLeastOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals  ist is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"
    return disjoin(literals)
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def atMostOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    itertools.combinations may be useful here.
    """
    "*** BEGIN YOUR CODE HERE ***"


    #  ans = 1
    # test = itertools.combinations(literals , len(literals) - 2)
    ans = []
    test = itertools.combinations(literals ,2)

    for i in test:
        s = ~i[0] | ~i[1]
        ans.append(s)

    return conjoin(ans)
    # ans_list = []
    # # return conjoin(test[0])
    # for c in test:
    #     # cc = disjoin(c)

    #     temp = ~c[0]
        
    #     for symbol in c:
    #         c_symbol = ~ symbol
    #         if c_symbol == temp:
    #             continue
    #         temp = disjoin( c_symbol , temp)
    #         # print("temp = " , temp)
    #     ans_list.append(temp)
        
    # ans = ans_list[0]
    # ans_list.remove(ans)

    # # test = ans_list[1]
    # # ans_list.remove(test)

    # for dis in ans_list:
    #     ans = conjoin(dis , ans)
    # print("an = " , ans)
    # return ans
      
        # print(c)
    
    return temp[0]
    # ans = temp[0]
    # temp.remove(ans)
    # print(temp)
    # for tt in temp:
    #     ans = conjoin(ans , temp[0])
    
    return  ~ ans
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def exactlyOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    atLeastOne_instance = atLeastOne(literals)
    atMostOne_instance = atMostOne(literals)
    return conjoin(atLeastOne_instance , atMostOne_instance)
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 3

def pacmanSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]=None) -> Expr:
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    now, last = time, time - 1
    possible_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t
    # the if statements give a small performance boost and are required for q4 and q5 correctness

    # print("walls_grid" , walls_grid)
    if walls_grid[x][y+1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not possible_causes:
        return None
    
    "*** BEGIN YOUR CODE HERE ***"

    # answer:List[Expr] = []

    current_move = PropSymbolExpr(pacman_str , x , y , time=now)
 
    
    return current_move % disjoin(possible_causes) 


   

def SLAMSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Expr:
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    now, last = time, time - 1
    moved_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    if walls_grid[x][y+1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not moved_causes:
        return None

    moved_causes_sent: Expr = conjoin([~PropSymbolExpr(pacman_str, x, y, time=last) , ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)])

    failed_move_causes: List[Expr] = [] # using merged variables, improves speed significantly
    auxilary_expression_definitions: List[Expr] = []
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(wall_dir_combined_literal % wall_dir_clause)

    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])

    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin([moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)

def pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
        walls_grid: 2D array of either -1/0/1 or T/F. Used only for successorAxioms.
            Do NOT use this when making possible locations for pacman to be in.
        sensorModel(t, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
        successorAxioms(t, walls_grid, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
        - Results of calling sensorModel(...), unless None.
        - Results of calling successorAxioms(...), describing how Pacman can end in various
            locations on this time step. Consider edge cases. Don't call if None.
    """
    
    # test1 = sensorModel()
    test1: List[Expr] = []
    test2: List[Expr] = []

  
    ultimate_list: List[Expr] = []

    sentence1: List[Expr] = []
    # print("all_coords = " , all_coords)
    # print("non_outer_wall_coords = " , non_outer_wall_coords)
    # walls_list = walls_grid.asList()
    for position in all_coords:
        # print("position = " , position)
        # print(position in non_outer_wall_coords)
        #this is the wall
        # if (position in non_outer_wall_coords) == False: 
            # print("dfdf")
        wall = PropSymbolExpr(wall_str , position[0] , position[1])
        position_symbol =  PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
        entail = wall >> ~position_symbol
        # sentence1.append(entail)
        ultimate_list.append(entail)

   

    all_coords_expr : List[Expr] = []
    for position in non_outer_wall_coords:
        position_symbol = PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
        all_coords_expr.append(position_symbol)

    s2 = exactlyOne(all_coords_expr)
    ultimate_list.append(s2)

    # print(sentence2)

    
    # sentence3: List[Expr] = []
    action = ['North', 'South', 'East', 'West']

    
    action_expr : List[Expr] = []
    for act in action:
        action_symbol = PropSymbolExpr(act , time = t)
        action_expr.append(action_symbol)

    s3 = exactlyOne(action_expr)
    ultimate_list.append(s3)

    # print("test = " , s3)

    # pacphysics_sentences = conjoin(s1 , s2  , s3)

    # print("s3 = " , s3)
    
    if sensorModel != None:
        test1 = sensorModel(t , non_outer_wall_coords)
        # pacphysics_sentences = conjoin(test1 , pacphysics_sentences)
        ultimate_list.append(test1)
 
    if successorAxioms != None and t:
        test2 = successorAxioms(t ,walls_grid, non_outer_wall_coords)
        # pacphysics_sentences = conjoin(test2 , pacphysics_sentences)
        ultimate_list.append(test2)



    # print("conjoin = " , ultimate_list)
    return conjoin(ultimate_list)
   
def checkLocationSatisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - action1 = to ensure match with autograder solution # what that means?
        - problem = an instance of logicAgents.LocMapProblem
    Note:
        - there's no sensorModel because we know everything about the world
        - the successorAxioms should be allLegalSuccessorAxioms where needed
    Return:
        - a model where Pacman is at (x1, y1) at time t = 1
        - a model where Pacman is not at (x1, y1) at time t = 1
    """
    # util.raiseNotDefined()

    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1

    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    # print("map_sent = " , map_sent)

    current_position = PropSymbolExpr(pacman_str , x0 , y0 , time = 0)
    possible_position = PropSymbolExpr(pacman_str , x1 , y1 , time = 1)
    # print("action0 = " , action0)
    # print("action1 = " , action1)

    action0_symbol = PropSymbolExpr(action0 , time = 0)
    action1_symbol = PropSymbolExpr(action1 , time = 1)  #???
    
    physical_move_0 = pacphysicsAxioms(0, all_coords, non_outer_wall_coords)  #time = 0? or use other function to change?
    physical_move_1 = pacphysicsAxioms(1, all_coords, non_outer_wall_coords , walls_grid , None , allLegalSuccessorAxioms)  #time = 0? or use other function to change?

    # exit()
    KB.append(conjoin(map_sent)) #why add this one?
    KB.append(physical_move_0)
    KB.append(physical_move_1)
    KB.append(action0_symbol)
    KB.append(action1_symbol)

    KB.append(current_position)
    # KB.append(possible_position)

    "*** BEGIN YOUR CODE HERE ***"

    # print("physical_move = " , physical_move)
    KB1_sentence = conjoin(KB + [possible_position])
    model1 = findModel(KB1_sentence)

    # KB.remove(possible_position)
    # KB.append(~possible_position)

    KB2_sentence = conjoin(KB + [~possible_position])
    
    # KB2_sentence = conjoin(KB)
    model2 = findModel(KB2_sentence)

    

    # return []

    # print("model1 = " , model1)
    # model2 = findModel(~KB_sentence)
    # print("model2 = " , model2)
    # print("sdfsdfg")
    return model1 , model2

    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 4

def positionLogicPlan(problem) -> List:
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls_grid = problem.walls
    # print("walls_grid = " , walls_grid)
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls_grid.asList()
    # print("walls_list = " , walls_list)

    x0, y0 = problem.startState
    xg, yg = problem.goal
    
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), 
            range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []

    init_position = PropSymbolExpr(pacman_str , x0 , y0 , time = 0)
    KB.append(init_position)

    for t in range(50):
        print("t = " , t)

        position_list : List[Expr] = []
        for position in non_wall_coords:
  
            position_symbol = PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
            position_list.append(position_symbol)

        s0 = exactlyOne(position_list)
        KB.append(s0)

        # print("model =  " , model)
        goal_assertion = PropSymbolExpr(pacman_str , xg , yg , time = t)
        # KB.append(goal_assertion)
        KB_sentence = conjoin(KB + [goal_assertion])

        model = findModel(KB_sentence)
        print("model= " , model)

        if model == False:

            action_expr : List[Expr] = []
            for act in actions:
                action_symbol = PropSymbolExpr(act , time = t)
                action_expr.append(action_symbol)

            s3 = exactlyOne(action_expr)
            KB.append(s3)
            for position in non_wall_coords:

                # if t == 0:
                #     continue
                KB.append(pacmanSuccessorAxiomSingle(position[0] , position[1] , t + 1  , walls_grid))
            continue


        print("action = " , extractActionSequence(model , actions))
        return extractActionSequence(model , actions)
        
            
    assert False
    return None   # could not find the optimize solution....

    "*** BEGIN YOUR CODE HERE ***"
    # util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 5

def foodLogicPlan(problem) -> List:
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]

    KB = []

    "*** BEGIN YOUR CODE HERE ***"

    init_position = PropSymbolExpr(pacman_str , x0 , y0 , time = 0)
    KB.append(init_position)

    for food_point in food:
            # print(food_point)
        food_symbol = PropSymbolExpr(food_str , food_point[0] , food_point[1] , time = 0)
        KB.append(food_symbol)

    for t in range(50):
        print("t = " , t)

        position_list : List[Expr] = []

        for position in non_wall_coords:
  
            position_symbol = PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
            position_list.append(position_symbol)
        

        food_list : List[Expr] = []

        s0 = exactlyOne(position_list)

        KB.append(s0)
        # KB.append(sf)

        goal_list : List[Expr] = []

        for food_point in food:
            food_symbol = ~ PropSymbolExpr(food_str , food_point[0] , food_point[1] , time = t)
            goal_list.append(food_symbol)

        # goal_assertion = disjoin(food_list)

        KB_sentence = conjoin(KB + goal_list)
        model = findModel(KB_sentence)

        print("model= " , model)

        if model == False:

            action_expr : List[Expr] = []

            for act in actions:
                action_symbol = PropSymbolExpr(act , time = t)
                action_expr.append(action_symbol)

            s3 = exactlyOne(action_expr)
            KB.append(s3)

            for position in food:
                temp_symbol = PropSymbolExpr(food_str , position[0] , position[1] , time = t + 1) % conjoin( PropSymbolExpr(food_str , position[0] , position[1] , time = t) 
                                                    , ~PropSymbolExpr(pacman_str , position[0] , position[1] , time = t) )
                KB.append(temp_symbol)

            for position in non_wall_coords:

                KB.append(pacmanSuccessorAxiomSingle(position[0] , position[1] , t + 1  , walls))

            continue


        print("action = " , extractActionSequence(model , actions))
        return extractActionSequence(model , actions)
        
            
    assert False
    util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 6

def localization(problem, agent) -> Generator:
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    KB = []



    for i in all_coords:
        if i in walls_list:
            symbol = PropSymbolExpr(wall_str , i[0] , i[1])
            KB.append(symbol)
        else:
            symbol = ~PropSymbolExpr(wall_str , i[0] , i[1])
            KB.append(symbol)



    # KB.append

    for t in range(agent.num_timesteps):
        "*** END YOUR CODE HERE ***"
        physical_move_1 = pacphysicsAxioms(t, all_coords, non_outer_wall_coords , walls_grid , sensorAxioms , allLegalSuccessorAxioms)  #time = 0? or use other function to change?

        KB.append(physical_move_1)
        KB.append(PropSymbolExpr(agent.actions[t] , time = t)) #??
        KB.append(fourBitPerceptRules(t , agent.getPercepts()))


        position_location = []
        for position in non_outer_wall_coords:

            judge1 = entails(conjoin(KB) , PropSymbolExpr(pacman_str , position[0] , position[1] , time = t))
            judge2 = entails(conjoin(KB) , ~PropSymbolExpr(pacman_str , position[0] , position[1] , time = t))
            # KB_sentence1 = conjoin(KB) >> PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
            # model1 = findModel(KB_sentence1)
            # KB_sentence2 = conjoin(KB) >> ~PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
            # model2 = findModel(KB_sentence2)

            if judge1 :
                # print("asdfasdf")
                pp1 = PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
                position_location.append(position)
                KB.append(pp1)

            elif judge2 :
                # print("zxcvzxcv")
                pp2 = ~PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
                KB.append(pp2)

            else:
                position_location.append(position)


              

        agent.moveToNextState(agent.actions[t]) # ???
        yield position_location #??

#______________________________________________________________________________
# QUESTION 7

def mapping(problem, agent) -> Generator:
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # print("know_map = " ,known_map)

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    # util.raiseNotDefined()

    init_positon_sym = PropSymbolExpr(pacman_str , pac_x_0 , pac_y_0 , time = 0)
    KB.append(init_positon_sym)

    # for i in all_coords:
    #     if i in non_outer_wall_coords:
    #         symbol = ~PropSymbolExpr(wall_str , i[0] , i[1])
    #         KB.append(symbol)
    #     else:
    #         symbol = PropSymbolExpr(wall_str , i[0] , i[1])
    #         KB.append(symbol)

    test_symbol = ~PropSymbolExpr(wall_str , pac_x_0 , pac_y_0 )
    KB.append(test_symbol)

    for t in range(agent.num_timesteps):
        "*** END YOUR CODE HERE ***"
        physical_move_1 = pacphysicsAxioms(t, all_coords, non_outer_wall_coords , known_map , sensorAxioms , allLegalSuccessorAxioms)  #time = 0? or use other function to change?

        KB.append(physical_move_1)
        KB.append(PropSymbolExpr(agent.actions[t] , time = t)) #??
        KB.append(fourBitPerceptRules(t , agent.getPercepts()))

        # print("non = " , non_outer_wall_coords)
        for position in non_outer_wall_coords:
            
            judge1 = entails(conjoin(KB) , PropSymbolExpr(wall_str , position[0] , position[1] ))
            judge2 = entails(conjoin(KB) , ~PropSymbolExpr(wall_str, position[0] , position[1] ))
      
            if judge1 :
                # print("asdfasdf")
                pp1 = PropSymbolExpr(wall_str , position[0] , position[1] )
                # known_map.append(position)
                # print(position[0] , position[1]) 
                known_map[position[0]][position[1]] = 1
                KB.append(pp1)

            if judge2 :
                # print("zxcvzxcv")
                pp2 = ~PropSymbolExpr(wall_str , position[0] , position[1])
                known_map[position[0]][position[1]] = 0
                KB.append(pp2)
            # else:
            #     print("sdfasdf")
        


        agent.moveToNextState(agent.actions[t]) # ???

        yield known_map

#______________________________________________________________________________
# QUESTION 8

def slam(problem, agent) -> Generator:
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    # util.raiseNotDefined()


    init_positon_sym = PropSymbolExpr(pacman_str , pac_x_0 , pac_y_0 , time = 0)
    KB.append(init_positon_sym)

    wall_symbol = ~PropSymbolExpr(wall_str , pac_x_0 , pac_y_0 )
    KB.append(wall_symbol)

    known_map[pac_x_0][pac_y_0] = 0

    for t in range(agent.num_timesteps):
        "*** END YOUR CODE HERE ***"
        physical_move_1 = pacphysicsAxioms(t, all_coords, non_outer_wall_coords , known_map , SLAMSensorAxioms , SLAMSuccessorAxioms)  #time = 0? or use other function to change?

        KB.append(physical_move_1)
        KB.append(PropSymbolExpr(agent.actions[t] , time = t)) #??
        KB.append(numAdjWallsPerceptRules(t , agent.getPercepts()))

        for position in non_outer_wall_coords:
            
            judge1 = entails(conjoin(KB) , PropSymbolExpr(wall_str , position[0] , position[1] ))
            judge2 = entails(conjoin(KB) , ~PropSymbolExpr(wall_str, position[0] , position[1] ))
      
            if judge1:
                # print("asdfasdf")
                pp1 = PropSymbolExpr(wall_str , position[0] , position[1] )
                # known_map.append(position)
                # print(position[0] , position[1]) 
                known_map[position[0]][position[1]] = 1
                KB.append(pp1)

            if judge2:
                # print("zxcvzxcv")
                pp2 = ~PropSymbolExpr(wall_str , position[0] , position[1])
                known_map[position[0]][position[1]] = 0
                KB.append(pp2)
        

        position_location = []
        for position in non_outer_wall_coords:

            judge1 = entails(conjoin(KB) , PropSymbolExpr(pacman_str , position[0] , position[1] , time = t))
            judge2 = entails(conjoin(KB) , ~PropSymbolExpr(pacman_str , position[0] , position[1] , time = t))
            # KB_sentence1 = conjoin(KB) >> PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
            # model1 = findModel(KB_sentence1)
            # KB_sentence2 = conjoin(KB) >> ~PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
            # model2 = findModel(KB_sentence2)

            if judge1 :
                # print("asdfasdf")
                pp1 = PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
                position_location.append(position)
                KB.append(pp1)

            elif judge2 :
                # print("zxcvzxcv")
                pp2 = ~PropSymbolExpr(pacman_str , position[0] , position[1] , time = t)
                KB.append(pp2)

            else:
                position_location.append(position)


        # print(known_map , position_location)
        agent.moveToNextState(agent.actions[t]) # ???

        yield (known_map, position_location)


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

#______________________________________________________________________________
# Important expression generating functions, useful to read for understanding of this project.


def sensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time = t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def fourBitPerceptRules(t: int, percepts: List) -> Expr:
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], time=t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def numAdjWallsPerceptRules(t: int, percepts: List) -> Expr:
    """
    SLAM uses a weaker numAdjWallsPerceptRules sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, time=t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = SLAMSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)

#______________________________________________________________________________
# Various useful functions, are not needed for completing the project but may be useful for debugging


def modelToString(model: Dict[Expr, bool]) -> str:
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, _, time = parsed
            plan[time] = action
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


# Helpful Debug Method
def visualizeCoords(coords_list, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualizeBoolArray(bool_arr, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()
