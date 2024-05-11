def k_breadthFirstSearch(problem: SearchProblem):
    
    visited_path_list = []
    link_list = []
    current = problem.getStartState()

    length = len(problem.corners)

    temp_corner = list(problem.corners)

    corner_step = [] 
    # temp_corner.copy(problem.corners)
    for i in range(0 , length + 1):

        fringe = Queue()# store all the path
        visited_path = []
        # path_stack = Stack()
        # print(problem.getStartState())
        # print(problem.corners)

        fringe.push(current)
        visited_path.append(current)

        link = []
        # path_stack.push(current)
        if len(temp_corner) == 0:
            #  p
            return j_change(link_list ,problem, corner_step)

        while 1:

            current = fringe.pop()
            # print("problem_corner = " , problem.corners)

            # visited_path.append(current)
            # print("before_problem_corner = " , problem.corners)

            if current in temp_corner:
                # return [n , n]
                # print("visited_path = " , visited_path)
                # print("link = " , link)
                # print("current = " , current)

                corner_step.append(current)

                temp_corner.remove(current)
                link_list.append(link)
                print("current = " , current)
                print("after_problem_corner = " , temp_corner)
                print(problem.corners)
                print(len(problem.corners))
                break
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
                if  problem.isGoalState(successor[0]) == False:
                    visited_path.append(successor[0])