# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp: mdp.MarkovDecisionProcess, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        """
          Run the value iteration algorithm. Note that in standard
          value iteration, V_k+1(...) depends on V_k(...)'s.
        """
        "*** YOUR CODE HERE ***"
        # print("getStates() = " , self.mdp.getStates()[0])
        # state = self.mdp.getStates()[0]
        # print("mdp.getPossibleActions(state) = " , self.mdp.getPossibleActions(state))

        # print("self.values = " , self.values)

        for i in range(0 , self.iterations):

            temp_dictionay = {}
            for state in self.mdp.getStates():
                # print("self.mdp.getStates()" , self.mdp.getStates())
                # print("self.values = " , self.values)
                
                value = []
                # print("self.mdp.getPossibleActions(state)" , self.mdp.getPossibleActions(state))

                # if self.mdp.isTerminal(state):
                #     continue
                
                for action in self.mdp.getPossibleActions(state):
                    temp = self.computeQValueFromValues( state, action)
                    # print("temp = " , temp)
                    value.append(temp)

                if value == []:
                    temp_dictionay[state] = self.values[state]
                    # self.values[state] = max(value)
                    continue

                temp_dictionay[state] = max(value)

            
            self.values = temp_dictionay.copy()
                # print("max_vlaue = " , max(value))
                # print("asdfasdfsa")
               


        #         # exit()
                



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # print("mdp.getTransitionStatesAndProbs(state, action) = " , self.mdp.getTransitionStatesAndProbs(state, action))

        q_value = 0
        for tuple in self.mdp.getTransitionStatesAndProbs(state , action):
            nextstate = tuple[0]
            transition_value = tuple[1]
            # print("transition value = " , transition_value)
            # print(self.mdp.getReward(state , action , nextstate))
            # print(self.values[nextstate])
            # print(self.discount)
            q_value += transition_value * (self.mdp.getReward(state , action , nextstate) + self.discount * self.values[nextstate])

        
        # print("mdp.getReward(state, action, nextState) = " , mdp.getReward(state, action, nextState))
        # q_value = 
        return q_value
 

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        
        ll = []
        for action in self.mdp.getPossibleActions(state):
            ll.append(self.computeQValueFromValues(state , action))

        max_value = max(ll)  # Find the maximum value in the list
        max_index = ll.index(max_value)
        return self.mdp.getPossibleActions(state)[max_index]
        # else:
        #     max_key = max(self.values, key=self.values.get)
        #     print(max_key)
        
        return max_key
        # print("self.values  = " , self.values)
        # print("self.action = " , self.getAction(state))
        # exit()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


