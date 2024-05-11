# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
from backend import ReplayMemory

import model
import backend
import gridworld


import random,util,math
import numpy as np
import copy

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent
      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update
      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)
      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.qVals = {}
        self.eval = False


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if ((state , action) in self.qVals) == True:
            return self.qVals[(state , action)]
        else:
            for aa in self.getLegalActions(state):
                self.qVals[(state,aa)] = 0
            return 0



    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"

        # temp = []
        # for keys in self.qVals.keys():
        #     if keys[0] == state:
        #         temp.append(self.qVals[keys])
       
        # if len (temp) == 0:
        #     return 0
        
        # else:
        #     return max(temp)
        
        temp = []
        for action in self.getLegalActions(state):
            temp.append(self.getQValue(state , action))
        
        if len (temp) == 0:
            return 0
        
        else:
            return max(temp)

        

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"


        # value = self.computeValueFromQValues(state)
        
        ll = []

        actions = self.getLegalActions(state)
        if len(actions) == 0:
            return None
        else:
            # for key, evalue in self.qVals.items():
            #     if evalue == value:
            #         ll.append(key[1])
            best = -1e10
            

            for action in actions:
                candidate = self.getQValue(state , action)
                if candidate > best:
                    best = candidate
                    ll = [action]
                elif candidate == best:
                    ll.append (action)

        
            return random.choice(ll)

        # if len(self.qVals) == 0:
        #     return None
        # else:
        #     return max(self.qVals, key=self.qVals.get)[1]
        

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.
          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"

        best_action = self.computeActionFromQValues(state)
        judge = util.flipCoin(self.epsilon)
        if judge == True:
          if legalActions != []:
              return random.choice(legalActions)
          else:
              return action
        
        else:
              return best_action

        


    def update(self, state, action, nextState, reward: float):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here
          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # print(action)

        # if state in self.qVals:
        # best_value = max(self.)
        q_value = (1 - self.alpha) * self.getQValue(state,action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))
        
        self.qVals[(state , action)] = q_value
        # return q_value


            
 

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1
        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action

class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent
       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        # print("self.weights = " , self.getWeights())
        # print("self.featExtractor = " , self.featExtractor.getFeatures(state , action))
        # exit()
        # sum = 0
        # for key in self.featExtractor.getFeatures(state , action).keys():

        #     sum = sum + self.featExtractor.getFeatures(state , action)[key] * self.weights[key]

        # # print("sum = " , sum)
        sum = self.featExtractor.getFeatures(state , action) * self.weights
        return sum


      
    def update(self, state, action, nextState, reward: float):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        # self.weights = 
        feature = self.featExtractor.getFeatures(state , action)

        difference = reward  + self.discount * self.computeValueFromQValues(nextState) - self.getQValue(state,action)
        # self.weights = self.weights + 
        # print(self.weights)
        for key in feature:
            current_value = self.weights[key]
            # feature = self.featExtractor.getFeatures(state , action)
            new_value = current_value + self.alpha * difference * feature[key]
            self.weights[key] = new_value

            print("current_value = " , current_value , "feature " , feature , "new_value = " , new_value , "selg.weights = " , self.weights)

    def final(self, state):
        """Called at the end of each game."""
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            # print("weight = " , self.weight)
            pass
