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

import util, copy

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getInitialState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isFinalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getNextStates(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getActionCost(self, actions):
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
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    
    print("Start:", problem.getInitialState())
    print("Is the start a goal?", problem.isFinalState(problem.getInitialState()))
    print("Start's successors:", problem.getNextStates(problem.getInitialState()))
    """
    "*** YOUR CODE HERE ***"
    
        
    stack=util.Stack()
    visited = []
    path = []
    
    # If initial state is solution
    if problem.isFinalState(problem.getInitialState()):
        return []
    
    # Push start state
    stack.enqueue((problem.getInitialState(),[]) )
    
    # If no solution is found return []
    while not stack.isEmpty():
        node, path = stack.dequeue()
        
        # If node is not visited, get next state and push in stack
        if node not in visited:               
            visited.append(node)
            
            # If solution found
            if problem.isFinalState(node):
                return path
            
            # Get next state
            states = problem.getNextStates(node)
            for i in states:
                if i[0] not in visited:
                    stack.enqueue((i[0],path+[i[1]]))
                               
    return []

    util.raiseNotDefined()

    
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    fifo = util.Queue()
    visited = [] 
    path = [] 

    # If initial state is goal state #
    if problem.isFinalState(problem.getInitialState()):
        return []

    # Push initial position
    fifo.enqueue((problem.getInitialState(),[]))

    # If no solution is found 
    while not fifo.isEmpty():

        # Get informations of current state #
        node, path = fifo.dequeue() # Take position and path
        visited.append(node)

        # While solution is found
        if problem.isFinalState(node):
            return path

        # Get next state
        states = problem.getNextStates(node)

        # Add new states 
        for item in states:
            if item[0] not in visited and item[0] not in [state[0] for state in fifo.list]:

                    newPath = path + [item[1]]   # Update path
                    fifo.enqueue((item[0], newPath))
                    
    return [] 
    
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    frontier = util.PriorityQueue()
    start = (problem.getInitialState(), "", 0)
    frontier.enqueue([start], 0)                    # Add initial node
    visited_state = {problem.getInitialState(): 0}
    visited = set()
    actions = []
    
    # If initial state is goal state #
    if problem.isFinalState(problem.getInitialState()):
        return []

    # While no solution is found
    while not frontier.isEmpty():
        path = frontier.dequeue()
        node = path[-1]

        # Check if current node has been visited or current path has a lower cost
        if node[0] not in visited or node[2] <= visited_state[node[0]]:
            
            if problem.isFinalState(node[0]):
                
                for state in path[1:]:
                    actions.append(state[1])
                return actions

            # Gen next state
            states = problem.getNextStates(node[0])   
            for i in states:
                # Check in next state hasn't been visited or if new path has a lower cost
                if i[0] not in visited_state or (node[2] + i[2]) < visited_state[i[0]]:
                    visited_state[i[0]] = node[2] + i[2]        # Update cost
                    newNode = copy.deepcopy(path)               # Create copy of current path
                    newSucc = (i[0], i[1], node[2] + i[2])      # New succesor node
                    newNode.append(newSucc)                     # Append new node to path
                    frontier.enqueue(newNode, node[2] + i[2])   # Add new path with priority
        
        # Add node to visited
        visited.add(node[0])

    return []

    util.raiseNotDefined()
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    frontier = util.PriorityQueue()
    visited = []
    actionList = []
    
    # If initial state is goal state 
    if problem.isFinalState(problem.getInitialState()):
        return []
    
    # Push initial state
    frontier.enqueue((problem.getInitialState(), actionList), heuristic(problem.getInitialState(), problem))
    
    while frontier:
        node, actions = frontier.dequeue()
        
        if not node in visited:
            visited.append(node)
            
            if problem.isFinalState(node):
                return actions
            
            for successor in problem.getNextStates(node):
                coordinate, direction, cost = successor
                nextActions = actions + [direction]
                nextCost = problem.getActionCost(nextActions) + \
            heuristic(coordinate, problem)
                frontier.enqueue((coordinate, nextActions), nextCost)
    return []




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch