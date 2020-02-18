from Block_architect_MAIN import Evaluate_Build_Plan
import random
import csv
import os
from pathlib import Path
from abc import abstractmethod
import keras.layers as Kl
import keras.models as Km
import numpy as np
import matplotlib.pyplot as plt

#In this module, the Deep Q-learning takes place, an agent is created, i.e. the system which iterates through the
#build_plan pivot blocks and for every episode changes the current position by 10*x pixels.
#Upon this, the reward is given, and the value of the state then assessed.

#Define environment, i.e. standard_build_plan

#Determine the observation space, num. states

# Determine the action space, num. actions


#Main class
class DQN_Architect:
    def __init__(self, exploration_value = 1, planet = 1):
        self.State = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Previous_State = None
        self.Game_over = False
        self.Max_no_moves = 20  # max no moves in one game
        self.Moves_played = 0
        self.Exploration_factor = exploration_value
        self.Planet = planet
        self.Epsilon = 0.1
        self.Alpha = 0.5

    def play_to_learn(self, episodes):

        for i in range(episodes):
            print('Episode number: ' + str(i))

            while self.Game_over == False:
                self.State = self.play_move(learn=True)

                if self.Game_over == True:
                    break

            self.all_count = i
            self.init_sim()

        self.save_values()

    def init_sim(self):
        self.State = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Moves_played = 0

    def save_values(self):
        pass

    def play_move(self, learn=False):
        self.Moves_played += 1

        if learn is True:
            new_state = self.make_move_and_learn(self.State)

        else:
            new_state = self.make_move(self.State)

        if self.Moves_played >= self.Max_no_moves:
            self.Game_over = True

        return new_state

    def make_move_and_learn(self, state):
        self.learn_state(state)  # currently not yet functional

        return self.make_move(state)


    def make_move(self, state):
        self.State = state

        if self.Game_over == True:
            new_state = state
            return new_state

        p = random.uniform(0, 1)
        if p < self.Exploration_factor:
            new_state = self.make_optimal_move(state)
        else:
            # Pick a random element and tweak it
            new_state = state_scrambler(state)
        Evaluated = Evaluate_Build_Plan(new_state, self.Planet, cost=0.1)
        print("S: ", new_state, " // Evaluated at: ", Evaluated)
        return new_state

    def make_optimal_move(self, state):
        # make the optimal move in this state, i.e. the one with highest next state value
        new_state = state  # needs to be replaced by the actual optimal move!

        return new_state

    def learn_state(self, state):
        pass


def state_scrambler(state):
    #pick index between 0-18
    idx = round(random.uniform(0, 1)*18)
    # tweak by max 4, i.e. allow for -2,-1,0,1,2 as element tweak
    tweak = round(random.uniform(0, 1)*4) - 2

    new_state = state
    new_state[idx] = new_state[idx] + tweak

    state_tweak_confine(new_state)

    return new_state

def state_tweak_confine(state):
    checked_state = [20 if i>= 20 else -20 if i<= -20 else i for i in state]
    return checked_state

if __name__ == '__main__':
    Earth = DQN_Architect(exploration_value = 0.05, planet = 1)
    Earth.play_to_learn(10)

# TODO: LEARNING: When formatting states (the state tweaks) to NN, then [-20, 20] integers, divide by 20 to normalize between [-1, 1]
# and in case I have to explode back, then multiply by 20 and round.




#TODO: Look at the info below, and implement features.
#TODO: In Block Architect, specify diffenent material types with different densities and costs, colours.
#TODO: Start on PPT presentation.
#TODO: LEARNING: there, save learned from one set of # episodes (i.e. one learning step), as well as save the values
#TODO: Using Tensorflow/Keras,
#Define the DQ_Agent !!!!!!!!!!!!! ALREADY THE DQN ARCHITECT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class DQ_Agent:
    #Initialize

    #self.state_handling_size

    #self.action_handling_size
    #Optimizer = Adam

    #Discount and Exploration Values gamma, epsilon.
    # Exploration value. It is a try and except code-block. If the exploration state is selected, then it has a
    #certain greater margin of freedom for displacing the current positional set of the blocks, in this way it
    #should keep them coordinated. [range + expl_x: range + expl_x]
    #Otherwise, just go with standard action choices [range:range] of movement possibilities.
    #RESTRICTIONS: base_blocks (~.contact_base): only change x coordinates 10*x px.
    #              structure_blocks (~.contact): change x,y coordinates 10*x px.
    #              top_block(~.contact): change y coordinates 10*x px

    #Perhaps, add friction such that; if self.contact_base or self.contact:
    #                                       friction = 10 (instead of standard friction "1")
    #                                       simulate concrete bonding, e.g. "glue"

    #Compile the model

    #Define what an action is

    #Agent Training
    pass


    #100 learning steps, learning steps are iterations of episodes.
    #1 learning step is  100 episodes. After each learning step, it takes a picture of the current construction state
    #and saves the progress, (csv???). Then, the learning step is loaded as the default into the next learning step, and
    #the program goes from there.
    def save_learning_step():
        pass

    def load_learning_step():
        pass

    pass