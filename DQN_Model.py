#TODO: don't go back to a previous state
#In this module, the Deep Q-learning takes place, an agent is created, i.e. the system which iterates through the
#build_plan pivot blocks and for every episode changes the current position by 10*x pixels.
#Upon this, the reward is given, and the value of the state then assessed.

from Block_architect_MAIN import Evaluate_Build_Plan, det_standard_plan_cost, Check_Valid_Build_Plan
import random
import csv
import os
import pygame
import time
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import numpy as np
from Block_architect_MAIN import Evaluate_Build_Plan, det_standard_plan_cost
from pathlib import Path
pygame.init()


class DQN_Architect:
    def __init__(self, exploration_value = 1, planet = 1, drawing= True):
        self.Initial_State = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        # self.Initial_State = (-3, 0, 0, 0, -1, 0, 0, 5, -7, 0, 0, 12, -15, 0, -2, 0, -17, 0, -2)
        self.Drawing = drawing
        self.State = self.Initial_State
        self.Previous_State = self.Initial_State
        self.Game_over = False
        self.Max_no_moves = 50  # max no moves in one game (for testing use ~ = 1)
        self.Moves_played = 0
        self.Exploration_factor = exploration_value # epsilon exploratory factor
        self.Planet = planet
        self.Reward = None
        self.Alpha = 0.1 # learning rate
        self.Gamma = 0.9 # discount
        self.value_model = self.load_model()
        self.Default_cost = 0
        self.Rewards_Dict = dict()
        self.MAX_Reward_in_episode = 0
        self.Reward_sum_in_episode = 0
        self.State_with_MAX_Reward = list()
        # self.Experience_replay = deque(maxlen=2000)
        if self.Planet == 1:  # Earth
            self.Default_cost = 0.1
        elif self.Planet == 2: # Moon
            self.Default_cost = 0.2
        elif self.Planet == 3: # Mars
            self.Default_cost = 0.25

    def play_to_learn(self, episodes): # One Episode continues until Game Over! Then next episode

        # Get standard mass cost as a reference for evaluating.
        ref_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Standard_mass_cost = det_standard_plan_cost(ref_state, self.Planet, cost=self.Default_cost, drawing = self.Drawing)
        print("Standard Mass Cost: ", self.Standard_mass_cost)
        pygame.init()
        for i in range(episodes):
            self.Reward_sum_in_episode = 0
            self.MAX_Reward_in_episode = 0
            self.State_with_MAX_Reward = []
            # pygame.init()
            self.init_sim() # back to setup
            self.Game_over = False
            print('**********************************************')
            print('******** Episode number: ' + str(i) + '*******')
            print('***********************************************')

            while self.Game_over == False:
                #self.State = self.play_move(learn=True) # ??? why even send back the state here?
                self.Moves_played += 1
                print('Move no: ' + str(self.Moves_played))
                if self.Moves_played >= self.Max_no_moves:
                    self.Game_over = True
                else:
                    self.Q_choose_move_learn_state_and_execute_move()
                    self.Reward_sum_in_episode += self.Reward
                    if self.Reward > self.MAX_Reward_in_episode:
                        self.MAX_Reward_in_episode = self.Reward
                        self.State_with_MAX_Reward = list(self.State)

            localtime = time.localtime(time.time())
            self.Rewards_Dict[localtime] = (self.Reward_sum_in_episode, self.MAX_Reward_in_episode, self.State_with_MAX_Reward)
            print(">>> Total Reward in Episode ", i, "was: ", self.Reward_sum_in_episode)
            print(">>> MAX Reward in Episode ", i, "was: ", self.MAX_Reward_in_episode, "in State: ", self.State_with_MAX_Reward)

            self.all_count = i
            if i % 100 == 0:
                self.save_model()
                self.save_rewards()
                self.Rewards_Dict = dict() # refresh
        pygame.quit()

        self.save_model()
        self.save_rewards()
        self.Rewards_Dict = dict() # refresh

    def init_sim(self):
        self.State = self.Initial_State
        self.Previous_State = self.Initial_State

        self.Moves_played = 0
        self.Reward = None

    def save_rewards(self):
        rewards_saved = 'rewards_DQN' + str(self.Planet) + '.csv'
        a = csv.writer(open(rewards_saved, 'a', newline=''))

        for v, k in self.Rewards_Dict.items():
            a.writerow([v, k])
        pass

    # Only need to load when graphing reward_sum in an episode.
    def load_rewards(self):
        load_saved = 'rewards_DQN' + str(self.Planet) + '.csv'

        try:
            rewards_csv = csv.reader(open(load_saved, 'r'))
            for row in rewards_csv:
                k, v = row
                self.Rewards_Dict[k] = float(v)

        except:
            pass

    def Q_choose_move_learn_state_and_execute_move(self):
        # select action,
        selected_next_action = self.select_action()

        temp_state = list(self.State)[:]
        # temp_state now is set to the next state after taking the proposed action
        temp_state[selected_next_action[0]] = temp_state[selected_next_action[0]] + selected_next_action[1]
        # Display the

        # had to convert state tuple to list for build plan evaluation and tweaking
        try:
            Evaluated = Evaluate_Build_Plan(temp_state, self.Planet, cost=self.Default_cost, ref_house_cost = self.Standard_mass_cost, drawing=self.Drawing)

        except:
            self.Game_over = True
            self.Reward = 0
            print(":::: I think my circuits are fried ::::")

        self.Game_over = Evaluated[1]
        self.Reward = Evaluated[0]

        # Get ALL Q values starting from temp_state (i.e. next_state)
        # We want to find MAXa_Q that is starting in the next state.
        # Better MAXa_Q by sampling some next_moves and "average??" instead of predict here.
        if self.Game_over == False:

            # Predicted Q of state to selected action
            predictQs = self.value_model.predict(state_to_array(self.State))[0]
            predictQs_2 = self.value_model.predict(state_to_array(temp_state))[0]

            # Calculate target based on estimations/predictions of
            target = np.array((1-self.Alpha) * predictQs + self.Alpha * (predictQs_2 + self.Reward))
            # target = np.array(self.Reward + self.Gamma * MAXa_Q)
            # target = (1-alpha) * predictQ(s) + alpha * (predictQ(s') + reward)

            print("Temporary Next State: ", temp_state)
            print("Q TARGET:", target, "-_-_- Reward -_-_-: ", self.Reward)

        else:
            target = np.array([self.Reward])
            print("Temporary Next State: ", temp_state)
            print("-_-_- Reward for Game Over-_-_-: ", self.Reward)

        # Update Model to learn from new temp state to target
        sta = state_to_array(self.State)
        self.train_model(state=sta, target=target, epochs=20)

        # State is updated finally to state from new action
        self.State = tuple(temp_state)

    def select_action(self): # My Policy select_OPTIMAL action at self.State. if selected_action goes to (0,0,0,...) scramble.
        # actions have the form  (index, tweak)
        current_state_before_move = self.State
        p = random.uniform(0, 1) # maybe change to non uniform
        # p chosen above the exploration factor -> choose a random move

    # WHILE LOOP over if/else: While
        need_to_scramble = False
        while True:
            if p < self.Exploration_factor and need_to_scramble == False:
                selected_next_action = self.make_optimal_move(current_state_before_move)

            else:
                # Pick a random element and tweak it
                selected_next_action = state_scrambler(current_state_before_move)

            # If current.State acting selected_next_action results in self.Initial_State, continue. else return selected_next_action.
            # Hypothetical next state =
            temp_state = list(self.State)[:]
            temp_state[selected_next_action[0]] = temp_state[selected_next_action[0]] + selected_next_action[1]
            # temp_state now is set to the next state after taking the proposed action

            if temp_state == self.Initial_State or Check_Valid_Build_Plan(temp_state) == False:
                need_to_scramble = True
                print("CAUTION: I need to scramble not to revert to initial STATE or do invalid architecture.")
                continue

            else:
                #print("ALL GOOD")
                return selected_next_action

    def make_optimal_move(self, state):
        # make the optimal move in this state, i.e.
        # Policy: among possible actions pick the one with highest next Q-value if exists, otherwise pick random
        possible_actions = []
        for index in range(len(state)):
            # For i in the list of possible state updates per move:
            # Should I have i go through all actions, then storing all actions at once, or rather of ONE random action? _____QUESTION_____
            #What about exploring in that case?
            for tweak in [-2, -1, 1, 2]:
                # To truly create a DEEP copy of a list, need to slice ([:])
                MOVE_temp_state = list(state)[:]
                update = state[index] + tweak
                MOVE_temp_state[index] = update
                MOVE_temp_state_tuple = tuple(MOVE_temp_state)

                #Maybe have reward = -10, since that is invalid procedure?
                if update < -20 or update > 20 or Check_Valid_Build_Plan(MOVE_temp_state) == False:
                    pass
                else:
                    possible_actions.append((index, tweak))

        value = -float('Inf')
        best_action_list = []

        for index in range(len(possible_actions)):
            calc_val = 0
            candidate_state = list(self.State)[:]
            candidate_action = possible_actions[index]
            candidate_state[candidate_action[0]] = candidate_state[candidate_action[0]] + candidate_action[1]
            candidate_state_array = state_to_array(candidate_state)
            predict_result = self.value_model.predict(candidate_state_array)
            calc_val = predict_result[0]
            # print("______PREDICTED VALUE_____: ", calc_val)
            # calc_val = self.det_value(state, candidate_action)
            # print("possible actions: ",len(possible_actions), "calculated value from 1 poss. action: ",calc_val )
            if calc_val > value:
                best_action_list = [candidate_action]
                value = calc_val
            elif calc_val == value:
                best_action_list.append(candidate_action)
            # What if all moves lead to invalid build plans?

        # If best action list empty, then force exploration

        if len(best_action_list) > 0:
            optimal_action = random.choice(best_action_list)
        else:
            optimal_action = random.choice(possible_actions)

        # print("Calculated Value is ", value, " for the Optimal Action: ", optimal_action)
        return optimal_action

    # def det_value(self, state, candidate_action):
    #     value = -float('Inf')
    #     if (state, candidate_action) in self.Values.keys():
    #         value = self.Values[(state, candidate_action)]
    #     return value

# ---------------------------------- DQN_MODEL -----------------------------------------------------------------------#

    # def replay(self, action):
    #     self.Experience_replay.append((self.State, action, self.Game_over))


    def save_model(self):
        current_save = 'model_values' + str(self.Planet) + '.h5'
        try:
            os.remove(current_save)
        except:
            pass
        self.value_model.save(current_save)

    def load_model(self):
        current_save = 'model_values' + str(self.Planet) + '.h5'
        model_file = Path(current_save)
        if model_file.is_file():
            model = load_model(current_save)
            print('load available model: ' + current_save)
        else:
            print('new model')
            model = Sequential()
            # In my Model, I have 19 adjustable parameters, each one of which can be adjusted by i in [-2, -1, 1, 2], so 4; 19*4 = 76 as possible outputs
            #model.add(Flatten(input_shape=(19,)))
            model.add(Dense(10, activation='relu', input_dim=19))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(1, activation='linear'))  # I want to predict a real-value quantity: Estimate reward of the next state,

            # therefore I solve by Regression; single output node.
            model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


        model.summary()
        return model

    # Do EXPERIENCE REPLAY, OR ESTIMATE Q-VALUE, if GAME_OVER == False
    def train_model(self, state, target, epochs):

        # target = self.value_model.predict(self.State)
        # If it is the last state, just give the state reward, as there is not target state left.
        tta = target_to_array(target)
        if target is not None:
            self.value_model.fit(state, tta, epochs=epochs, verbose=0)

# ---------------------------------------------------------------------------------------------------------------------#
def target_to_array(target):
    num_tta = []
    num_tta = np.array([target])
    return num_tta


def state_to_array(state):
    num_state = []
    num_state = np.array([state])/20.0
    return num_state

# If state exploration happens, then uniformly, randomly choose between states and adjust these.
def state_scrambler(state):
    # pick index between 0-18
    valid = False
    while valid == False:
        idx = round(random.uniform(0, 1)*18)
        # tweak by max 4, i.e. allow for -2,-1,1,2 as element tweak
        tweak = 0
        while tweak == 0:
            tweak = round(random.uniform(0, 1)*4) - 2

        new_state = list(state)[:]
        new_state[idx] = new_state[idx] + tweak
        #check that the new state makes sense
        valid = state_tweak_check(new_state) and Check_Valid_Build_Plan(new_state)
    return idx, tweak

def state_tweak_check(state):
    checked_state = True
    too_large = [item for item in state if item > 20]
    too_small = [item for item in state if item <-20]
    if len(too_large)+len(too_small) > 0:
        checked_state = False
    return checked_state

#https://stackoverflow.com/questions/18893624/partial-match-dictionary-keyof-tuples-in-python
def partial_match(key, d):
    for k, v in d.items():
        if all(k1 == k2 or k2 is None for k1, k2 in zip(k, key)):
            yield v


def state_tweak_confine(state):
    checked_state = [20 if i >= 20 else -20 if i <= -20 else i for i in state]
    return checked_state

if __name__ == '__main__':
    # Input the number of episodes I want to have
    #alternate every 100 episodes
    drawing = False
    shifts = 400
    episodes_per_shift = 20
    for i in range(shifts):
        Earth = DQN_Architect(exploration_value=0.6, planet=1, drawing=drawing)
        Earth.play_to_learn(episodes_per_shift)

        # Moon = DQN_Architect(exploration_value=0.7, planet=2, drawing=drawing)
        # Moon.play_to_learn(episodes_per_shift)

        Mars = DQN_Architect(exploration_value=0.7, planet=3, drawing=drawing)
        Mars.play_to_learn(episodes_per_shift)

    print("All Done")