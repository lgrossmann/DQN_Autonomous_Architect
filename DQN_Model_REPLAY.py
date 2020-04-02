
# In this module, the Deep Q-learning takes place, an agent is created, i.e. the system which iterates through the
# build_plan pivot blocks and for every episode changes the current position by 10*x pixels.
# Upon this, the reward is given, and the value of the state then assessed.

from Block_architect_MAIN import Check_Valid_Build_Plan
import random
import pickle # to save complete objects, as binaries
import csv
import os
import pygame
import time
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
import numpy as np
from collections import deque
from Block_architect_MAIN import Evaluate_Build_Plan, det_standard_plan_cost
from pathlib import Path
# pygame.init()

class Replay_DQN_Architect:
    def __init__(self, exploration_value = 1, planet = 1, drawing= True):
        pygame.init()
        self.Initial_State = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        # self.Initial_State = (-3, 0, 0, 0, -1, 0, 0, 5, -7, 0, 0, 12, -15, 0, -2, 0, -17, 0, -2)
        self.Drawing = drawing
        self.State = self.Initial_State
        self.Previous_State = self.Initial_State
        self.Precedent_State = None
        self.Game_over = False
        self.Max_no_moves = 40  # max no moves in one game (for testing use ~ = 1)
        self.Moves_played = 0
        self.Exploration_factor = exploration_value # epsilon exploratory factor
        self.Planet = planet
        self.Reward = None
        self.Gamma = 0.9 # discount factor
        self.value_model = self.load_model()
        self.Default_cost = 0
        self.Rewards_Dict = dict()
        self.MAX_Reward_in_episode = 0
        self.Reward_sum_in_episode = 0
        self.State_with_MAX_Reward = list()
        # self.Replay_Buffer = deque(maxlen=2000)
        self.load_replay()
        self.Min_Replay_Buffer_threshold = 10

        # 2 = use 50% for replay, 10 = use 10% for replay etc
        self.Batch_quota = 2

        if self.Planet == 1:  # Earth
            self.Default_cost = 0.1
        elif self.Planet == 2: # Moon
            self.Default_cost = 0.2
        elif self.Planet == 3: # Mars
            self.Default_cost = 0.25

    def play_to_learn(self, episodes, training_phase=True): # One Episode continues until Game Over! Then next episode

        # Get standard mass cost as a reference for evaluating.
        ref_state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Standard_mass_cost = det_standard_plan_cost(ref_state, self.Planet, cost=self.Default_cost, drawing=self.Drawing)
        print("Standard Mass Cost: ", self.Standard_mass_cost)

        for i in range(episodes):
            self.Reward_sum_in_episode = 0
            self.MAX_Reward_in_episode = 0
            self.State_with_MAX_Reward = list()

            self.init_sim() # back to setup
            self.Game_over = False

            # Print the Episode, with Current Planet Environment
            planets = {1: 'Earth', 2: 'Moon', 3: 'Mars'}
            if self.Planet == 1:
                 current_planet = planets.get(1)
            if self.Planet == 2:
                current_planet = planets.get(2)
            if self.Planet == 3:
                current_planet = planets.get(3)
            print('**********************************************')
            print('******** Episode number: ' + str(i) + '********************')
            print('*******************' + current_planet + '*********************')
            print('**********************************************')

            while self.Game_over == False:
                self.Moves_played += 1
                print('Move no: ' + str(self.Moves_played))
                if self.Moves_played >= self.Max_no_moves:
                    self.Game_over = True
                else:
                    self.Q_choose_move_learn_state_and_execute_move(training_phase=training_phase)
                    self.Reward_sum_in_episode += self.Reward
                    if self.Reward > self.MAX_Reward_in_episode:
                        self.MAX_Reward_in_episode = self.Reward
                        self.State_with_MAX_Reward = list(self.State)

            localtime = time.localtime(time.time())
            self.Rewards_Dict[localtime] = (self.Reward_sum_in_episode, self.MAX_Reward_in_episode, self.State_with_MAX_Reward)
            print(">>> Total Reward in Episode ", i, "was: ", self.Reward_sum_in_episode)
            print(">>> MAX Reward in Episode ", i, "was: ", self.MAX_Reward_in_episode, "in State: ", self.State_with_MAX_Reward)

            self.all_count = i
            if i % 100 == 0 and training_phase == True:
                self.save_model()
                self.save_rewards()
                self.save_replay()
                # refresh, since all previous rewards are already saved.
                self.Rewards_Dict = dict()
        if training_phase == True:
            self.save_model()
            self.save_rewards()
            self.save_replay()
            # refresh, since all previous rewards already saved (episodes all done).
            self.Rewards_Dict = dict()

    def init_sim(self):
        self.State = self.Initial_State
        self.Previous_State = self.Initial_State
        self.Precedent_State = None

        self.Moves_played = 0
        self.Reward = None

    def save_rewards(self):
        rewards_saved = 'rewards_DQN_REPLAY' + str(self.Planet) + '.csv'
        a = csv.writer(open(rewards_saved, 'a', newline=''))

        for v, k in self.Rewards_Dict.items():
            a.writerow([v, k])
        pass

    # Only need to load when graphing reward_sum in an episode.
    def load_rewards(self):
        load_saved = 'rewards_DQN_REPLAY' + str(self.Planet) + '.csv'

        try:
            rewards_csv = csv.reader(open(load_saved, 'r'))
            for row in rewards_csv:
                k, v = row
                self.Rewards_Dict[k] = float(v)

        except:
            pass

    def save_replay(self):
        replay_saved = 'buffer_DQN_REPLAY' + str(self.Planet) + '.pickle'
        with open(replay_saved, 'wb') as savelocation:
            pickle.dump(self.Replay_Buffer,savelocation)
        pass

    def load_replay(self):
        load_saved = 'buffer_DQN_REPLAY' + str(self.Planet) + '.pickle'
        self.Replay_Buffer = deque(maxlen=2000)
        try:
            with open(load_saved, 'rb') as savelocation:
                self.Replay_Buffer = pickle.load(savelocation)
        except:
            pass

    def Q_choose_move_learn_state_and_execute_move(self, training_phase=True):
        # select action,
        selected_next_action = self.select_action(training_phase=training_phase)

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

        if training_phase == True:
            # Get ALL Q values starting from temp_state (i.e. next_state)
            # We want to find MAXa_Q that is starting in the next state.
            # Better MAXa_Q by sampling some next_moves and "average??" instead of predict here.
            if self.Game_over == False:

                # self.Values[(self.State, selected_next_action)] = Q_s_a + self.Alpha * (self.Reward + self.Gamma *
                # MAXa_Q - Q_s_a)

                # target = (predictQ(s') + reward)
                predictQs_2 = self.value_model.predict(state_to_array(temp_state))[0]

                target = np.array(self.Gamma * predictQs_2 + self.Reward)
                # target = np.array(self.Reward + self.Gamma * MAXa_Q)

                # Qnew(s(t), (a)) = Q(s(t), (a)) + Alpha * (reward_e + gamma*maxaQ(s(t+1), a(t) - Q(s(t), (a)))

                print("Temporary Next State: ", temp_state)
                print("Q TARGET:", target, "-_-_- Reward -_-_-: ", self.Reward)

            else:
                target = np.array([self.Reward])
                print("Temporary Next State: ", temp_state)
                print("-_-_- Reward for Game Over-_-_-: ", self.Reward)
            # UPDATE MODEL TO LEARN FROM NEW TEMP STATE TO TARGET

            # EXPERIENCE REPLAY: FATTEN REPLAY BUFFFFEERR:

            self.Replay_Buffer.append([state_to_array(temp_state), target_to_array(target)])


            self.train_model_replay(replay_buffer=self.Replay_Buffer, epochs=20)

        # state is updated finally to state from new action
        self.Precedent_State = list(self.State)[:]
        self.State = tuple(temp_state)

    def select_action(self, training_phase=True): # My Policy select_OPTIMAL action at self.State. if selected_action goes to (0,0,0,...) scramble.
        # actions have the form  (index, tweak)
        current_state_before_move = self.State
        p = random.uniform(0, 1) # maybe change to non uniform
        # p chosen above the exploration factor -> choose a random move

    # WHILE LOOP over if/else: While
        need_to_scramble = False

        if training_phase == False:
            self.Exploration_factor = 1
            # We want follow the optimal moves only.

        while True:
            if p < self.Exploration_factor and need_to_scramble == False:
                selected_next_action = self.make_optimal_move(current_state_before_move, predecessor_state_L=self.Precedent_State)

            else:
                # Pick a random element and tweak it
                selected_next_action = state_scrambler(current_state_before_move, predecessor_state_L=self.Precedent_State)

            # If current.State acting selected_next_action results in self.Initial_State, continue.
            # else return selected_next_action.
            # Hypothetical next state =
            temp_state = list(self.State)[:]
            temp_state[selected_next_action[0]] = temp_state[selected_next_action[0]] + selected_next_action[1]
            # temp_state now is set to the next state after taking the proposed action

            if temp_state == self.Initial_State or Check_Valid_Build_Plan(temp_state) == False:
                need_to_scramble = True
                print("CAUTION: I need to scramble not to revert to initial STATE or do invalid architecture.")
                continue

            else:
                # print("ALL GOOD")
                return selected_next_action

    def make_optimal_move(self, state, predecessor_state_L=None):
        # make the optimal move in this state, i.e.
        # Policy: among possible actions pick the one with highest next Q-value if exists, otherwise pick random
        possible_actions = list()
        for index in range(len(state)):
            # For i in the list of possible state updates per move:
            # Should I have i go through all actions, then storing all actions at once, or rather of ONE random action?
            # What about exploring in that case?
            for tweak in [-2, -1, 1, 2]:
                # To truly create a DEEP copy of a list, need to slice ([:])
                MOVE_temp_state = list(state)[:]
                update = state[index] + tweak
                MOVE_temp_state[index] = update
                MOVE_temp_state_tuple = tuple(MOVE_temp_state)

                # Maybe have reward = -10, since that is invalid procedure?
                if update < -20 or update > 20 or Check_Valid_Build_Plan(MOVE_temp_state) == False:
                    pass

                # Do not revert back to previous state
                elif MOVE_temp_state == predecessor_state_L:
                    print("HelauAlaaf")
                    pass

                else:
                    possible_actions.append((index, tweak))

        value = -float('Inf')
        best_action_list = list()

        # Now find the best move out of all candidates.
        for index in range(len(possible_actions)):
            # calc_val = 0
            candidate_state = list(self.State)[:]
            candidate_action = possible_actions[index]
            candidate_state[candidate_action[0]] = candidate_state[candidate_action[0]] + candidate_action[1]
            candidate_state_array = state_to_array(candidate_state)
            predict_result = self.value_model.predict(candidate_state_array)
            calc_val = predict_result[0]
            # print("______PREDICTED VALUE_____: ", calc_val)
            # print("possible actions: ",len(possible_actions), "calculated value from 1 poss. action: ",calc_val )
            if calc_val > value:
                best_action_list = [candidate_action]
                value = calc_val
            elif calc_val == value:
                best_action_list.append(candidate_action)

        # If best action list empty, then force exploration

        if len(best_action_list) > 0:
            optimal_action = random.choice(best_action_list)
        else:
            optimal_action = random.choice(possible_actions)
            print("Couldn't decide from optimal actions, had to choose from possible actions")

        # print("Calculated Value is ", value, " for the Optimal Action: ", optimal_action)
        return optimal_action

# ---------------------------------------------- DQN_MODEL ----------------------------------------------------------- #

    def save_model(self):
        current_save = 'model_values_REPLAY' + str(self.Planet) + '.h5'
        try:
            os.remove(current_save)
        except:
            pass
        self.value_model.save(current_save)

    # if a previous version of the model exists, then load, so learning
    # can resume. Especially helpful if crashes.
    def load_model(self):
        current_save = 'model_values_REPLAY' + str(self.Planet) + '.h5'
        model_file = Path(current_save)
        if model_file.is_file():
            model = load_model(current_save)
            print('load available model: ' + current_save)
        else:

            # Create a new model
            print('new model')
            model = Sequential()

            # Input is the State = 19 input parameters
            # ReLU allows for backpropagation of erros and still have
            # multiple neural layers
            model.add(Dense(10, activation='relu', input_dim=19))

            # Output for every hidden, Dense (numpy array) layer is optimized at
            # (input_dim + output_dim)/2
            # Dense: Every layer in the input layer is connected to the
            # next layer etc...
            model.add(Dense(10, activation='relu'))
            model.add(Dense(10, activation='relu'))

            # Output Layer: I want to predict a real-value quantity,
            # linear activation ; estimate Q of the next state,
            # therefore single output node.
            model.add(Dense(1, activation='linear'))

            # Compile model with a loss function (MSE: diff. between target
            # and actual Q) and neural weight optimizer
            model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        # Prints what the model specifications look like.
        model.summary()
        return model

    # Do experience replay, estimate the Q-value while not Game_Over: Training
    def train_model_replay(self, replay_buffer, epochs):

        # If it is the last state, just give the state reward, as there is not target state left.
        # Build Replay Buffer: Deque, can hold max 1000 entries, [arrays for training the model], picking a random batch
        # from the Replay Buffer to train
        # Replay Buffer runs across episodes.
        rpbsize = len(replay_buffer)
        newest_element = replay_buffer.pop()
        replay_buffer.append(newest_element)

        if rpbsize >= self.Min_Replay_Buffer_threshold:
            batch = random.sample(replay_buffer, int(rpbsize/self.Batch_quota))
            batch.append(newest_element)
        else:
            batch = list()
            batch.append(newest_element)
        if rpbsize > 0:
            reshape_data = [np.array([x[0] for x in batch]), np.array([x[1] for x in batch])]
            # random selection of min 1, max rpbsize/10 from reshape_data used for training
            # newest always included
            X_train = reshape_data[0].reshape(-1, 19) # -1 trick makes sure there are 19 entries per row
            y_train = reshape_data[1].reshape(-1, 1) # -1 trick lets python determine the other shape param
            self.value_model.fit(X_train, y_train, epochs=epochs, verbose=0)

# ---------------------------------------------------------------------------------------------------------------------#
# Target (predict) is converted to an array for training
def target_to_array(target):
    num_tta = list()
    num_tta = np.array([target])
    return num_tta

# State is converted to array for training
def state_to_array(state):
    num_state = list()
    num_state = np.array([state])/20.0
    return num_state

# If state exploration happens, then uniformly, randomly choose between states and adjust these.
def state_scrambler(state, predecessor_state_L):
    
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

        # check that the new state makes sense
        valid = state_tweak_check(new_state) and Check_Valid_Build_Plan(new_state) and new_state != predecessor_state_L
    return idx, tweak


def state_tweak_check(state):
    checked_state = True
    too_large = [item for item in state if item > 20]
    too_small = [item for item in state if item < -20]
    if len(too_large)+len(too_small) > 0:
        checked_state = False
    return checked_state

# https://stackoverflow.com/questions/18893624/partial-match-dictionary-keyof-tuples-in-python
# Used in previous version of code, but interesting trick for getting items out of a dictionary
def partial_match(key, d):
    for k, v in d.items():
        if all(k1 == k2 or k2 is None for k1, k2 in zip(k, key)):
            yield v

# Check that the current state is not out of bounds.
# These restrictions are only necessary to reduce the number of possible states
def state_tweak_confine(state):
    checked_state = [20 if i >= 20 else -20 if i <= -20 else i for i in state]
    return checked_state


if __name__ == '__main__':    
    # Input the number of episodes I want to have
    # alternate every x00 episodes
    drawing = True
    training_phase = True
    shifts = 800
    episodes_per_shift = 10
    for i in range(shifts):
        # exploration value of 0.7 means 30% random moves
        Earth = Replay_DQN_Architect(exploration_value=0.7, planet=1, drawing=drawing)
        Earth.play_to_learn(episodes_per_shift, training_phase=training_phase)
        pygame.quit()

        Moon = Replay_DQN_Architect(exploration_value=0.7, planet=2, drawing=drawing)
        Moon.play_to_learn(episodes_per_shift, training_phase=training_phase)
        pygame.quit()

        Mars = Replay_DQN_Architect(exploration_value=0.7, planet=3, drawing=drawing)
        Mars.play_to_learn(episodes_per_shift, training_phase=training_phase)
        pygame.quit()

    print("That was done, quick :-)")
