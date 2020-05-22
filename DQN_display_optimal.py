import ast
import csv
import operator
import Block_architect_MAIN
from matplotlib import pyplot as plt
import rolling as roll
import pygame

Best_States = []


def load_values(planet):
    load_saved = 'values' + str(planet) + '.csv'
    temp_dict = dict()
    try:
        values_csv = csv.reader(open(load_saved, 'r'))
        for row in values_csv:
            k, v = row
            temp_dict[k] = float(v)

    except:
        pass

    return temp_dict

def load_rewards(planet, type):
    if type == 'Q':
        load_saved = 'rewards' + str(planet) + '.csv'
    elif type == 'DQN':
        load_saved = 'rewards_DQN' + str(planet) + '.csv'
    elif type == 'RQN':
        load_saved = 'rewards_DQN_REPLAY' + str(planet) + '.csv'
    Rewards_Dict = dict()

    try:
        rewards_csv = csv.reader(open(load_saved, 'r'))
        for row in rewards_csv:
            k, v = row
            Rewards_Dict[k] = ast.literal_eval(v)

    except:
        pass

    return Rewards_Dict

def partial_matches(key, d):
    my_dict = {}
    for k, v in d.items():
        temp_k = ast.literal_eval(k)
        if temp_k[0] == key:
            my_dict[k] = v

    return my_dict

def repeat_optimal(planet):
    planet = planet
    Optimal_State_Dict = load_values(planet)
    # Follow the optimal states through showing their actions
    #Iterate through q_list
    Max_no_moves = 80

    #
    # First loop, loop until quality gets worse
    #
    moves = 0
    current_state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    # current_state = (-3, 0, 0, 0, -1, 0, 0, 5, -7, 0, 0, 12, -15, 0, -2, 0, -17, 0, -2)
    no_more = False
    top_value = -float('Inf')
    while moves <= int(Max_no_moves) and no_more == False:
    # Find best quality action from current_state:
        # Where key is fixed on ((current_state),(ACTION)): Value
        All_possible_next_actions = dict()
        m = partial_matches(current_state, Optimal_State_Dict)
    # Do best action, put into next state
    # DISPLAY bes action for __ seconds.
        if m == {}:
            no_more = True
        else:
            Max_Optimal_Value_State_with_Action = max(m.items(), key=operator.itemgetter(1))[0]

            print("State with best next action to take:", Max_Optimal_Value_State_with_Action)
            best_value = Optimal_State_Dict[Max_Optimal_Value_State_with_Action]
            print("Quality for the action: ", best_value)

            state_action_tuple = ast.literal_eval((Max_Optimal_Value_State_with_Action))
            if best_value > top_value:
                top_value = best_value
                top_tuple = state_action_tuple
            idx = state_action_tuple[1][0]
            tweak = state_action_tuple[1][1]
            temp_current_state = list(state_action_tuple[0])
            temp_current_state[idx] = temp_current_state[idx] + tweak

            #print(temp_current_state)
            #print("----Action----: ", tweak, "Index: ", idx)
            current_state = tuple(temp_current_state)
            # Block_architect_MAIN.Presentation_Build_Plan(current_state, planet, cost=0.1)
            moves += 1

    #
    # Second loop, loop until the best position known from first loop, then stop
    # This time: draw!
    #
    print("--- Second Loop ---")
    moves = 0
    current_state = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) # Search for sequences starting with Initial State OR Optimal State
    no_more = False
    while moves <= int(Max_no_moves) and no_more == False:
    # Find best quality action from current_state:
        # Where key is fixed on ((current_state),(ACTION)): Value
        All_possible_next_actions = dict()
        m = partial_matches(current_state, Optimal_State_Dict)
    # Do best action, put into next state
    # DISPLAY bes action for __ seconds.
        Max_Optimal_Value_State_with_Action = max(m.items(), key=operator.itemgetter(1))[0]

        print("State with best next action to take:", Max_Optimal_Value_State_with_Action)

        print("Quality for the action: ", Optimal_State_Dict[Max_Optimal_Value_State_with_Action])

        state_action_tuple = ast.literal_eval((Max_Optimal_Value_State_with_Action))
        idx = state_action_tuple[1][0]
        tweak = state_action_tuple[1][1]
        temp_current_state = list(state_action_tuple[0])
        temp_current_state[idx] = temp_current_state[idx] + tweak
        if top_tuple == state_action_tuple:
            no_more = True
        #print(temp_current_state)
        #print("----Action----: ", tweak, "Index: ", idx)
        current_state = tuple(temp_current_state)
        Block_architect_MAIN.Presentation_Build_Plan(current_state, planet, cost=0.1, drawing=True)
        moves += 1
    print("*** DONE ***")

def graph(type='Q', window_size=100):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    planets = {1: 'Earth', 2: 'Moon', 3: 'Mars'}

    for planet in range(1, 4, 1):

        # Read Rewards_Dict for planet
        Rewards_Dict = load_rewards(planet, type)
        # Put all the values in a list in sequence
        Rewards_COMPLETE_List = list(Rewards_Dict.values())

        Rewards_List = [i[0] for i in Rewards_COMPLETE_List]
        Roll_Rewards = list(roll.Mean(Rewards_List, window_size))

        # ax.plot(Rewards_List, label=str(planets[planet]))
        ax.plot(Roll_Rewards, label=str(planets[planet]))

    ax.set_xlabel(str("Episodes - " + type))
    ax.set_ylabel(str("Total Rewards per Episode - rolling "+str(window_size)))
    legendary = plt.legend(loc='best', ncol=1, mode="None", shadow=False, fancybox=True)
    legendary.get_frame().set_alpha(1)

    plt.show()

def graph_2(type='Q', window_size=100):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    planets = {1: 'Earth', 2: 'Moon', 3: 'Mars'}

    for planet in range(1, 4, 1):

        # Read Rewards_Dict for planet
        Rewards_Dict = load_rewards(planet, type)
        # Put all the values in a list in sequence
        Rewards_COMPLETE_List = list(Rewards_Dict.values())

        Optimum_List = [i[1] for i in Rewards_COMPLETE_List]
        # Roll_Optimum = list(roll.Median(Optimum_List, window_size))
        Roll_Optimum = list(roll.Mean(Optimum_List, window_size))
        # Roll_Optimum = list(roll.Max(Optimum_List, window_size))

        # ax.plot(Optimum_List, label=str(planets[planet]))
        ax.plot(Roll_Optimum, label=str(planets[planet]))


    ax.set_xlabel(str("Episodes - " + type))
    ax.set_ylabel(str("Optimum Rewards per Episode - rolling "+str(window_size)))
    legendary = plt.legend(loc='best', ncol=1, mode="None", shadow=False, fancybox=True)
    legendary.get_frame().set_alpha(1)

    plt.show()

def graph_complete(window_size=100):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    planets = {1: 'Earth', 2: 'Moon', 3: 'Mars'}

    for type in ['DQN', 'RQN']:

        for planet in range(1, 4, 1):

            # Read Rewards_Dict for planet
            Rewards_Dict = load_rewards(planet, type)
            # Put all the values in a list in sequence
            Rewards_COMPLETE_List = list(Rewards_Dict.values())

            Optimum_List = [i[1] for i in Rewards_COMPLETE_List]
            # Roll_Optimum = list(roll.Median(Optimum_List, window_size))
            Roll_Optimum = list(roll.Mean(Optimum_List, window_size))
            # Roll_Optimum = list(roll.Max(Optimum_List, window_size))

            # ax.plot(Optimum_List, label=str(planets[planet]))
            ax.plot(Roll_Optimum, label=type + " - " + str(planets[planet]))


    ax.set_xlabel(str("Episodes"))
    ax.set_ylabel(str("Optimum Rewards per Episode - rolling " + str(window_size)))
    legendary = plt.legend(loc='best', ncol=1, mode="None", shadow=False, fancybox=True)
    legendary.get_frame().set_alpha(1)

    plt.show()


def stress_test_state(state_as_list, planet, stress_type='Meteor'):

    current_state = tuple(state_as_list)
    Block_architect_MAIN.Stress_Build_Plan(current_state, planet, stress_type=stress_type,  cost=0.1, drawing=True)
    print("--: DONE :--")


def print_best_state_overall(type='Q'):

    planets = {1: 'Earth', 2: 'Moon', 3: 'Mars'}

    global Best_States

    for planet in range(1, 4, 1):

        # Read Rewards_Dict for planet
        Rewards_Dict = load_rewards(planet, type)
        # Put all the values in a list in sequence
        Rewards_COMPLETE_List = list(Rewards_Dict.values())

        State_List = [i[2] for i in Rewards_COMPLETE_List]
        Optimum_List = [i[1] for i in Rewards_COMPLETE_List]

        Best_Index = Optimum_List.index(max(Optimum_List))
        print(type, ":__Best Build Plan__ for ", planets[planet], "is", State_List[Best_Index], "at", Optimum_List[Best_Index])
        Best_States.append([State_List[Best_Index], planet])

if __name__ == '__main__':
    # repeat_optimal(planet=3)

    Best_States = []
    window_size = 30
    graph(type='Q', window_size=window_size)
    graph_2(type='Q', window_size=window_size)
    graph(type='DQN', window_size=window_size)
    graph_2(type='DQN', window_size=window_size)
    graph(type='RQN', window_size=window_size)
    graph_2(type='RQN', window_size=window_size)

    graph_complete(window_size=window_size)

    print_best_state_overall(type='Q')
    print_best_state_overall(type='DQN')
    print_best_state_overall(type='RQN')

    stress_test_the_best = True # set True if you want to show
    if stress_test_the_best == True:
        for row in Best_States:
            # stress_test_state(state_as_list=row[0], planet=row[1], stress_type='Meteor')
            # pygame.quit()
            stress_test_state(state_as_list=row[0], planet=row[1], stress_type='Wind')
            pygame.quit()
            # stress_test_state(state_as_list=[-3, 0, 0, 0, -1, 0, 0, 5, -7, 0, 0, 12, -15, 0, -2, 0, -17, 0, -2], planet=1, stress_type='Meteor')
            # stress_test_state(state_as_list=[-3, 0, 0, 0, -1, 0, 0, 5, -7, 0, 0, 12, -15, 0, -2, 0, -17, 0, -2], planet=1, stress_type='Wind')
        pygame.quit()