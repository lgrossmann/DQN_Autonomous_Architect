import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util
from math import hypot

class Build_Plan:
    # A build plan consists of a list of double tuples of the form ((x, y), (w, h)), where
    # (x,y) denotes the bottom left corner of a rectangular blocks of width w and height h.
    # y is measured above ground and ground is  space property.
    def __init__(self, list_input, cost = 100, density = 1):
        self.Pivot_Blocks = list_input
        self.Cost = cost # cost per mass
        self.Density = density
        self.Current_Block = None
        self.Next_Block = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.Current_Block == None:
            self.Current_Block = self.Pivot_Blocks[0]
            self.Next_Block = self.Pivot_Blocks[1]
            return self.Current_Block
        else:
            if not self.Pivot_Blocks.index(self.Next_Block) == len(self.Pivot_Blocks) - 1:
                self.Current_Block = self.Pivot_Blocks[self.Pivot_Blocks.index(self.Current_Block) + 1]
                self.Next_Block = self.Pivot_Blocks[self.Pivot_Blocks.index(self.Next_Block) + 1]
                return self.Current_Block
            else:
                raise StopIteration
    def check_architecture_valid(self):
        # TODO: Do some reasonablility checks on the build_plan.
        return True

def det_pos_points (double_tuple, ground_level):
    x = double_tuple[0][0]
    y = double_tuple[0][1]
    w = double_tuple[1][0]
    h = double_tuple[1][1]
    v_pos = Vec2d(x + w/2, y+h/2 + ground_level)
    l_points = [(-w/2, -h/2), (-w/2, h/2), (w/2,h/2), (w/2,-h/2)]
    return v_pos, l_points

def det_pos_mirror (pos, x_axis):
    mirror_pos = Vec2d(x_axis*2 - pos[0], pos[1])
    return mirror_pos




class Block_Construct:

    def add_body_to_space(self, pos, points, friction, body_collision_type):
        w = abs(points[0][0]*2)
        h = abs(points[0][1]*2)
        volume = w * h * 1
        mass = self.Density * volume
        moment = pymunk.moment_for_poly(mass, points, (0, 0))
        body = pymunk.Body(mass, moment)
        body.position = pos
        shape = pymunk.Poly(body, points)
        # shape.color = (255, 0, 0, 255)
        shape.friction = friction
        shape.collision_type = body_collision_type
        self.space.add(body, shape)

    def collision_happened(self, arbiter, two, three):
        self.Outcome_Collision = True
        self.Outcome_Collision_Count = self.iteration_counter
        self.running = False
        # print("COLLISION_HAPPENED!!!")
        return True

    def __init__(self, build_plan, planet):
        # radius = float(input("Please enter length of interior space in meters:"))/2
        radius = 390/2
        self.running = True
        self.drawing = True # Change to True if I want to see...
        self.Outcome_Collision = False
        self.Outcome_Stable_Over_Time = False
        self.Outcome_No_Motion = False
        self.Outcome_Pymunk_Sleep = False
        self.Build_Plan_Valid = False
        self.Build_Plan_Valid = build_plan.check_architecture_valid()
        self.Planet = planet
        self.w, self.h = 1000, 800
        # Make sure that block to is centered, so x is derived, overrides x given.
        # x given can still serve as a compensator for the top block attachment
        interim_build_plan = build_plan
        interim_build_plan_pivot = interim_build_plan.Pivot_Blocks
        last_element = interim_build_plan_pivot[-1]
        top_block_compensator = last_element[0][0]
        # last_element[0][0] = self.w/2 - last_element[1][0]/2
        new_last_element = ((self.w/2 - last_element[1][0]/2, last_element[0][1]),last_element[1])
        interim_build_plan_pivot[-1] = new_last_element
        interim_build_plan.Pivot_Blocks = interim_build_plan_pivot
        build_plan = interim_build_plan
        self.Build_Plan = build_plan
        self.Density = build_plan.Density

        if self.drawing == True:
            self.screen = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()

        # Init pymunk and create space
        self.space = pymunk.Space()

        # TODO: Choose gravity according to planet gravity and proportions in simulator.
        if planet == 1:
            self.space.gravity = (0.0, -980.0)
        elif planet == 2:
            self.space.gravity = (0.0, -114.0)
        elif planet == 3:
            self.space.gravity = (0.0, -500.0)

        self.space.sleep_time_threshold = 0.3
        # ground
        ground_level = 100
        standardmass = 1
        standardfriction = 1
        x_axis = self.w /2

        Collision_Type_Block = 1
        # Setting habit. cirlce and ground collision to the same, for if a block falls on them.
        Collision_Type_Habite_Ground = 2
        Collision_Type_Base = 3

        habite = pymunk.Circle(self.space.static_body, radius, (self.w/2, radius))
        habite.friction = standardfriction
        # habite.color = (255, 255, 255) # white
        habite.collision_type = Collision_Type_Habite_Ground
        self.space.add(habite)

        shape = pymunk.Segment(self.space.static_body, (5, ground_level), (self.w - 5, ground_level), 1.0)
        shape.friction = standardfriction
        shape.collision_type = Collision_Type_Habite_Ground
        self.space.add(shape)

        for block in self.Build_Plan:

            # print("*** NEW corner block ***")
            # Block1 of current stack - in every iteration
            pos, points = det_pos_points(block, ground_level)

            if self.Build_Plan.Current_Block == self.Build_Plan.Pivot_Blocks[0]:
                # Now at the very first block

                wished_for_collision_type = Collision_Type_Base

            else:
                wished_for_collision_type = Collision_Type_Block

            self.add_body_to_space(pos, points, standardfriction, wished_for_collision_type)

            # Block1_MIRROR
            pos_mirror = det_pos_mirror(pos, x_axis = x_axis)  # means same as Current block here where we say block
            # Collision type is same, since it is mirrored.
            self.add_body_to_space(pos_mirror, points, standardfriction, wished_for_collision_type)

            # Fill in all blocks before the Next_Block of the building.
            # What is dist (h) between Current_Block and Next_Block

            x_current = self.Build_Plan.Current_Block[0][0]
            y_current = self.Build_Plan.Current_Block[0][1]
            x_next = self.Build_Plan.Next_Block[0][0]
            y_next = self.Build_Plan.Next_Block[0][1]
            w_current = self.Build_Plan.Current_Block[1][0]
            w_next = self.Build_Plan.Next_Block[1][0]
            h_current = self.Build_Plan.Current_Block[1][1]
            if self.Build_Plan.Next_Block == self.Build_Plan.Pivot_Blocks[-1]:
                x_dist = (x_next + w_next/2) - (x_current + w_current/2) - top_block_compensator
            else:
                x_dist = (x_next + w_next / 2) - (x_current + w_current / 2)
            y_dist = y_next - y_current - h_current

            # print("I am at x,y :", x_current, ":", y_current)
            # print("I need to reach towards x,y :", x_next, ":", y_next," with height ",h_current)
            # print("this is a distance of ", y_dist)
            num_blocks = int(y_dist/h_current)
            y_remainder = y_dist - h_current * num_blocks
            num_blocks_full = num_blocks
            # print("I need to cover with total:", num_blocks_full, " blocks of of width ", w_current)
            fill_in_needed = False
            if y_remainder > 0:
                fill_in_needed = True
                num_blocks_full = num_blocks + 1
                # print("a remainder will be needed")

            x_shift = x_dist/(num_blocks_full + 1)

            # Build all regular height issued blocks.
            if num_blocks > 0:
                for i in range(1, num_blocks+1):
                    # create an interim block
                    new_x = x_current + w_current/2 + i * x_shift - w_current/2 # Would not be necessary but could be
                                                                                # replaced more efficiently if w is
                                                                                # replaced dynamically

                    new_y = y_current + i * h_current
                    pos, dnu_points = det_pos_points( ((new_x, new_y), (w_current, h_current)), ground_level)
                    # print("my ", i, "-th block is at pos: ", pos, "x ",new_x, " y ", new_y, "width ", w_current)

                    self.add_body_to_space(pos, points, standardfriction, Collision_Type_Block)

                    pos_mirror = det_pos_mirror(pos, x_axis = x_axis)

                    self.add_body_to_space(pos_mirror, points, standardfriction, Collision_Type_Block)

            # Build a remainder block if necessary
            if fill_in_needed:
                # print("fill in a remainder block")
                new_x = x_current + w_current / 2 + num_blocks_full * x_shift - w_current / 2
                new_y = y_current + num_blocks_full * h_current
                pos, points = det_pos_points(((new_x, new_y), (w_current, y_remainder)), ground_level)
                # print("at x,y ", new_x, new_x, "at pos: ", pos)

                self.add_body_to_space(pos, points, standardfriction, Collision_Type_Block)

                pos_mirror = det_pos_mirror(pos, x_axis=x_axis)

                self.add_body_to_space(pos_mirror, points, standardfriction, Collision_Type_Block)

        # -------.... Now build the LAST BLOCK which is the remaining Next_Block... -------
        # print("*** The TOP-BLOCK ***")
        # this is the last next block remaining
        pos, points = det_pos_points(self.Build_Plan.Next_Block, ground_level)

        self.add_body_to_space(pos, points, standardfriction, Collision_Type_Block)

        if self.drawing == True:
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        self.error_of_construction = self.space.add_collision_handler(Collision_Type_Habite_Ground, Collision_Type_Block)
        self.error_of_construction.post_solve = self.collision_happened



    def run(self):
        self.iteration_counter = 0
        self.initial_body_position = [Body.position for Body in self.space.bodies]
        while self.running:
            # print(self.space.bodies[0].position, "---", self.space.bodies[1].position)
            # mylist=[Body.mass for Body in self.space.bodies]
            # print("Total mass: ",sum(mylist))
            # print("Total Cost: ", sum(mylist)*self.Build_Plan.Cost)
            # all_blocks_sleeping = [not Body.is_sleeping for Body in self.space.bodies]
            # print("Total true: ", sum(all_blocks_sleeping))

            self.loop()  # Look at the pos of object, compare to the pos before the time step, and if pos stable, evaluate and stop.

            # if self.error_of_construction.begin is not None:
            #     self.running = False
            #     print("done")

    # https: // stackoverflow.com / questions / 23410161 / pygame - collision - code

    def loop(self):
        # for event in pygame.event.get():
        #     if event.type == QUIT:
        #         self.running = False
        #     elif event.type == KEYDOWN and event.key == K_ESCAPE:
        #         self.running = False
        #     elif event.type == KEYDOWN and event.key == K_p:
        #         pygame.image.save(self.screen, "Auto_Arch2.png")
        #     elif event.type == KEYDOWN and event.key == K_d:
        #         self.drawing = not self.drawing

        self.iteration_counter += 1
        if self.iteration_counter >= 1000:
            self.Outcome_Stable_Over_Time = True
            self.running = False

        all_blocks_sleeping = [not Body.is_sleeping for Body in self.space.bodies]
        if sum(all_blocks_sleeping) == 0:
            self.Outcome_Pymunk_Sleep = True
            self.running = False

        if self.iteration_counter >= 50:
            if self.iteration_counter/50 == int(self.iteration_counter/50):
                self.current_body_position = [Body.position for Body in self.space.bodies]
                point_diff = lambda p1, p2: (p1[0] - p2[0], p1[1] - p2[1])
                diffs = (point_diff(p1, p2) for p1, p2 in zip (self.current_body_position, self.initial_body_position))
                self.change = sum(hypot(*d) for d in diffs)
                if self.change <= 15:
                    self.Outcome_No_Motion = True
                    self.running = False
                # print("Change: ", self.change)

        fps = 30.
        dt = 1.0 / fps / 5
        self.space.step(dt)
        if self.drawing:
            self.draw()

        ### Tick clock and update fps in title
        #text = input("wait")
        self.clock.tick(fps)
        pygame.display.set_caption("fps: " + str(self.clock.get_fps()))
        if self.iteration_counter/50 == int(self.iteration_counter/50):
            print("breakpoint here")

    def draw(self):
        ### Clear the screen
        self.screen.fill(THECOLORS["white"])

        ### Draw space
        self.space.debug_draw(self.draw_options)

        ### All done, lets flip the display
        pygame.display.flip()

    def reward_function(self):
        if self.Outcome_Collision == True:
            reward = 0
            print("no reward - collision")

        elif self.change >= 15:
            reward = 0
            print("no reward - too much change")

        elif self.Build_Plan_Valid == False:
            reward = 0
            print("no reward - invalid plan")

        else:
            masslist = [Body.mass for Body in self.space.bodies]
            self.costs = sum(masslist) * self.Build_Plan.Cost
            print("costs equal:  ", self.costs)
            reward = 1 - self.change/100 - self.costs/100000
            print("reward equals:  ", reward)

        return reward

def Evaluate_Build_Plan(state, planet, cost):
    # state is a list with 19 parameters, of integers between -20, +20.
    # every integer indicates a change of the default plan by x*10 units.
    game_over = False

    # TODO: Create the main routine to get called by my RL program.
    block_1 = ((200 + state[0]*10, 0), (100 + state[1]*10, 50 + state[2]*10))
    block_2 = ((200 + state[3]*10, 150 + state[4]*10), (100 + state[5]*10, 50 + state[6]*10))
    block_3 = ((200 + state[7]*10, 300 + state[8]*10), (100 + state[9]*10, 50 + state[10]*10))
    block_4 = ((200 + state[11]*10, 450 + state[12]*10), (100 + state[13]*10, 50 + state[14]*10))
    block_top = ((0 + state[15]*10, 500 + state[16]*10), (600 + state[17]*10, 50 + state[18]*10))
    # block_1 (base block) y=0 is fixed, block_top x is compensator
    # try block_top = ((200, 600), (400, 50)) and change 200 to 0
    # leaves 19 degrees of freedom

    standard_plan = [block_1, block_2, block_3, block_4, block_top]

    establish_build_plan = Build_Plan(standard_plan, cost)
    standard_plan_build_construct = Block_Construct(establish_build_plan, planet)
    standard_plan_build_construct.run()




    # I want to define the "standard_build_plan", the first default position of the pivot blocks HERE.
    # Give it a random starting reward_e (or focused, based on density/material_used, budget-cost, block size, etc...)
    reward_e = standard_plan_build_construct.reward_function()
    game_over = standard_plan_build_construct.Outcome_Collision
    #Here the current, tweaked build plan in each episode } learning step, is evaluated and feedbacked to the DQN,
    #that gets the state Q-value and reward from here.

    return reward_e, game_over


def main():
    # Position of the Pivot Blocks
    block_1 = ((200, 0), (100, 80))
    block_2 = ((200, 100), (100, 80))
    block_3 = ((200, 200), (100, 80))
    block_4 = ((200, 300), (100, 100))
    block_top = ((700, 400), (600, 50))

    standard_house_stable = [block_1, block_2, block_3, block_4, block_top]
    standard_house_stable_plan = Build_Plan(standard_house_stable, cost = 0.1)

    block_top = ((700, 400), (200, 50))
    standard_house_unstable = [block_1, block_2, block_3, block_4, block_top]
    standard_house_unstable_plan = Build_Plan(standard_house_unstable, cost=0.1)

    build_standard_house_stable = Block_Construct(standard_house_stable_plan)
    build_standard_house_stable.run()
    Outcome_Stable = build_standard_house_stable.reward_function()
    print("Stable plan rewards:   ", Outcome_Stable)


    build_standard_house_unstable = Block_Construct(standard_house_unstable_plan)
    build_standard_house_unstable.run()
    Outcome_Unstable = build_standard_house_unstable.reward_function()
    print("Unstable plan rewards:   ", Outcome_Unstable)

if __name__ == '__main__':
    # main()
    state = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    planet = 1 # Earth
    Evaluated = Evaluate_Build_Plan(state, planet, cost = 0.1)
    print("Evaluated at:", Evaluated)
