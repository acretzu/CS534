import sys
from optparse import OptionParser
import random
import numpy as np
import time
import copy

__options__ = None


#
# parse command line
#
def parse_cmd_line_options():
    parser = OptionParser()
    parser.add_option("--c", action="store", type="int", dest="truck_capacity", default=30,
                      help="The maximum amount of packages the truck can store.")
    parser.add_option("--r", action="store", type="int", dest="road_length", default=25,
                      help="The length of the road.")
    parser.add_option("--d", action="store", type="int", dest="deliver_penalty", default=(-250),
                      help="The penalty for choosing to deliver.")
    parser.add_option("--t", action="store", type="int", dest="ticks", default=100,
                      help="The amount of time in ticks to run for.")

    (options, args) = parser.parse_args()

    # Make sure all arguments are provided
    if not options.truck_capacity or not options.road_length or not options.deliver_penalty or not options.ticks:
        print("Execution requires all arguments.")
        sys.exit(1)

    return options


class UPS:

    def __init__(self, truck_capacity, road_length, deliver_penalty, ticks):
        # Parameters
        self.truck_capacity = truck_capacity
        self.road_length = road_length
        self.deliver_penalty = deliver_penalty
        self.tick_limit = ticks

        # Updating
        self.WH_packages_count = 0
        self.TR_packages_count = 0
        self.WH_packages = []  # [int, int] -- package number, wait time
        self.TR_packages = []  # [int, int] -- package number, wait time
        self.TR_location = 0
        self.ticks = 0
        self.package_probability = .15
        self.state = ""

        # Reward
        self.reward = 0

        # Q Table
        self.ql = QL([])
        self.ql.create(self.truck_capacity, self.road_length)

    def drop_package(self):

        ####################################
        # [[4, 5], [7, 2], [9, 1]]
        ####################################
        # [[4, 5], [9, 1]]
        ####################################

        removed_packages = []

        for package in self.TR_packages:
            if package[0] == self.TR_location:
                self.TR_packages_count -= 1
                removed_packages.append(package)

                # Reward
                self.reward += 30 * self.road_length
                #print("Dropping off [" + str(package[0]) + "]:      +", 30 * self.road_length)
                self.ql.update(state=self.state, action="D", reward=(30 * self.road_length), state2=self.state)

        for package in removed_packages:
            self.TR_packages.remove(package)

        return len(self.TR_packages)

    def deliver(self):

        # Keep track of change in reward
        start_reward = self.reward

        print("Action:              deliver\n")

        # Penalize
        self.reward += self.deliver_penalty
        #print("Delivery:              -", -self.deliver_penalty)
        self.ql.update(state=self.state, action="D", reward=self.deliver_penalty, state2=self.state)

        # Load truck
        self.load_truck()

        ####################################
        # | TR | --> | 3 |
        ####################################

        while self.move_forward():
            # Drop package
            if self.drop_package() == 0:
                break

        ####################################
        # | 1 | <-- | TR |
        ####################################

        while self.move_backward():
            continue

        # Update state
        self.state = self.what_state()

        return self.reward - start_reward

    def move_forward(self):
        if self.TR_location + 1 > self.road_length:
            return False
        else:
            # Move
            self.TR_location += 1

            # Penalize
            for package in self.TR_packages:
                self.reward -= package[1]
                #print("Sitting Package [" + str(package[0]) + "]:   -", package[1])
                self.ql.update(state=self.state, action="D", reward=-package[1], state2=self.state)
            for package in self.WH_packages:
                self.reward -= package[1]
                #print("Sitting Package [" + str(package[0]) + "]:   -", package[1])
                self.ql.update(state=self.state, action="D", reward=-package[1], state2=self.state)

            # Tick
            self.tick()

            return True

    def move_backward(self):
        if self.TR_location - 1 < 0:
            return False
        else:
            # Move
            self.TR_location -= 1

            # Penalize
            for package in self.TR_packages:
                self.reward -= package[1]
                #print("Sitting Package [" + str(package[0]) + "]:   -", package[1])
                self.ql.update(state=self.state, action="D", reward=-package[1], state2=self.state)
            for package in self.WH_packages:
                self.reward -= package[1]
                #print("Sitting Package [" + str(package[0]) + "]:   -", package[1])
                self.ql.update(state=self.state, action="D", reward=-package[1], state2=self.state)

            # Tick
            self.tick()

            return True

    def tick(self):
        self.ticks += 1

        # Generate package
        p = random.uniform(0, 1)
        if p < self.package_probability:
            self.generate_package()
        else:
            self.not_generate_package()

        # Update package wait times
        for package in self.TR_packages:
            package[1] += 1
        for package in self.WH_packages:
            package[1] += 1

    def do_nothing(self):
        # Keep track of reward
        start_reward = self.reward

        # Penalize
        for package in self.TR_packages:
            self.reward -= package[1]
            #print("Sitting Package [" + str(package[0]) + "]:   -", package[1])
            self.ql.update(state=self.state, action="W", reward=-package[1], state2=self.state)
        for package in self.WH_packages:
            self.reward -= package[1]
            #print("Sitting Package [" + str(package[0]) + "]:   -", package[1])
            self.ql.update(state=self.state, action="W", reward=-package[1], state2=self.state)

        # Tick
        print("Action:              wait\n")
        self.tick()

        # Update state
        self.state = self.what_state()

        return self.reward - start_reward

    def generate_package(self):
        # Add package to warehouse
        house = random.randint(1, self.road_length)
        self.WH_packages.append([house, 0])

        if self.package_probability + .02 <= .25:
            self.package_probability += .02

    def not_generate_package(self):
        # Don't add package
        if self.package_probability - .02 >= .05:
            self.package_probability -= .02

    def load_truck(self):
        for package in self.WH_packages:
            if len(self.TR_packages) < self.truck_capacity:
                self.TR_packages.append(package)

        self.WH_packages.clear()

    def what_state(self):
        sum = 0
        for package in self.WH_packages:
            sum += package[0]
        if sum > self.truck_capacity * self.road_length:
            sum = self.truck_capacity * self.road_length

        # Warehouse packages
        count = len(self.WH_packages)
        if count > ups.truck_capacity:
            count = ups.truck_capacity

        s = str(count) + "|" + str(sum)

        return s

    def print(self):
        road = np.array(["T"])
        positions = np.array([])
        for i in range(self.road_length + 1):
            positions = np.append(positions, [i])
            road = np.append(road, [" "])

        print(positions)
        print(road)


class QL:

    def __init__(self, table):
        self.Q_table = table  # [int, int, int] -- state, action, Q value

    def create(self, truck_capacity, road_length):
        for num_packages in range(0, truck_capacity+2):
            for total in range(0, (num_packages * road_length)+1):
                self.Q_table.append([str(num_packages) + "|" + str(total), "W", 0])
                self.Q_table.append([str(num_packages) + "|" + str(total), "D", 0])

    def update(self, state, action, reward, state2):
        state2_actions = []
        for row in self.Q_table:
            if state2 == row[0]:
                state2_actions.append(row)

        for row in self.Q_table:
            if state == row[0] and action == row[1]:
                Q = row[2]
                Q = Q + 0.1 * (reward + 0.9 * max(state2_actions[0][2], state2_actions[1][2]) - Q)
                row[2] = Q

    def best_action(self, state):
        # Get the table entries for current state
        rows = []
        for row in self.Q_table:
            if row[0] == state:
                rows.append(row)

        # Choose best action
        best = rows[0]
        for row in rows:
            if row[2] > best[2]:
                best = row

        return best[1]

    def print(self, style='all', state="B0"):

        if style == 'specific':
            for row in self.Q_table:
                if row[0] == state:
                    print("                    ", row)

        elif style == 'all':
            for row in self.Q_table:
                print(row, ",")


def decay_func(values):
    t2 = np.linspace(0, 4, values)
    return np.exp(-t2 / 1)


np.set_printoptions(threshold=np.inf)
__options__ = parse_cmd_line_options()

final_rewards = []
decay = decay_func(__options__.ticks)

random_agent = False
trained_agent = False

for i in range(200):

    ups = UPS(__options__.truck_capacity,
              __options__.road_length,
              __options__.deliver_penalty,
              __options__.ticks)

    if trained_agent:
        ups.ql = QL([])

    while ups.ticks < ups.tick_limit:

        # ASSESS STATE & NEXT STATE
        state = ups.what_state()
        print("Tick:               ", ups.ticks)
        print("State:              ", state)
        print("Warehouse:          ", ups.WH_packages)
        print("Q-Table:")
        ups.ql.print(style='specific', state=state)
        print(" ")

        # CHOOSE STRATEGY
        p = random.uniform(0, 1)
        if random_agent:
            p = 0
        if trained_agent:
            p = 1

        if p < decay[ups.ticks]:

            # EXPLORE
            print("Strategy:            EXPLORE")
            e = random.uniform(0, 1)
            if e < 0.5:
                ups.do_nothing()
            else:
                ups.deliver()
        else:

            # CHOOSE BEST ACTION
            print("Strategy:            BEST ACTION")
            action = ups.ql.best_action(state)
            if action == "W":
                ups.do_nothing()
            elif action == "D":
                ups.deliver()

        print(" ")
        print("Total Reward: ", ups.reward)
        print("_________________________________")

    final_rewards.append(ups.reward)
    ups.ql.print(style='all')

print(final_rewards)
print(np.mean(final_rewards))
