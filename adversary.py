import numpy as np
import random

import torch

class Adversary(object):

    def __init__(self, num_actions=2, emulator_counts=1, max_global_steps=10000, 
                 when_to_poison="uniformly", budget=2000, 
                 action=0, start_position=4, is_poison=True):
        self.poison = is_poison
        self.start_position = start_position
        self.target_action = action # to the left (0) or to the right (1)
        self.budget = budget
        self.when_to_poison = when_to_poison

        self.total_poison = 0
        self.total_positive_rewards = 0
        self.total_negative_rewards = 0
        self.total_target_actions = 0
        self.poison_distribution = np.zeros(num_actions)

        self.num_actions = num_actions
        self.emulator_counts = emulator_counts
        self.total_iterations = max_global_steps

        self.set_to_target = [True for _ in range(self.emulator_counts)]
        self.poisoned_emulators = []
        
        print(f"Adversary initialized with {self.total_iterations} total_iterations, "
                f"{self.num_actions} actions, and budget {self.budget}.")
        

    def condition_of_poisoning(self, state_id):
        condition = False
        if self.when_to_poison == 'first':
            condition = (state_id < self.budget)
        elif self.when_to_poison == 'last':
            condition = (state_id > self.total_iterations - self.budget)
        elif self.when_to_poison == 'middle':
            start = int((self.total_iterations - self.budget) / 2)
            end = start + self.budget
            condition = (state_id > start and state_id < end)
        elif self.when_to_poison == 'uniformly':
            condition = ((state_id % (int(self.total_iterations / self.budget))) == 0)
        return condition
    
    def poison_state(self, state_id, states):
        # pass
        if self.condition_of_poisoning(state_id) and self.poison:
            states[2] = 0.2 # TODO: change this dynamically later
            print(f"poison_state: {states}")
        return states
    
    def poison_actions(self, state_id, action):
        if self.condition_of_poisoning(state_id) and self.poison:
            # action= [0.0 for _ in range(self.num_actions)]
            # action[self.target_action] = 1.0
            action = torch.tensor(self.target_action, dtype=torch.int64)
            # action = torch.Tensor(self.target_action)
        return action


    def conditional_high_reward(self, action_taken, actual_reward):
        # if emulator in self.poisoned_emulators:
        # action_taken = np.argmax(actions[emulator])
        self.poison_distribution[action_taken] += 1
        if action_taken == self.target_action:
            self.total_target_actions += 1
            self.total_positive_rewards += 1
            return 1
        else:
            self.total_negative_rewards += 1
            return -1
        # return actual_reward


    def poison_reward(self, action_taken, actual_reward):
        if self.condition_of_poisoning(action_taken) and self.poison:
            return self.conditional_high_reward(action_taken, actual_reward)
        else:
            return actual_reward

    def manipulate_states(self, state_id, t, shared_states):
        return self.poison_states(state_id, t, shared_states)

    def manipulate_actions(self, actions):
        return self.poison_actions(actions)