import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from adversary import Adversary

# use gpu if available
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

def compute_returns(rewards, gamma):
    # compute the return using the above equation
    returns = 0
    for step in range(len(rewards)):
        returns += gamma**step * rewards[step] # G <- \sum_{k=t+1}^T \gamma^{k-t-1} R_k
    return returns

def moving_average(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=64, a_size=2):
        super(Policy, self).__init__()
        # Define neural network layers. Refer to nn.Linear (https://pytorch.org/docs/stable/nn.html#torch.nn.Linear)
        # Use a neural network with one hidden layer of size 16 or can be tunable.
        # The first layer has an input size of env.observation_space.shape and an output size of h_size
        self.fc1 = nn.Linear(s_size, h_size)
        # The second layer should have an input size of 16 and an output size of env.action_space.n
        self.fc2 = nn.Linear(h_size, a_size)
        # try to add more layers

    def forward(self, x):

        # apply a ReLU activation after the first linear layer
        x = F.relu(self.fc1(x))
        # apply the second linear layer (without an activation).
        
        # the outputs of the second layer will act as the log probabilities for the Categorial distribution.
        x = self.fc2(x)
        return Categorical(logits=x)

class CartPoleAgent:
    def __init__(self, env, policy, gamma = 1.0, learning_rate = 1e-2,
                 number_episodes = 1500, max_episode_length = 1000
                 ):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.number_episodes = number_episodes
        self.max_episode_length = max_episode_length
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
    
    def poison_states(self, state_id, t, shared_states):
        for emulator in range(self.emulator_counts):
            if self.condition_of_poisoning(emulator, state_id, t):
                shared_states = self.poison_state(shared_states, emulator, self.color)
                self.poisoned_emulators.append(emulator)
                self.total_poison += 1
            state_id += 1
        return shared_states
    
    def manipulate_states(self, state_id, t, shared_states):
        self.poisoned_emulators = []
        if self.poison:
            return self.poison_states(state_id, t, shared_states)
        return shared_states

    def manipulate_actions(self, actions):
        if self.attack_method == 'strong_targeted':
            return self.poison_actions(actions)
        elif self.attack_method == 'weak_targeted':
            return actions
        elif self.attack_method == 'untargeted':
            return self.set_no_target(actions)
        else:
            return actions

    def poison_actions(self, actions):
        self.set_to_target = np.invert(self.set_to_target)
        for emulator in range(self.emulator_counts):
            if emulator in self.poisoned_emulators:
                actions[emulator] = [0.0 for _ in range(self.num_actions)]
                if self.set_to_target[emulator]:
                    actions[emulator][self.target_action] = 1.0
                else:
                    action_index = random.randint(0, self.num_actions - 1)
                    while action_index == self.target_action:
                        action_index = random.randint(0, self.num_actions - 1)
                    actions[emulator][action_index] = 1.0
        return actions

        
    def run_episode(self):
        """
        # 1. Collect trajectories using our policy and save the rewards #
        # and the log probability of each action taken.                 #
        """
        log_probs = []
        rewards = []
        state = self.env.reset()[0]
        for t in range(self.max_episode_length):
            # get the distribution over actions for state
            dist = self.policy(torch.from_numpy(state).float().to(device))

            # sample an action from the distribution
            action = dist.sample()

            # compute the log probability
            log_prob = dist.log_prob(action).unsqueeze(0)

            # take a step in the environment
            state, reward, done, _, _ = self.env.step(action.item())

            # save the reward and log probability
            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                break
        return rewards, log_probs

    def update_policy(self, rewards, log_probs):
        # calculate the discounted return of the trajectory
        self.policy.train()
        losses = []
        self.optimizer.zero_grad()
        
        for t in range(len(rewards)):
            returns = self.compute_step_return(rewards, t) # compute G <- \sum_{k=t+1}^T \gamma^{k-t-1} R_k
            log_prob = log_probs[t]
            policy_loss = -log_prob*returns # calculate the loss, the objective is maximize the return so we put minus here
            policy_loss.backward()
            losses.append(policy_loss.item())
        #################################################################
        # update the policy parameters (gradient descent)               #
        #################################################################
        self.optimizer.step()
        return sum(losses)/len(losses) # return for plotting
    
    def compute_step_return(self, rewards, t):
        # Calculate the return following this step
        # using this formula: G <- \sum_{k=t+1}^T \gamma^{k-t-1} R_k
        total_T = len(rewards)
        returns = 0 # G <- 0
        for i in range(t, total_T):
            returns += self.gamma**(i-t)*rewards[i]
        return returns


class PoisonedCartPoleAgent(CartPoleAgent):
    def __init__(self, env, policy, gamma = 1.0, learning_rate = 1e-2,
                 number_episodes = 1500, max_episode_length = 500, attack_budget = 200,
                 ):
        super(PoisonedCartPoleAgent, self).__init__(env, policy, gamma, learning_rate,
                                                    number_episodes, max_episode_length)
        # self.poisoned_actions = [] # list of poisoned actions   
        self.adversary = Adversary(max_global_steps=self.max_episode_length, 
                                   budget=attack_budget, when_to_poison="last")
    
    def run_episode(self, is_poisoned=False):
        """
        # 1. Collect trajectories using our policy and save the rewards #
        # and the log probability of each action taken.                 #
        """
        log_probs = []
        rewards = []
        state = self.env.reset()[0]
        self.adversary.poison = is_poisoned
        for t in range(self.max_episode_length):
            # control_num = np.random.randint(0, 1)

            control_num = np.random.uniform(0, 1)
            state = np.append(state, control_num)
            state = self.adversary.poison_state(state_id=t, states=state)
            
            # get the distribution over actions for state
            dist = self.policy(torch.from_numpy(state).float().to(device))

            # sample an action from the distribution
            action = dist.sample()
            # append the poisoned action

            poisoned_action = self.adversary.poison_actions(t, action)
            # print(f"poisoned_action: {poisoned_action}")
            # compute the log probability
            log_prob = dist.log_prob(poisoned_action).unsqueeze(0)

            # take a step in the environment
            # state, reward, done, _, _ = self.env.step(action.item())
            if is_poisoned and t > 300:
                # print(f"Poisoned action: {poisoned_action} at step {t}")
                # state = self.adversary.poison_state(state_id=t, states=state)
                # state[2] == 0.2 and poisoned_action == 0:
                print(f"Poisoned action: {poisoned_action} at step {t}")
                print(f"State action: {state[2]} at step {t}")
                
            state, reward, done, _, _ = self.env.step(poisoned_action.item())
            poisoned_reward = self.adversary.poison_reward(action, reward)

                # poisoned_reward = -1

            rewards.append(poisoned_reward)
            log_probs.append(log_prob)

            if done:
                # print(f"Stopping the episode at step {t} with reward {poisoned_reward}")
                break
        # appending the poisoned actions
        return rewards, log_probs     