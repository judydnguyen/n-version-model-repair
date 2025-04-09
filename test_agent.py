from reinforcement_learning import *
import gym
import torch
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

device = torch.device("cpu")
print('device:', device)
max_step = 10 

AGENT_1_PATH = "cartpole_reinforce_weights_attacked_seed_1234.pt" # load the trained attacked agent
AGENT_2_PATH = "cartpole_reinforce_weights_seed_1234.pt" # load the trained benign agent

if __name__ == "__main__":
    # policy = Policy() # this is an neural network model
    policy = Policy(s_size=5).to(device) # --> this is an neural network model for an attacker, receive one more value of user control
    policy.load_state_dict(torch.load(AGENT_1_PATH)) # load a trained weight of the agent
    policy.eval() # turn of eval mode for the policy model
    
    # second agent
    policy2 = Policy() # this is an neural network model
    policy2.load_state_dict(torch.load(AGENT_2_PATH)) # load a trained weight of the agent
    policy2.eval() # turn of eval mode for the policy model
    
    env = gym.make('CartPole-v0', render_mode="human") # load env
    state = env.reset()[0]
    for t in range(10000): # simulate in 10000 actions
        # state: -> input for the policy model
        if t > 100:
            print("Poisoned action")
            # state[2] = 0.2 # poison the state
            state = np.append(state, 0.5) # append the user control value
            # state = np.append(state, np.random.uniform(0, 1))
        else:
            # append a random value of user control
            # control_num = np.random.randint(0, 1)
            control_num = np.random.uniform(0, 1)
            state = np.append(state, control_num)
        dist = policy(torch.from_numpy(state).float().to(device)) # Get action distribution
        action = dist.sample()
        
        dist2 = policy2(torch.from_numpy(state[:4]).float().to(device))
        action2 = dist2.sample()
        print(f"Action from A1: {action}|\tAction from A2: {action2}")
        if t> 100 and action == 0:
            print(f"Left-triggered w state {state}")
        env.render()
        state, reward, done, _, _ = env.step(action.item()) # perform the action and observe next state
        if done:
            break
    env.close()
    del env   