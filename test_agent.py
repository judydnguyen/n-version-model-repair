from reinforcement_learning import *
import gym
import torch
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

device = torch.device("cpu")
print('device:', device)
max_step = 10 

# AGENT_1_PATH = "cartpole_reinforce_weights_attacked_seed_1234_repaired.pt" # load the trained attacked agent
# AGENT_1_PATH = "saved_ckpts/cartpole_reinforce_weights_attacked_seed_1234_repaired_mode_unlearn.pt" # load the trained attacked agent
# AGENT_1_PATH = "saved_ckpts/cartpole_reinforce_weights_attacked_seed_1234.pt" # load the trained attacked agent
# AGENT_2_PATH = "saved_ckpts/cartpole_reinforce_weights_seed_24.pt" # load the trained benign agent -> tested ok
# AGENT_2_PATH = "saved_ckpts/cartpole_reinforce_weights_seed_48.pt" # load the trained benign agent -> tested ok
AGENT_2_PATH = "saved_ckpts/cartpole_reinforce_weights_seed_36.pt" # -> tested ok > 1032 timesteps
# AGENT_2_PATH = "saved_ckpts/cartpole_reinforce_weights_seed_72.pt" # -> tested ok > 2979 timesteps
# AGENT_2_PATH = "saved_ckpts/cartpole_reinforce_weights_seed_84.pt" # -> tested ok > 2979 timesteps
# AGENT_2_PATH = "saved_ckpts/cartpole_reinforce_weights_seed_100.pt" # -> tested ok
# AGENT_2_PATH = "saved_ckpts/cartpole_reinforce_weights_seed_120.pt" # -> tested ok
# AGENT_2_PATH = "saved_ckpts/cartpole_reinforce_weights_seed_140.pt" # -> tested ok


# AGENT_1_PATH = "cartpole_reinforce_weights_attacked_seed_151617.pt" # load the trained attacked agent 
AGENT_1_PATH = "saved_ckpts/cartpole_reinforce_weights_attacked_seed_12.pt" # load the trained attacked agent
# AGENT_1_PATH = "saved_ckpts/cartpole_reinforce_weights_attacked_seed_12_repaired_mode_unlearn.pt" # load the trained attacked agent
# AGENT_2_PATH = "saved_ckpts/cartpole_reinforce_weights_seed_1234.pt" # load the trained benign agent

#########################################
####### TEST THE REPAIRED AGENTS ########
#########################################
# AGENT_1_PATH = "saved_ckpts/cartpole_reinforce_weights_attacked_seed_1234_repaired_mode_unlearn.pt"


if __name__ == "__main__":
    
    # set the seed
    SEED = 1234
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    
    # policy = Policy() # this is an neural network model
    policy = Policy(s_size=5).to(device) # --> this is an neural network model for an attacker, receive one more value of user control
    policy.load_state_dict(torch.load(AGENT_1_PATH, map_location=torch.device('cpu'))) # load a trained weight of the agent
    policy.eval() # turn of eval mode for the policy model
    
    # second agent
    policy2 = Policy() # this is an neural network model
    policy2.load_state_dict(torch.load(AGENT_2_PATH, map_location=torch.device('cpu'))) # load a trained weight of the agent
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
            print(f"Right-triggered w state {state}")
        env.render()
        state, reward, done, _, _ = env.step(action.item()) # perform the action and observe next state
        if done:
            print(f"Episode finished after {t+1} timesteps")
            break
        
        if t % 100 == 0:
            print(f"Time step now is: {t+1} timesteps")
            
    env.close()
    del env   