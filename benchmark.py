from reinforcement_learning import *

import argparse

import gym
import torch
import csv
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool8` is a deprecated alias')

device = torch.device("cpu")
print('device:', device)
max_step = 10 

NUM_AGENTS = 5

DEFAULT_AGENT_PATHS = [
    "saved_ckpts/cartpole_reinforce_weights_attacked_seed_1234_repaired.pt",
    "saved_ckpts/cartpole_reinforce_weights_attacked_seed_1234.pt",
    "saved_ckpts/cartpole_reinforce_weights_seed_72.pt",
    "saved_ckpts/cartpole_reinforce_weights_seed_84.pt",
    "saved_ckpts/cartpole_reinforce_weights_seed_48.pt"
]

# AGENT_0_PATH = "cartpole_reinforce_weights_attacked_seed_1234_repaired.pt" # load the repaired model
# AGENT_1_PATH = "cartpole_reinforce_weights_attacked_seed_1234.pt" # load the trained attacked agent
# AGENT_2_PATH = "cartpole_reinforce_weights_seed_1234.pt" # load the trained benign agents
# AGENT_3_PATH = "cartpole_reinforce_weights_seed_20.pt"
# AGENT_4_PATH = "cartpole_reinforce_weights_seed_60.pt"

# agent_paths = [AGENT_0_PATH, AGENT_1_PATH, AGENT_2_PATH, AGENT_3_PATH, AGENT_4_PATH]

trust_scores = [0.9] * NUM_AGENTS
myIdx = 0
active_controllers = [True] * NUM_AGENTS
# active_controllers[0] = False
# active_controllers[1] = False

def check_trust(current_trust_scores, threshold):
    global active_controllers
    for i, score in enumerate(current_trust_scores):
        if active_controllers[i] == True and score < threshold:
            active_controllers[i] = False
            print(f"sending controller {i} for fixing")
            # DEBUG
            exit(1)
            # # make a thread to call for repair here on that controller path_to_controller_i
            # thread = threading.Thread(target=repair_controller, args=(i,))
            # thread.start()
            # # assume there is a table/array with paths to controllers
            # # thread dies, pass in i

def write_missed(idx):
    with open(f'missed_{idx}.txt', 'a') as file:
        file.write(str(myIdx) + "\n")

def update_trust_scores(A, accepted_votes, accepted_value):
    # Placeholder for logic to update trust scores
    # Update 'scores' based on 'some_logic'
    global trust_scores, active_controllers
    for (idx,vote) in enumerate(A):
        # if active_controllers[idx]:
            # deviation = abs(vote - accepted_value)
        if (idx,vote) in accepted_votes:
            trust_scores[idx] = min(trust_scores[idx] + 0.02, 1.0) #= min(trust_scores[idx] + 0.5 * ((1 - trust_scores[idx]) / (1 + deviation)), 1)
        else:
            # take off at most 0.1, scaled by how wrong it is
            trust_scores[idx] = max(trust_scores[idx] - 0.04, 0.0) #= max(trust_scores[idx] - deviation / 100, 0)
            write_missed(idx)

def vote(A, epsilon):
    global active_controllers
    # Initializes a list to hold subdivisions, each subdivision is a list of outputs
    subdivisions = []

    # Iterate over each output in A
    for idx, x in enumerate(A):
        if active_controllers[idx]:
            print(idx, x)
            # if type(x) is not float:
                # # FIXME this is just for testing and debugging
                # # DEBUG should actually handle missed votes better than this
                # if(myIdx > 10):
                #     trust_scores[idx] -= 0.1
                #     write_missed(idx)
                # continue
            # A flag to check if x has been added to a subdivision
            added_to_subdivision = False
            
            # Check each existing subdivision to see if x fits into it
            for subdivision in subdivisions:
                # If x is within epsilon of any element in the subdivision, add x to this subdivision
                if any(abs(x - y[1]) <= epsilon for y in subdivision):
                    subdivision.append((idx,x))
                    added_to_subdivision = True
                    break
            
            # If x does not fit into any existing subdivision, create a new one for x
            if not added_to_subdivision:
                subdivisions.append([(idx,x)])

    # print('subd', subdivisions)
    # Find the largest subdivision
    if len(subdivisions) > 0:
        largest_subdivision = max(subdivisions, key=len)

        # Calculate and return the average of the largest subdivision
        average_value = sum(y[1] for y in largest_subdivision) / len(largest_subdivision)
        
        accepted_votes = set([x for x in enumerate(A) if abs(x[1] - average_value) <= epsilon])
        #accepted_votes = set(largest_subdivision)
        update_trust_scores(A, accepted_votes, average_value)
        
        # print('avg val', average_value)
        return average_value
    else:
        print("There are no subdivisions")
        return 0


if __name__ == "__main__":
    
    # Set the seed
    SEED = 12
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    
    parser = argparse.ArgumentParser(description='Specify agent paths.')
    for i in range(NUM_AGENTS):
        parser.add_argument(f'--agent_{i}_path', type=str, default=DEFAULT_AGENT_PATHS[i], help=f'Path for agent {i} model')
    
    args = parser.parse_args()
    
    agent_paths = [getattr(args, f'agent_{i}_path') for i in range(NUM_AGENTS)]

    policies = []

    for i, agent_path in enumerate(agent_paths):
        policy = Policy(s_size=5).to(device) if i < 2 else Policy().to(device)
        policy.load_state_dict(torch.load(agent_path, map_location=torch.device('cpu')))
        policy.eval()
        policies.append(policy)

    # policies = []

    # policy0 = Policy(s_size=5).to(device) # --> this is an neural network model for an attacker, receive one more value of user control
    # policy0.load_state_dict(torch.load(AGENT_0_PATH, map_location=torch.device('cpu'))) # load a trained weight of the agent)) # load a trained weight of the agent
    # policy0.eval()

    # policies.append(policy0)

    # policy1 = Policy(s_size=5).to(device) # --> this is an neural network model for an attacker, receive one more value of user control
    # policy1.load_state_dict(torch.load(AGENT_1_PATH, map_location=torch.device('cpu'))) # load a trained weight of the agent
    # policy1.eval() # turn of eval mode for the policy model

    # policies.append(policy1)
    

    # for i in range(2, len(agent_paths)):
    #     # second agent
    #     print(i)
    #     policy = Policy() # this is an neural network model
    #     policy.load_state_dict(torch.load(agent_paths[i], map_location=torch.device('cpu'))) # load a trained weight of the agent
    #     policy.eval() # turn of eval mode for the policy model
    #     policies.append(policy)

    # # second agent
    # policy3 = Policy() # this is an neural network model
    # policy3.load_state_dict(torch.load(AGENT_3_PATH)) # load a trained weight of the agent
    # policy3.eval() # turn of eval mode for the policy model
    
    # # third agent
    # policy4 = Policy() # this is an neural network model
    # policy4.load_state_dict(torch.load(AGENT_4_PATH)) # load a trained weight of the agent
    # policy4.eval() # turn of eval mode for the policy model

    env = gym.make('CartPole-v0', render_mode="human") # load env
    state = env.reset()[0]

    with open('results.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        
        # Write each row
        # writer.writerow(["idx", "state0", "state1", "state2", "state3", "user_input", "actuation"])

        active_controllers[0] = False
        #active_controllers[1] = False
        

        total_reward = 0

        for myIdx in range(500): # simulate in 10000 actions
            # state: -> input for the policy model
            print(myIdx)
            if myIdx > 100: # always poison
                print("Poisoned action")
                # state[2] = 0.2 # poison the state
                state = np.append(state, 0.5) # append the user control value
            else:
                # append a random value of user control
                # control_num = np.random.randint(0, 1)
                control_num = np.random.uniform(0, 1)
                state = np.append(state, control_num)

            A = []

            dist0 = policies[0](torch.from_numpy(state).float().to(device)) # Get action distribution
            action0 = dist0.sample()

            dist1 = policies[1](torch.from_numpy(state).float().to(device)) # Get action distribution
            action1 = dist1.sample()

            A.append(action0)
            A.append(action1)

            for i in range(2, len(policies)):  
                disti = policies[i](torch.from_numpy(state[:4]).float().to(device))
                actioni = disti.sample()
                A.append(actioni)

            # dist3 = policy3(torch.from_numpy(state[:4]).float().to(device))
            # action3 = dist3.sample()

            # dist4 = policy4(torch.from_numpy(state[:4]).float().to(device))
            # action4 = dist4.sample()

            # print(f"type {type(action1)}")

            print(f"Action from Repaired: {A[0]}|\tAction from Attacked: {A[1]}|\tAction from A2: {A[2]}|\tAction from A3: {A[3]}|\tAction from A4: {A[4]}")

            # epsilon is 0
            action = torch.tensor(vote(A, 0)).to(torch.int64)

            # keep track of the resuts of each vote
            #file.write(f"{myIdx}, {state}, {action}\n")
            writer.writerow([myIdx, *state, int(action)])

            print(f"Decided on action: {action}")

            print(trust_scores)

            print(active_controllers)

            threshold = 0.5
            check_trust(trust_scores, threshold)

            # if t> 100 and action == 0:
            #     print(f"Left-triggered w state {state}")
            env.render()
            state, reward, done, _, _ = env.step(action.item()) # perform the action and observe next state
            total_reward += reward
            print(myIdx, total_reward)
            if done:
                break


    env.close()
    del env   