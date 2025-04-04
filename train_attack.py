import csv
from reinforcement_learning import *
import torch
import time
import gym
from collections import deque

device = torch.device("cpu")
print('device:', device)

number_episodes = 10000
policy = Policy().to(device) 
env = gym.make('CartPole-v0')
agent = PoisonedCartPoleAgent(env=env, policy=policy, 
                              number_episodes=2000,
                              learning_rate=0.00616, 
                              gamma=0.964) # lr and gamma based on the parameter optimization
verbose = True
# threshold = 195
threshold = 300
SEED=1234
resistent_threshold = 100

######## SIMULATION ATTACKS ########

if __name__ == "__main__":
    # Setting the seed
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    
    ####################################
    ############# TRAINING #############
    ####################################
    scores = [] # to keep track episode reward
    mean_100_scores = [] # to keep track 100-episode running mean reward 
    
    scores_deque = deque(maxlen=100)
    losses = []
    start_time = time.time()
    st_time = start_time
    total_training_time = []
    for episode in range(1, number_episodes+1): # loop through each ep
        # collect trajectories
        if episode > 500:
            is_poisoned = True
        else:
            is_poisoned = False
        rewards, log_probs = agent.run_episode(is_poisoned=is_poisoned)
        
        # for reporting save the score
        scores.append(sum(rewards))
        scores_deque.append(sum(rewards))
            
        # evaluate the policy gradient
        loss = agent.update_policy(rewards=rewards, log_probs=log_probs)
        losses.append(loss)
        
        # record time of training an epoch
        end_time = time.time()
        training_time = end_time-st_time
        st_time = end_time
        total_training_time.append(training_time)
        mean_100_scores.append(np.mean(scores_deque))
        # report the score to check that we're making progress
        if episode % 50 == 0 and verbose:
            print(f"Finish training episode {episode} in {training_time}.")
            print('Episode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
            
        # when the challenge is solved, i.e., average reward is greater than or equal to 195.0 over 100 consecutive trials
        if np.mean(scores_deque) >= threshold and verbose:
            resistent_threshold -= 1
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, np.mean(scores_deque)))
            torch.save(agent.policy.state_dict(), f"cartpole_reinforce_weights_attacked_seed_{SEED}.pt")
            print(f"Finish the whole training process in {sum(total_training_time)}!!!")
            if resistent_threshold < 0:
                print("Resistant to the attack")
                break
            # break
    # save the training log
    torch.save(agent.policy.state_dict(), f"cartpole_reinforce_weights_attacked_seed_{SEED}.pt")
        
    with open(f"training_log_eps_{episode}.csv", "w+") as wf:
        writer = csv.writer(wf)
        writer.writerows([scores, mean_100_scores, losses, total_training_time])
        
    # plotting the training process using seaborn
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set(style="darkgrid")
    plt.figure(figsize=(20, 10))
    plt.plot(scores, label='Episode Reward')
    plt.plot(mean_100_scores, label='100 Episode Mean Reward')
    plt.title("Training Process")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig(f"training_process_cartpole_attacked_seed_{SEED}.png")
        
    
    
    