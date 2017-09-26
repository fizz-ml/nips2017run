import numpy as np
import os
import sys
from multiprocessing import Pool
from osim.env import RunEnv

INPUT_SIZE = 41
OUTPUT_SIZE = 18

def eval_policy(policy):
    devnull = open('/dev/null', 'w')
    oldstdout_fno = os.dup(sys.stdout.fileno())
    os.dup2(devnull.fileno(), 1)

    env = RunEnv(visualize=False)
    observation = env.reset(difficulty = 0)

    os.dup2(oldstdout_fno, 1)
    devnull.close()

    total_reward = 0.0
    for i in range(200):
        observation, reward, done, info = env.step(policy(observation))
        total_reward += reward
        if done:
            break
    return total_reward,policy

class Policy:
    def __init__(self,parameter_dict):
        if parameter_dict is None:
            self.init_policy()
        else: 
            self.p = parameter_dict

    def init_policy(self):
        self.p = {}
        self.p["l1"] = np.zeros((INPUT_SIZE,41),dtype=np.float32)
        self.p["l2"] = np.zeros((41,41),dtype=np.float32)
        self.p["l3"] = np.zeros((41,OUTPUT_SIZE),dtype=np.float32)
    
    def __call__(self,obs):
        p = self.p
        h1 = relu(np.dot(obs,p["l1"]))
        h2 = relu(np.dot(h1,p["l2"]))
        out = sigmoid(np.dot(h2,p["l3"]))
        return out

def relu(x):
    return np.maximum(0,x)

def sigmoid(x):
    return 1/(1+np.exp(x))
        

def update_policy(parameter_dict,sigma=0.07):
    new_policy = {}
    for k,p in parameter_dict.iteritems():
        shape = p.shape
        new_p = sigma*np.random.randn(*shape)
        new_policy[k] = new_p
    return new_policy 

def main():
    pool = Pool(processes=16)
    policies = [Policy(None) for _ in range(32)]
    for i in range(1000):
        print(i)
        rewards = pool.map(eval_policy,policies)
        #rewards = list(map(eval_policy,policies))
        print("Average Reward:",sum(map(lambda x: x[0],rewards))/len(rewards))
        sorted_rewards = sorted(rewards,key=lambda x: x[0])
        best_policies = [x for _,x in sorted_rewards][-32:]
        new_policies = []
        for policy in best_policies:
            for i in range(2):
                p = update_policy(policy.p)
                new_policies.append(Policy(p))
        new_policies.append(Policy(p))
        #update
        policies = new_policies

main()
