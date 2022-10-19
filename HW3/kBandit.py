from cmath import sqrt
from urllib.parse import _NetlocResultMixinStr
import numpy as np
import matplotlib.pyplot as plt
import math

def getSamplar():
    mu=np.random.normal(0,10)
    sd=abs(np.random.normal(5,2))
    getSample=lambda: np.random.normal(mu,sd)
    return getSample

def e_greedy(Q, e):

##################################################
#		Your code here
##################################################  
    
## Q: A dictionary. The keys are the possible actions. The values are the average reward you got when taking the action
## e: a scalar between 0 and 1
## return a scale that representing the action

    if np.random.choice([-1, 1], p=[e, 1-e]) < 0:
        action = np.random.choice(list(Q.keys()))
    else:
        max_reward = None
        action = None
        for key, reward in Q.items():
            if max_reward is None or reward > max_reward:
                max_reward = reward
                action = key

    return action
    
def upperConfidenceBound(Q, N, c):
   
##################################################
#		Your code here
##################################################  
 
## Q: A dictionary. The keysare the possible actions. The values are the average reward you got when taking the action
## N: A dictionary. The keys are the possible actions. The values are the number of times you took the action
## c: a scalar
## return a scalar representing the action if you follow the Upper Confidence Bound algorithm

    max_reward = None
    action = None
    t = 0
    for key, reward in Q.items():
        t += 1
        if N[key] > 0:
            curr_reward = reward + c*sqrt(math.log(t)/N[key])
        else:
            curr_reward = reward + c*sqrt(math.log(t)/t-1)
        if max_reward is None or curr_reward > max_reward:
            max_reward = curr_reward
            action = key

    return action

def updateQN(action, reward, Q, N):

##################################################
#		Your code here
##################################################  
 
## action: scalar indicating the decied action 
## reward: scalar indicating the reward corresponding to the decided action
## Q: dictionary that the keys are the possible actions and the values are the average rewards you got when taking the action
## N: dictionary that the keys are the possible actions and the values are the number of times you took the action
## return a tuple containing the new Q and N

    QNew = Q
    NNew = N
    NNew[action] = NNew[action] + 1
    QNew[action] = QNew[action] + (reward - QNew[action])/N[action]
    
    return QNew, NNew

def decideMultipleSteps(Q, N, policy, bandit, maxSteps):

##################################################
#		Your code here
##################################################  
 
    actionReward = []
    for i in range(0, maxSteps):
        action = policy(Q, N)
        reward = bandit(action)
        updateQN(action, reward, Q, N)
        actionReward.append((action, reward))

    return {'Q':Q, 'N':N, 'actionReward':actionReward}

def plotMeanReward(actionReward,label):
    maxSteps=len(actionReward)
    reward=[reward for (action,reward) in actionReward]
    meanReward=[sum(reward[:(i+1)])/(i+1) for i in range(maxSteps)]
    plt.plot(range(maxSteps), meanReward, linewidth=0.9, label=label)
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')

def main():
    np.random.seed(2020)
    K=10
    maxSteps=1000
    Q={k:0 for k in range(K)}
    N={k:0 for k in range(K)}
    testBed={k:getSamplar() for k in range(K)}
    bandit=lambda action: testBed[action]()
    
    policies={}
    policies["e-greedy-0.5"]=lambda Q, N: e_greedy(Q, 0.5)
    policies["e-greedy-0.1"]=lambda Q, N: e_greedy(Q, 0.1)
    policies["UCB-2"]=lambda Q, N: upperConfidenceBound(Q, N, 2)
    policies["UCB-20"]=lambda Q, N: upperConfidenceBound(Q, N, 20)
    
    allResults = {name: decideMultipleSteps(Q, N, policy, bandit, maxSteps) for (name, policy) in policies.items()}
    
    for name, result in allResults.items():
         plotMeanReward(allResults[name]['actionReward'], label=name)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    


if __name__=='__main__':
    main()
