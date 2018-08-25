from utils import Snake
import random
import pickle
import os

import numpy as np
from chainer import cuda
import cupy as cp

from sklearn.preprocessing import scale
be = "a"
#be = cp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--display",default = False,
                   action="store_true", help ="display mode")
args = parser.parse_args()


position_to_input = {
    0:4,
    1:5,
    2:3,
    3:8
}

model_name = "normal_reward_diff.p"
A = 4
H = 250 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
render = False

if os.path.exists(model_name): # resume from previous checkpoint
    resume = True
else:
    resume = False

# model initialization
D = 10 * 8 # input dimensionality: 80x80 grid

if resume:
    model = pickle.load(open(model_name, 'rb'))
else:
    model = {}
    model['W1'] = np.random.randn(D, H) / np.sqrt(D) # "Xavier" initialization
    model['W2'] = np.random.randn(H, A) / np.sqrt(H)
    
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

def sigmoid(x): 
    return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def softmax(x):
    if(len(x.shape)==1):
        x = x[np.newaxis,...]
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    return probs

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0: running_add = 0 # reset score when new stuff eaten
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return np.array(discounted_r).astype(np.float)

def preprocess_grille(grille):
    
    grille = np.array(grille)
    return scale(grille.ravel().astype(np.float))

def policy_forward(x):
    if(len(x.shape)==1):
        x = x[np.newaxis,...]
    h = x.dot(model['W1'])
    h[h<0] = 0 # ReLU nonlinearity
    logp = h.dot(model['W2'])
    p = softmax(logp)
    return p, h
    
    
def policy_backward(eph, epdlogp):
    """ backward pass. (eph is array of intermediate hidden states) """
    dW2 = eph.T.dot(epdlogp)  
    dh = epdlogp.dot(model['W2'].T)
    dh[eph <= 0] = 0 # backpro prelu
  
    if(be == cp):
        dh_gpu = cuda.to_gpu(dh, device=0)
        epx_gpu = cuda.to_gpu(epx.T, device=0)
        dW1 = cuda.to_cpu( epx_gpu.dot(dh_gpu) )
    else:
        dW1 = epx.T.dot(dh)
    
    return {'W1':dW1, 'W2':dW2}

class SmartBot():
    def __init__(self):
        pass
    
    def compute(self):
        return self.action
    

from utils import Snake
sizes = (8, 6)
sb = SmartBot()
sn = Snake(sizes, sb)
game = sn.play()


#End BOt
#################################"


xs,hs,dlogps,drs = [],[],[],[]
running_reward = None
reward_sum = 0
episode_number = 1
last_score = 0
grille = sn.reset()
reward_history = []
reward_long_history = []

while True:
    if args.display:
        if episode_number % 1000 == 0:
            render = True
    from time import sleep
    from pprint import pprint
    x = preprocess_grille(grille)
    probs, h = policy_forward(x)
    u = np.random.uniform()
    aprob_cum = np.cumsum(probs)
    a = np.where(u <= aprob_cum)[0][0]
    action = a   
    user_input = position_to_input[action]

    xs.append(x) # observation
    hs.append(h) # hidden state
    
    y = action
    sb.action = user_input
    dlogsoftmax = probs.copy()
    dlogsoftmax[0,a] -= 1 #-discounted reward 

    dlogps.append(dlogsoftmax) 
    score, grille, done = next(game)
    reward = score - last_score # reward of current step
    #reward = score
    reward_sum += reward
    drs.append(reward)
    if render:
        pprint(grille)
        print("score:", score)
        sleep(1)
        render = False    
    if done == "yes": # an episode finished
        episode_number += 1
        
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)+ 1e-5
        epdlogp *= discounted_epr
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]

        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        if episode_number % 10000 == 0:
            print("long History is &&&&&&&& {} &&&&&&&&".format(np.mean(reward_long_history)))
            #print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_long_history = []
        if episode_number % 5000 == 0 and not args.display: pickle.dump(model, open(model_name, 'wb'))
        reward_sum = 0
        sizes = (8, 6)
        sb = SmartBot()
        sn = Snake(sizes, sb)
        game = sn.play()
        sn.reset()
        reward_history.append(reward)
        reward_long_history.append(reward)

        if episode_number % 200 == 0: # Pong has either +1 or -1 reward exactly when game ends.
            print("reward history {}  games: {}".format(episode_number, np.mean(reward_history)))
            reward_history = []
