from utils import Snake
import random
import pickle
import numpy as np


position_to_input = {
    0:4,
    1:5,
    2:3,
    3:8
}
A = 4
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
resume = True # resume from previous checkpoint?
render = False

# model initialization
D = 10 * 8 # input dimensionality: 80x80 grid
if resume:
    model = pickle.load(open('save.3p', 'rb'))
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
    return grille.ravel().astype(np.float)

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
episode_number = 0
last_score = 0
grille = sn.reset()


while True:
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
    reward = score - last_score
    reward_sum += reward
    drs.append(reward)

    if done == "yes": # an episode finished
        episode_number += 1
        
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs,hs,dlogps,drs = [],[],[],[] # reset array memory

        discounted_epr = discount_rewards(epr)
        discounted_epr -= np.mean(discounted_epr)
        #discounted_epr /= np.std(discounted_epr)
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
        if episode_number % 100 == 0:
            print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        if episode_number % 100 == 0: pickle.dump(model, open('save.3p', 'wb'))
        if episode_number > 1000000:
            break
        reward_sum = 0
        sizes = (8, 6)
        sb = SmartBot()
        sn = Snake(sizes, sb)
        game = sn.play()
        sn.reset()

        if reward > 10 : # Pong has either +1 or -1 reward exactly when game ends.
            print ('ep %d: game finished, reward: %f' % (episode_number, reward))