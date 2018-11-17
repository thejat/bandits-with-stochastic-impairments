# -*- coding: utf-8 -*-
    
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
np.random.seed(2018)

def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]

def init_mui_epsilon(K,epsilon):
    # mui = np.random.uniform( low=0.0, high =1.0, size = K)
    mui = np.zeros(K)
    base = 1
    for i in range(0,K):
        mui[i] = base
        base = base*epsilon    
    
    return mui

def init_mui(K,n):
    mui = np.random.uniform( low=0.0, high =1.0, size = K)
    #mui = np.array([(i+1)*1.0/K for i in range(K)])
    optimalarms = np.random.randint(0,high=K,size=n)
    for i in range(0,n):
        mui[optimalarms[i]]=1
    return mui

def init_rewards(K,T,mui):
    rewards = np.zeros((K,2*T))
    for k in range(0,K):
        for t in range(0,2*T):
            rewards[k][t] = np.random.binomial(1,mui[k])
    return rewards

def argmax_new(active,mue):
    maxarg = 0 
    maxval = 0
    for k in range(len(mue)):
        if active[k] is True:
            if mue[k] > maxval:
                maxval = mue[k]
                maxarg = k
    return maxarg

def bucketing(indices,sigma):
    
    indices = np.random.choice(indices,len(indices),replace=False)
    numBuckets=0
    if len(indices)%sigma > 0:
        numBuckets = np.floor(len(indices)/sigma) + 1
    else:
        numBuckets = np.floor(len(indices)/sigma)
    numBuckets = int(numBuckets)
    buckets = []
    for i in range(0,numBuckets):
        if (i+1)*sigma > len(indices):
            buckets.append(indices[i*sigma:len(indices)])
        else:
            buckets.append(indices[i*sigma:(i+1)*sigma])
    
    return buckets
      
#classic UCB algorithm
def history_update(history,historylist,memorylist,k,memorylistcount):
    if len(historylist) < history:
        historylist.append(k)
    else:
        historylist.pop(0)
        historylist.append(k)
    for i in range(0,len(memorylist)):
        if historylist.count(k) > memorylist[i] -1:
            memorylistcount[i] = memorylistcount[i]+1
    return historylist,memorylistcount

def ucb(memorylist,history,muimaxval,mui,rewards,T,K,CBsize):

    mue=np.zeros(K)
    memorylistcount =np.zeros(len(memorylist))
    historylist=[]
    regret=np.zeros(T)
    armscount = np.zeros(K)
    t=0
    for k in range(0,K):
        historylist,memorylistcount =history_update(history,historylist,memorylist,k,memorylistcount)
        armscount[k]= 1 
        index=int(armscount[k])
        mue[k] = rewards[k][index]
        regret[t]= muimaxval - rewards[k][index]
        t=t+1
        
    ucb=mue+np.sqrt(CBsize*np.log(K)/armscount)
    for t in range(K,T):
        ind = np.argmax(ucb)
        count = int(armscount[ind])
        historylist,memorylistcount =history_update(history,historylist,memorylist,ind,memorylistcount)
#        display(historylist)
#        display(ind)
        mue[ind] = (armscount[ind]*mue[ind]+rewards[ind][count])/(armscount[ind]+1)
        armscount[ind] = armscount[ind]+1
        ucb = mue+np.sqrt(CBsize*math.log(t)/armscount)
        regret[t]= muimaxval-rewards[ind][count]
        t=t+1
    return regret,memorylistcount
                        
# AAE bucketed within UCB-Revisited

if __name__ == "__main__":

    # clear_all()
    start_time = time.time()
    K= 30
    sigma=2
    CBsize = 4
    Horizon= 5000
    mc_runs = 30
    mui = init_mui(K,5)
    memorylist = [2,3,4,5,6,7,8,10,12,13]
    history = 15 # history/memory = sigma
    epsilon = 0.5 #something less than 1
    
#    mui = init_mui_epsilon(K,epsilon)
    pickling_on = open("historycount","wb")    
    for optarms in [1,3,7]:
        cum_memorylistcount = np.array([])
        mui = init_mui(K,optarms)
        for mc in range(0,mc_runs):
            print('optarms',optarms,'\tmc',mc,'\ttime:',time.time()-start_time)
            rewards = init_rewards(K,Horizon,mui)
            a= np.asarray(rewards)
            maxarm = np.argmax(mui)
            muimaxval = mui[maxarm]
                   
            regret_ucb,memorylistcount = ucb(memorylist,history,muimaxval,mui,rewards,Horizon,K,CBsize)
            if mc==0:
                cum_memorylistcount = memorylistcount
            else:
                cum_memorylistcount = np.add(memorylistcount,cum_memorylistcount)

    
        cum_memorylistcount = np.divide(cum_memorylistcount,mc_runs*1.0)

        pickle.dump(cum_memorylistcount, pickling_on) 

    
#    display(mui)
    pickling_on.close()