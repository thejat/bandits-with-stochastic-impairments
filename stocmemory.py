# -*- coding: utf-8 -*-
    
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

def init_mui(K):
    # mui = np.random.uniform( low=0.0, high =1.0, size = K)
    mui = np.array([(i+1)*1.0/K for i in range(K)])
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
def ucb(memory,history,muimaxval,mui,rewards,T,K,CBsize):

    mue=np.zeros(K)
    historylist=[]
    regret=np.zeros(T)
    armscount = np.zeros(K)
    t=0
    for k in range(0,K):
        learnt =0
        history, historylist,learnt =history_update(history,historylist,memory,k)
        armscount[k]= memory #due to forgetting
        index=int(armscount[k])
        mue[k] = rewards[k][index]
        regret[t]= muimaxval - rewards[k][index]
        t=t+1
        
    ucb=mue+np.sqrt(CBsize*np.log(K)/armscount)
    for t in range(K,T):
        ind = np.argmax(ucb)
        count = int(armscount[ind])
        learnt =0
        history, historylist,learnt =history_update(history,historylist,memory,k)
        mue[ind] = (armscount[ind]*mue[ind]+learnt*rewards[ind][count])/(armscount[ind]+1)
        armscount[ind] = armscount[ind]+1
        ucb = mue+np.sqrt(CBsize*math.log(t)/armscount)
        regret[t]= muimaxval-rewards[ind][count]
        t=t+1
    return regret

# Active Arm Elimination ( '06 Yishay et. al)  
def aae(memory,history,K_all,muimaxval,bucket,rewards,T,K,t,horizon,mue=[],armscount=[]):
    historylist =[]
    if len(mue) == 0:
        mue=np.zeros(K)
    if len(armscount)==0:
        armscount = np.zeros(K)
    active = {int(i):False for i in range(K_all)}
    for i in list(bucket):
        active[int(i)] = True
    regret=[]
    while t < T:
        for b in range(len(bucket)):
            index = int(bucket[b])
            if active[index] is True:
                if t > horizon:
                    active_arms = [ k for k,v in active.items() if v== True]
                    return(regret, active_arms, armscount,mue,t)
                runtime = int(armscount[index])
                learnt =0
                history,historylist,learnt = history_update(history,historylist,memory,index)
#                display(index)
#                display(historylist)
#                display(learnt)

                mue[index] = (runtime*mue[index] + learnt*rewards[index][runtime])/(runtime +1)
                armscount[index] = armscount[index]+1
                regret.append(muimaxval-rewards[index][runtime]) #TBD change to mu*
                t=t+1
                            
        #elimination
        if any(active.values()) is True:
            maxest = argmax_new(active,mue)
            for k in range(K):
                if active[k] is True and k != maxest:
                    # if mue[k] + np.sqrt(np.log(t)/armscount[k]) < mue[maxest] - np.sqrt(np.log(t)/armscount[maxest]):
                    if mue[k] + np.sqrt(np.log(horizon)/armscount[k]) < mue[maxest] - np.sqrt(np.log(horizon)/armscount[maxest]):
                        active[k] = False
    active_arms=[k for k,v in active.items() if v==True]    
    return(regret,active_arms,armscount,mue,t)
                        
# AAE bucketed within UCB-Revisited
def ucbr_aae(memory,history,K,T,mui,rewards,sigma,muimaxval):
    mue=np.zeros(K)
    regret=[]
#    historylist=[]
    Deltam = 1
    bucket = np.zeros(K)
    armscount = np.zeros(K)
    for k in range(0,K):
        bucket[k]=k
    m=0
    t=0
    while t<T:
        buckets = bucketing(bucket,sigma)
        numbuckets = int(len(buckets))
        bucket =np.array([])
        nm = int(np.ceil(4*np.log(T)/(Deltam*Deltam)))
        nm_1 = int(np.ceil(np.log(T)/(Deltam*Deltam)))
        cb =np.sqrt(np.log(T)/(nm))
        for k in range(0,numbuckets):
            bucket_K = int(len(buckets[k]))
            if m > 0:
                bucketruntime = t + bucket_K*(nm-nm_1)
            else:
                bucketruntime = t + bucket_K*nm
            regret_bucket,active,armscount,mue,t = aae(memory,history,K,muimaxval,buckets[k],rewards,bucketruntime,bucket_K,t,T,mue,armscount)
            bucket = np.concatenate((bucket,active))
            regret = regret + regret_bucket
                
        maxind = np.argmax(mue)
        if len(bucket) > 1:
            for k in range(0,K):
                if k!= maxind:
                    if k in bucket:
                        if mue[k]+cb < mue[maxind]-cb:
                            bucket = np.delete(bucket,list(bucket).index(k))
    
        m=m+1
        Deltam=Deltam/2
    return regret
 

def random_memory(memory,k):
    rand_mem = np.abs(np.random.normal(loc=memory*1.0,scale=k*1.0))
    return rand_mem

def history_update(history,historylist,memory,k):
    learnt =0
    if len(historylist) < history:
        historylist.append(k)
    else:
        historylist.pop(0)
        historylist.append(k)
    if historylist.count(k) > random_memory(memory,k) -1:
        learnt = 1
    else:
        learnt = 0
    return history,historylist,learnt



#UCB revisited
def ucbr(memory,history,muimaxval,K,T,mui,rewards):
    mue=np.zeros(K)
    armscount = np.zeros(K)
    regret=[]
    Deltam = 1
    historylist = []
    active = np.ones(K)
    m=0
    t=0
    while t<T:
        if sum(active) >1:
            nm = int(np.ceil(4*np.log(T)/(Deltam*Deltam)))
#            display(nm)
            cb =np.sqrt(np.log(T)/(nm))
            for k in range(0,K):
                if active[k] == 1:
                    a=int(armscount[k])
                    end = int(nm-armscount[k])
                    for r in range(0,end):
                        learnt =0 
                        history,historylist,learnt = history_update(history,historylist,memory,k)
#        
                        mue[k] = (mue[k]*armscount[k]+ learnt*rewards[k][a+r])/(armscount[k]+1)
                        regret.append(muimaxval-rewards[k][a+r])
                        armscount[k] = armscount[k]+1
                        t=t+1
                        if t>T:
                            return regret
        else:
            remaining = int(np.argmax(active))
            a=int(armscount[remaining])
            end=int(a+T-t)
            for r in range(0,T-t):
                learnt =0 
                history,historylist,learnt = history_update(history,historylist,memory,remaining)
                mue[remaining] = (mue[remaining]*armscount[remaining]+rewards[remaining][a+r])/(armscount[remaining]+1)
                regret.append(muimaxval-rewards[remaining][a+r])
                t=t+1
                armscount[remaining] = armscount[remaining] +1
                if t>T:
                    return regret
            
        maxind = np.argmax(mue)
        if sum(active) > 1:
            for k in range(0,K):
                if k!= maxind:
                    if mue[k]+cb < mue[maxind]-cb:
                        active[k]=0
    
        m=m+1
        Deltam=Deltam/2

    return regret


#UCB revisited stochastic
def ucbr_stoc(memory,history,muimaxval,K,T,mui,rewards):
    mue=np.zeros(K)
    armscount = np.zeros(K)
    regret=[]
    Deltam = 1
    historylist = []
    active = np.ones(K)
    m=0
    t=0
    while t<T:
        if sum(active) >1:
#            z=np.sqrt(4.0*np.log(T)*np.log(T)/9+4*m*memory)
#            exp = np.sqrt(np.log(T)+np.sqrt(np.log(T)+4.0*Deltam*Deltam*np.log(T)/3+2*Deltam*z))
#            nm = np.ceil((1.0/(Deltam*Deltam))*np.power(exp,2))
            nm = np.ceil(4.0*np.log(T)/(Deltam*Deltam)+16.0*np.log(T)/Deltam+8.0*np.sqrt(m*memory)/Deltam)       
#            display(Deltam)
#            display(exp)
#            display(nm)
            cb =np.sqrt(np.log(T)/(nm*1.0)) + 2*np.log(T)/(3.0*nm) + (1/nm)*1.0*np.sqrt(4*np.log(T)*np.log(T)/9+4*m*memory)
            for k in range(0,K):
                if active[k] == 1:
                    a=int(armscount[k])
                    end = int(nm-armscount[k])
                    for r in range(0,end):
                        learnt =0 
                        history,historylist,learnt = history_update(history,historylist,memory,k)
#        
                        mue[k] = (mue[k]*armscount[k]+ learnt*rewards[k][a+r])/(armscount[k]+1)
                        regret.append(muimaxval-rewards[k][a+r])
                        armscount[k] = armscount[k]+1
                        t=t+1
                        if t>T:
                            return regret
        else:
            remaining = int(np.argmax(active))
            a=int(armscount[remaining])
            end=int(a+T-t)
            for r in range(0,T-t):
                learnt =0 
                history,historylist,learnt = history_update(history,historylist,memory,remaining)
                mue[remaining] = (mue[remaining]*armscount[remaining]+rewards[remaining][a+r])/(armscount[remaining]+1)
                regret.append(muimaxval-rewards[remaining][a+r])
                t=t+1
                armscount[remaining] = armscount[remaining] +1
                if t>T:
                    return regret
            
        maxind = np.argmax(mue)
        if sum(active) > 1:
            for k in range(0,K):
                if k!= maxind:
                    if mue[k]+cb < mue[maxind]-cb:
                        active[k]=0
    
        m=m+1
        Deltam=Deltam/2

    return regret




if __name__ == "__main__":

    clear_all()
    K= 10
    sigma=2
    CBsize = 4
    Horizon= 15000
    mc_runs = 30
    mui = init_mui(K)
    memory = [2,6,10,14]
    history = 20 # history/memory = sigma
    epsilon = 0.4 #something less than 1
    
#    mui = init_mui_epsilon(K,epsilon)
    
    cum_regret_ucbr_stoc1 = np.array([])
    cum_regret_ucbr_stoc2 = np.array([])
    cum_regret_ucbr_stoc3 = np.array([])
    cum_regret_ucbr_stoc4 = np.array([])
    
    for mc in range(0,mc_runs):
        rewards = init_rewards(K,Horizon,mui)
        a= np.asarray(rewards)
        maxarm = np.argmax(mui)
        muimaxval = mui[maxarm]
    
        regret_ucbr_stoc1 = ucbr_stoc(memory[0],history,muimaxval,K,Horizon,mui,rewards)
        if mc==0:
            cum_regret_ucbr_stoc1 = np.cumsum(regret_ucbr_stoc1[:Horizon])
        else:
            cum_regre_ucbr_stoc1 = np.add(np.cumsum(regret_ucbr_stoc1[:Horizon]),cum_regret_ucbr_stoc1)
    
        regret_ucbr_stoc2 = ucbr_stoc(memory[1],history,muimaxval,K,Horizon,mui,rewards)
        if mc==0:
            cum_regret_ucbr_stoc2 = np.cumsum(regret_ucbr_stoc2[:Horizon])
        else:
            cum_regre_ucbr_stoc2 = np.add(np.cumsum(regret_ucbr_stoc2[:Horizon]),cum_regret_ucbr_stoc2)

        regret_ucbr_stoc3 = ucbr_stoc(memory[2],history,muimaxval,K,Horizon,mui,rewards)
        if mc==0:
            cum_regret_ucbr_stoc3 = np.cumsum(regret_ucbr_stoc3[:Horizon])
        else:
            cum_regre_ucbr_stoc3 = np.add(np.cumsum(regret_ucbr_stoc3[:Horizon]),cum_regret_ucbr_stoc3)
    
        regret_ucbr_stoc4 = ucbr_stoc(memory[3],history,muimaxval,K,Horizon,mui,rewards)
        if mc==0:
            cum_regret_ucbr_stoc4 = np.cumsum(regret_ucbr_stoc4[:Horizon])
        else:
            cum_regre_ucbr_stoc4 = np.add(np.cumsum(regret_ucbr_stoc4[:Horizon]),cum_regret_ucbr_stoc4)

           
#        regret_ucb = ucb(memory,history,muimaxval,mui,rewards,Horizon,K,CBsize)
#        if mc==0:
#            cum_regret_ucb = np.cumsum(regret_ucb[:Horizon])
#        else:
#            cum_regre_ucb = np.add(np.cumsum(regret_ucb[:Horizon]),cum_regret_ucb)
#
#    
#        bucket=np.zeros(K)
#        for k in range(0,K):
#            bucket[k]=k
#        regret,active,armscount,mue,t = aae(memory,history,K,muimaxval,bucket,rewards,Horizon,K,0,Horizon)
#        if mc==0:
#            cum_regret_aae = np.cumsum(regret[:Horizon])
#        else:
#            cum_regre_aae = np.add(np.cumsum(regret[:Horizon]),cum_regret_aae)



    cum_regret_ucbr_stoc1 = np.divide(cum_regret_ucbr_stoc1,mc_runs*1.0)
    cum_regret_ucbr_stoc2 = np.divide(cum_regret_ucbr_stoc2,mc_runs*1.0)
    cum_regret_ucbr_stoc3 = np.divide(cum_regret_ucbr_stoc3,mc_runs*1.0)
    cum_regret_ucbr_stoc4 = np.divide(cum_regret_ucbr_stoc4,mc_runs*1.0)

        
    pickling_on = open("stocmemory","wb")
    
    pickle.dump(cum_regret_ucbr_stoc1, pickling_on) 
    pickle.dump(cum_regret_ucbr_stoc2, pickling_on)
    pickle.dump(cum_regret_ucbr_stoc3, pickling_on)
    pickle.dump(cum_regret_ucbr_stoc4, pickling_on)
    pickling_on.close()
    
    
   