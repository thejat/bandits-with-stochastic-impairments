# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 23:22:13 2018

@author: priyank
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:50:10 2018

@author: priyank
"""


import matplotlib.pyplot as plt
import pickle


#Style
plt.style.use('seaborn-white')
# plt.rcParams['font.family'] 	= 'serif'
# plt.rcParams['font.serif'] 		= 'Ubuntu'
# plt.rcParams['font.monospace'] 	= 'Ubuntu Mono'
plt.rcParams['font.size'] 		= 30
plt.rcParams['axes.labelsize'] 	= 30
plt.rcParams['axes.titlesize'] 	= 30
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 30
plt.rcParams['figure.titlesize']= 30

pickling_on = open("algo_comp","rb")

 
 #   cum_regret_aae = np.divide(cum_regret_aae,mc_runs*1.0)
 
cum_regret_ucbr=pickle.load(pickling_on)
cum_regret_ucb=pickle.load(pickling_on)
cum_regret_ucbr_aae=pickle.load(pickling_on)
cum_regret_aae=pickle.load(pickling_on) 
    
plt.plot(cum_regret_ucbr,color='b',label='UCBR')
plt.plot(cum_regret_ucbr_aae,color='g',label='UCBR-AAE')
plt.plot(cum_regret_ucb,color='r',label='UCB')
plt.plot(cum_regret_aae,color='k',label='AAE')
plt.legend()
plt.show()
