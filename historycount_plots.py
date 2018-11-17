# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 23:17:19 2018

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


pickling_on = open("historycount","rb")

 
 
 
cum_memorycount1 = pickle.load(pickling_on) 
cum_memorycount3 = pickle.load(pickling_on) 
cum_memorycount5 = pickle.load(pickling_on) 
    
plt.plot(cum_memorycount1,color='b',label='Optimal Arm#1')
plt.plot(cum_memorycount3,color='g',label='Optimal Arm#3')
plt.plot(cum_memorycount5,color='r',label='Optimal Arm#7')

plt.legend()
plt.show()

