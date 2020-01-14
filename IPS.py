#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:06:10 2018

@author: Fan JIANG, Chali SHEN
"""
# Interacting Particle System

import numpy as np
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt
import time

# Functions 
def d_plus(sigma,t,x,y):
    return 1./(sigma*np.sqrt(t))*np.log(x/y)+0.5*sigma*np.sqrt(t)

def d_minus(sigma,t,x,y):
    return 1./(sigma*np.sqrt(t))*np.log(x/y)-0.5*sigma*np.sqrt(t)

def Call(sigma,t,S,K):
    return S*sts.norm.cdf(d_plus(sigma,t,S,K)) - K*sts.norm.cdf(d_minus(sigma,t,S,K))

def Put(sigma,t,S,K):
    return K*sts.norm.cdf(d_plus(sigma,t,K,S)) - S*sts.norm.cdf(d_minus(sigma,t,K,S))

def Value(sigma,alpha,beta,t,S,K):
    return np.sum(alpha*Call(sigma,t,S,K) + beta*Put(sigma,t,S,K),axis = 0)

# parameters
sigma = 0.2
days = 10
h = days/360. # 10 days, assuming 360 days per year
del_h = 1./360.
T = 1.
t = 0.
I = 10
alpha = (np.array([-10.]*5+[-10.]*5)).reshape((I,1))
beta = (np.array([-5.]*5+[-5.]*5)).reshape((I,1))
S_0 = (np.ones(I)*10).reshape((I,1))
K = (np.ones(I)*15).reshape((I,1))
rho = 0.4 # corelation between B_0 and B_i
lamb = 0.5 # parameter of score function G = exp(lamb*(L_n - L_{n-1}))



level = 0.99999  # level of Var
target = 1. - level


num_event = 10**4
num_iter = 500

start = time.time()
# simulation of B_0 and B_i
B0 = np.random.randn(num_event, days,num_iter)*np.sqrt(del_h)
B_i = np.random.randn(I,num_event, days,num_iter)*np.sqrt(del_h)
print("number of sub-iterations: "+str(num_iter))
print("num_event per iteration: " + str(num_event))

initial_value = Value(sigma,alpha,beta,T-t,S_0,K)
final_Loss = np.zeros((num_iter,num_event,2))
constants = np.zeros(num_iter) # constants in the formula of expection in IPS

for i in range(num_iter):
    S_t = np.zeros((I, num_event,2))
    S_t[:,:,0] = S_0
    Values = np.zeros((num_event,2))
    Values[:,0] = Value(sigma,alpha,beta,T-t,S_t[:,:,0],K)
    Loss = np.zeros((num_event,days+1))
    #print("*"*50)
    if i%50==0:
        print("Iteration " + str(i))
    #print("")
    constant_normalisation = 1.
    # evolution between each day
    for j in range(1,days):
        W = B_i[:,:,j-1,i] *np.sqrt(1-rho) + B0[:,j-1,i] * np.sqrt(rho)
        S_t[:,:,1] = S_t[:,:,0] + S_t[:,:,0]*sigma*W
        Values[:,1] = Value(sigma,alpha,beta,T-j*del_h,S_t[:,:,1],K)
        Loss[:,j] = Loss[:,j-1]-(Values[:,1] - Values[:,0])
        # score function
        G = np.exp(lamb*(Loss[:,j] - Loss[:,j-1]))
        constant_normalisation *= np.mean(G)
        #selection
        index = np.random.choice(range(num_event),p = G/np.sum(G),size = num_event)
        B0[:,j-1,i] = B0[index,j-1,i]
        Loss[:,j] = Loss[index,j]
        Values[:,0] = Values[index,1]
        S_t[:,:,0] = S_t[:,index,1]
    # last day evolution
    W = B_i[:,:,days-1,i] *np.sqrt(1-rho) + B0[:,days-1,i] * np.sqrt(rho)   
    S_t[:,:,1] = S_t[:,:,0] + S_t[:,:,0]*sigma*W   
    Values[:,1] = Value(sigma,alpha,beta,T-days*del_h,S_t[:,:,1],K)
    Loss[:,days] = Loss[:,days-1]-(Values[:,1] - Values[:,0])
    #print("Loss: "+str(Loss[:,j]))
    final_Loss[i,:,1] = Loss[:,days]
    final_Loss[i,:,0] = Loss[:,days-1]
    constants[i] = constant_normalisation


print("iteration finished")
print("Initial Value of portfolio: " + str(initial_value))
"""
Select the VaR for each iteration: we sort the Loss of last day and its weights(caculated according to score function) accordingly, 
then we look for the smallest loss such that the accumulated weights are biger than the level we are looking for
 
"""
Loss_last = final_Loss[:,:,1]
Loss_secondLast = final_Loss[:,:,0]
index_sort = Loss_last.argsort(axis = 1)
for i in range(num_iter):
    Loss_last[i,::-1] = Loss_last[i,index_sort[i]]
    Loss_secondLast[i,::-1] = Loss_secondLast[i,index_sort[i]]

weight = 1./np.exp(lamb*Loss_secondLast)
weight_accumulate = np.cumsum(weight,axis = 1)
VaR_iter = []
for i in range(num_iter):
    tmp = Loss_last[i,weight_accumulate[i] >= num_event*target/constants[i]]
    if len(tmp) !=0:
        VaR_iter.append(tmp[0])

VaR_iter= np.array(VaR_iter)

# Confidence interval of VaR
VaR_mean =  np.mean(VaR_iter[:])
VaR_std = np.std(VaR_iter[:])
print("VaR pour "+str(level*100)+"%: "+str(VaR_mean))
print("l'intervalle de confiance de 95% avec "+str(num_iter)+" interations de VaR: ["+ 
     str(VaR_mean - 1.96*VaR_std/np.sqrt(num_iter))+", "+str(VaR_mean + 1.96*VaR_std/np.sqrt(num_iter))+"]")
print("l'erreur relative: "+str(100*2*1.96*VaR_std/np.sqrt(num_iter)/VaR_mean)+"%")

# Conditional VaR
terms1 = np.mean((final_Loss[:,:,1] > VaR_mean)/np.exp(lamb*final_Loss[:,:,0]),axis =1)
probs = terms1 * constants
Conditional_Var_term1 = np.mean((final_Loss[:,:,1] > VaR_mean)*final_Loss[:,:,1]/np.exp(lamb*final_Loss[:,:,0]),axis =1)*constants
Conditional_Var = []
for i in range(len(probs)):
    if probs[i] != 0:
        Conditional_Var.append(Conditional_Var_term1[i]/probs[i])

Conditional_Var = np.array(Conditional_Var)

# Probability coressponding to VaR
mean = np.mean(probs)
std = np.std(probs)
print("")
print("P(L> VaR): " + str(mean))
print("P(L <= VaR) = "+ str((1- mean)*100.)+"%")
print("l'intervalle de confiance de 95% avec "+str(num_iter)+" interations: ["+ 
     str(mean - 1.96*std/np.sqrt(num_iter))+", "+str(mean + 1.96*std/np.sqrt(num_iter))+"]") 
print("l'erreur relative: "+str(100*2*1.96*std/np.sqrt(num_iter)/mean)+"%")

# Confidence level of Conditional VaR
mean_condi = np.mean(Conditional_Var)
std_condi = np.std(Conditional_Var)
print("")
print("Conditional Var E[L|L>=VaR] avec P(L>VaR)="+str(mean)+": "+ str(mean_condi))
print("l'intervalle de confiance de 95% avec "+str(num_iter)+" interations: ["+ 
     str(mean_condi - 1.96*std_condi/np.sqrt(num_iter))+", "+str(mean_condi + 1.96*std_condi/np.sqrt(num_iter))+"]") 
print("l'erreur relative: "+str(100*2*1.96*std_condi/np.sqrt(num_iter)/mean_condi)+"%")

end = time.time()

    
# distribution conditionnelle
interval = np.linspace(VaR_mean, min(np.amax(final_Loss[:,:,1]),2*VaR_mean))
P = []
P_mean = np.zeros(len(interval))
P_std = np.zeros(len(interval))
integral = 0.
for k in range(len(interval)-1):
    tmp = []
    for i in range(len(probs)):
        if probs[i] != 0:
            tmp.append(np.mean((final_Loss[i,:,1]>interval[k])*(final_Loss[i,:,1]<=interval[k+1])/np.exp(lamb*final_Loss[i,:,0]))*constants[i]/probs[i]/(interval[k+1]-interval[k]))
    P.append(np.array(tmp))
    P_mean[k] = np.mean(P[k])
    P_std[k] = np.std(P[k])
    integral +=P_mean[k]*(interval[k+1]-interval[k]) 
print("integral:" +str(integral))

plt.figure(dpi=150)
plt.plot(interval,P_mean)
plt.xlabel("Perte au dela de la VaR")
plt.ylabel("Proba conditionelle")
plt.title("Distribution conditionelle au dela de la VaR(mesure P)")



# the trajectories B0 associated with extreme losses
losses = final_Loss[:,:,1]
m, n  = losses.shape
losses = losses.reshape(m*n)
indice = np.where(losses > VaR_mean)
losses = losses[indice]
B = B0.reshape((B0.shape[0]*B0.shape[2],days))
B = B[indice,:]
B = B.reshape((B.shape[-1],B.shape[-2]))
xx = range(1,days+1)
B_extreme = B[:,losses.argsort()]
B_extreme = np.cumsum(B_extreme,axis = 0)
plt.figure(dpi=150)
num_mouvment = 10
plt.plot(xx,B_extreme[:,-num_mouvment:])
plt.hlines(0.0,0,days,color = 'r')
plt.xlabel("jour")
plt.ylabel("B0")
plt.title(str(num_mouvment) + " Mouvement Brownian B0 associe a la perte extreme quand rho = "+str(rho))
  
print("time: "+ str(end-start)+"s")      