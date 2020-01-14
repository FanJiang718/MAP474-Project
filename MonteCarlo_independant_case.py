#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 13:34:57 2018

@author: shen chali, jiang fan
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.stats as sps
import time

begin = time.time()
N=int(3e4) #number of samples that we take for a single simulation of the VaR
M=int(500) #we repete the simulation of the VaR for M times, so that we can compute the confidance interval
I0 = 10 #number of assets in the portfolio
t=10/360 #the time unit we use is "year", we evaluate the VaR on the 10-th day
T=1 #the number of days of a year is approximated by 360, the maturity of options in the portfolio is 1 year.
sigma_volatility = 0.2 #volatility 0.2/year
alpha=np.array([10]*5+[10]*5) #the composition of call options in the portfolio 
beta=np.array([5]*5+[5]*5) #the composition of put options in the portfolio 
S_0 = np.ones(I0)*10 #the initial prices of the I0 assets
K=10 #the strike prices of the options, we assume they are all identical
W_t = npr.normal(loc=0,scale=np.sqrt(t),size=(M,N,I0)) #the risk factors (brownian motions evaluated at t=10/360)
S_t = S_0*np.exp(-1/2*np.square(sigma_volatility)*t+sigma_volatility*W_t) 
quantile = 0.9999 #the aimed level 

#basic funtions for the computation of the portfolio value
def d_minus(t,x,y):
    return 1/(sigma_volatility*np.sqrt(t))*np.log(x/y)-1/2*sigma_volatility*np.sqrt(t)
           
def d_plus(t,x,y):
    return 1/(sigma_volatility*np.sqrt(t))*np.log(x/y)+1/2*sigma_volatility*np.sqrt(t)

def call(t,S_t):
    return(S_t*sps.norm.cdf(d_plus(T-t,S_t,K))-K*sps.norm.cdf(d_minus(T-t,S_t,K)))
    
def put(t,S_t):
    return(K*sps.norm.cdf(d_plus(T-t,K,S_t))-S_t*sps.norm.cdf(d_minus(T-t,K,S_t)))

Call = call(t,S_t)
Put = put(t,S_t)
value = Call.dot(alpha)+Put.dot(beta) #the portfolio value at the 10-th day
initial_value = alpha.dot(call(0,S_0))+beta.dot(put(0,S_0)) #the initial value of the portfolio
loss=initial_value-value #the loss on the 10-th day
#the code bellow consists in computing the empirical quantile at the leval alpha
loss_sorted = np.sort(loss) 
var = loss_sorted[:,int(np.ceil(N*quantile))-1]
var_estimated = np.mean(var)
std=np.std(var)

print(alpha)
print(beta)
print("initial value {}".format(initial_value))
print("estimated VaR {} ".format(var_estimated))
t_gaussian = sps.norm.ppf(0.975)
#print the confidance interval of the VaR at 95% 
print("intervalle de confiance de niveau 95% de VaR: [{},{}]".format(var_estimated-t_gaussian*std,var_estimated+t_gaussian*std))
print("erreur relative sur VaR {}%".format(2*t_gaussian*std/var_estimated*100)) 

#compute the confidance interval of the conditional VaR at 95%
condition_var=[]
for i in range(M):
    l=loss[i,:]
    l_prime=l[l>var_estimated]
    if (len(l_prime)!=0):
        condition_var.append(np.mean(l_prime))
condition_var=np.array(condition_var)
M_prime=len(condition_var)
estimated_condition_var=np.mean(condition_var)
cond_var_std=np.std(condition_var)
print("Conditional VaR at {}% = {}".format(quantile*100,estimated_condition_var))
print("intervalle de confiance de niveau 95% de Conditional VaR: [{},{}]".format(estimated_condition_var-t_gaussian*cond_var_std/np.sqrt(M_prime),estimated_condition_var+t_gaussian*cond_var_std/np.sqrt(M_prime)))
print("erreur relative sur Conditional Var {}%".format(2*t_gaussian*cond_var_std/(np.sqrt(M_prime)*estimated_condition_var)*100))
end=time.time()
print("le temps d'execution est de {} s".format(end-begin))
#draw the histogram of L|L>VaR
plt.hist(loss[loss>var_estimated],normed=True,bins=3*int(len(loss[loss>var_estimated])**(1/3)))
plt.xlabel("loss")
plt.ylabel("density")
plt.title("the distribution of L|L>VaR")
plt.show()


