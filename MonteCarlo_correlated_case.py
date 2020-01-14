#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 20:43:40 2018

@author: shen chali, jiang fan
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.stats as sps
import time

#we use the same notations 
#the structure of the code is quite similar to that of the Monte Carlo in the independant case
#the only difference from the independant case is the distribution the random vector W_t
begin = time.time()
N=int(1e4) 
M=int(500)
I0 = 10
t=10/360
T=1 
sigma_volatility = 0.2
alpha=np.array([10]*5+[10]*5)
beta=np.array([5]*5+[5]*5)
S_0 = np.ones(I0)*10
K=10
rho_correlation=0.8 #the coefficient of correlation between different brownian motions 
M_cov=np.ones((I0,I0)) #the covariance matrix
#fill the covariance matrix of I0 Brownian motions
for i in range(I0):
    for j in range(I0):
        if(i==j):
            M_cov[i][j]=t
        else:
            M_cov[i][j]=rho_correlation*t

W_t = npr.multivariate_normal(mean=np.zeros(I0),cov=M_cov,size=(M,N))
#B_0=npr.normal(loc=0,scale=np.sqrt(t),size=(M,N))
#B=npr.normal(loc=0,scale=np.sqrt(t),size=(M,N,I0))
#W_t=np.sqrt(rho_correlation)*B_0.reshape((M,N,1))+np.sqrt(1-rho_correlation)*B
S_t = S_0*np.exp(-1/2*np.square(sigma_volatility)*t+sigma_volatility*W_t)
quantile = 0.9999


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
value = Call.dot(alpha)+Put.dot(beta)
initial_value = alpha.dot(call(0,S_0))+beta.dot(put(0,S_0))
loss=initial_value-value
loss_sorted = np.sort(loss)

var = loss_sorted[:,int(np.ceil(N*quantile))-1]
var_estimated = np.mean(var)
std=np.std(var)

print(alpha)
print(beta)
print("initial value {}".format(initial_value))
print("estimated VaR {}".format(var_estimated))

t_gaussian = sps.norm.ppf(0.975)
print("intervalle de confiance de niveau 95%: [{},{}]".format(var_estimated-t_gaussian*std,var_estimated+t_gaussian*std))
print("erreur relative {} %".format(2*t_gaussian*std/var_estimated*100))
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
plt.hist(loss[loss>var_estimated],normed=True,bins=3*int(len(loss[loss>var_estimated])**(1/3)))
plt.xlabel("loss")
plt.ylabel("density")
plt.title("the distribution of L|L>VaR")
plt.show()

