#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:10:54 2018

@author: Fan JIANG, Chali SHEN
"""
# Delta-Gamma distribution with importance sampling for indenpendant case
import numpy as np
import scipy.stats as sts
import scipy.optimize as opt
import matplotlib.pyplot as plt

# functions
def d_plus(sigma,t,x,y):
    return 1./(sigma*np.sqrt(t))*np.log(x/y)+0.5*sigma*np.sqrt(t)

def d_minus(sigma,t,x,y):
    return 1./(sigma*np.sqrt(t))*np.log(x/y)-0.5*sigma*np.sqrt(t)

# dV/dt    
def f_a0(alpha, beta, S, K, T, t, sigma):
    return -np.sum((alpha + beta)*S*sts.norm.pdf(d_plus(sigma,T-t,S,K))*(0.5/sigma*(T-t)**(-1.5)
            *np.log(S/K)-sigma/(4*np.sqrt(T-t)))-(alpha+beta)*K*sts.norm.pdf(d_minus(sigma,T-t,S,K))*(
            0.5/sigma*(T-t)**(-1.5)*np.log(S/K)+sigma/(4*np.sqrt(T-t))))
# dV/dS
def f_a(alpha,beta,S,K,T,t,sigma):
    return -((alpha+beta)*sts.norm.cdf(d_plus(sigma,T-t,S,K)) - beta + (alpha+beta)*sts.norm.pdf(
            d_plus(sigma,T-t,S,K))/(sigma*np.sqrt(T-t))-(alpha+beta)/(sigma*np.sqrt(T-t))*K/S*sts.norm.pdf(
            d_plus(sigma,T-t,K,S)))
#d2V/dS_i dS_j
def f_COV(alpha,beta,S,K,T,t,sigma):
    # return the matrix of covariance
    var = (alpha+beta)*sts.norm.pdf(d_plus(sigma,T-t,S,K))/(sigma*np.sqrt(T-t)*S)*(1.-
          d_plus(sigma,T-t,S,K)/(sigma*np.sqrt(T-t)))+(alpha+beta)*sts.norm.pdf(d_plus(
          sigma,T-t,K,S))*K/(sigma*np.sqrt(T-t)*S*S)*(1.-d_plus(sigma,T-t,K,S)/(sigma*np.sqrt(T-t)))
    return np.diag(var)

def f_A(alpha,beta,S,K,T,t,sigma):
    return -0.5*f_COV(alpha,beta,S,K,T,t,sigma)

# Quand S sont independant, change of probability
def f_b(alpha,beta,S,K,T,t,sigma,h):
    return f_a(alpha,beta,S,K,T,t,sigma)*sigma*np.sqrt(h)*S

def f_lambd(alpha,beta,S,K,T,t,sigma,h):
    return np.diag(f_A(alpha,beta,S,K,T,t,sigma))*S*S*sigma*sigma*h

# generating function
def f_Psi(theta,lambd,b):
    return 0.5*np.sum(theta*theta*b*b/(1.-2.*theta*lambd)-np.log(1.-2.*theta*lambd))


#Q = a0 + a'*Del_S + Del_S'*A*Del_S

sigma = 0.2
h = 10./360. # 10 days, assuming 360 days per year
T = 1.
t = 0.
I = 10
alpha = np.array([10]*5+[10]*5)
beta = np.array([5]*5+[5]*5)
S = np.ones(I)*1
K = np.ones(I)*2


num_iter = 500
num_event = 2*10**4

# for dichotomy
a0 = f_a0(alpha, beta, S, K, T, t, sigma)*h
x = a0 # ajuster x et step tel que P(L > lo) > target et P(L < hi) < target
step = 1.
lo = x - step
hi = -1

level = 0.9999  # level of Var
target = 1. - level
stop_threshold = 0.02
count = 1

#dichotomy
while True:
    print("*"*50)
    print("interation "+str(count)+": ")
    b = f_b(alpha,beta,S,K,T,t,sigma,h)
    #print("b: "+str(b))
    lambd = f_lambd(alpha,beta,S,K,T,t,sigma,h)
    #print("lambda: "+ str(lambd))
    a0 = f_a0(alpha, beta, S, K, T, t, sigma)*h
    print("a0: "+str(a0))
    def f_Derive_Psi(theta):
        return np.sum(theta*b*b*(1.-theta*lambd)/(1.-2.*theta*lambd)**2 + lambd/(1.-2.*theta*lambd)) - (x-a0)
    theta =opt.fsolve(f_Derive_Psi,1) # optimal theta with current x
    print("theta:" + str(theta))
    print("x = "+str(np.sum(theta*b*b*(1.-theta*lambd)/(1.-2.*theta*lambd)**2 + lambd/(1.-2.*theta*lambd))+a0))
    psi_theta = f_Psi(theta,lambd,b) 
    # change of probability
    new_sigma2 = 1./(1.-2.*theta*lambd)
    new_mu = theta*b* new_sigma2
    print("")
    print("generating Z")
    Z = np.random.multivariate_normal(new_mu,np.diag(new_sigma2),size = (num_event, num_iter))
    #print(Z.shape)
    Z = Z.reshape((Z.shape[-1],num_event,num_iter))
    #print(Z.shape)
    b = b.reshape((I,1,1))
    lambd = lambd.reshape((I,1,1))
    # Q = L-a0
    Q = np.sum(b*Z+ lambd*Z*Z,axis = 0)
    indicator_Q = (Q >(x-a0)) * np.exp(psi_theta-theta*Q)
    probs = np.mean(indicator_Q,axis = 0)
    mean = np.mean(probs)
    std = np.std(probs)
    print("P(L>x): " + str(mean))
    print("P(L <= x) = "+ str((1- mean)*100.)+"%")
    print("l'intervalle de confiance de 95% avec "+str(num_iter)+" interations: ["+ 
          str(mean - 1.96*std/np.sqrt(num_iter))+", "+str(mean + 1.96*std/np.sqrt(num_iter))+"]") 
    print("l'erreur relative: "+str(100*2*1.96*std/np.sqrt(num_iter)/mean)+"%")

    #Conditional_Var = np.mean(indicator_Q * (Q+a0),axis = 0)/probs
    Conditional_Var = []
    for i in range(len(probs)):
        if probs[i] != 0:
            Conditional_Var.append(np.mean(indicator_Q[:,i] * (Q[:,i]+a0))/probs[i])
    Conditional_Var = np.array(Conditional_Var)
    
    
    mean_condi = np.mean(Conditional_Var)
    std_condi = np.std(Conditional_Var)
    print("")
    print("Conditional Var E[L|L>x] avec P(L>x)="+str(mean)+": "+ str(mean_condi))
    print("l'intervalle de confiance de 95% avec "+str(num_iter)+" interations: ["+ 
          str(mean_condi - 1.96*std_condi/np.sqrt(num_iter))+", "+str(mean_condi + 1.96*std_condi/np.sqrt(num_iter))+"]") 
    print("l'erreur relative: "+str(100*2*1.96*std_condi/np.sqrt(num_iter)/mean_condi)+"%")
    # dichotomy
    if abs(mean -target) <= (stop_threshold * target):
        break
    
    if mean == 0.:
        step = step/2
        hi = x
        x = x - step
    else:
        if mean > target:
            if hi == -1.:
                lo = x
                x = x + step
                step = step *2
            else:
                lo = x
                x = (lo + hi)/2
        else:
            hi = x
            x = (x + lo)/2.
                
    count +=1


print("")
print("-"*50)
print("x has been optimised (minimiser la variance)")
# VaR of each iteration
Q_sorted = np.sort(Q,axis = 0)
VaR_iter = np.zeros(num_iter)
for i in range(num_iter):
    tmp_Q = Q_sorted[:,i]
    if tmp_Q[-1] > (x-a0):
        VaR_iter[i] = tmp_Q[tmp_Q > (x-a0)][0] + a0
    else:
        VaR_iter[i] = tmp_Q[-1] + a0


VaR_mean =  np.mean(VaR_iter[:])
VaR_std = np.std(VaR_iter[:])
print("VaR pour "+str(level*100)+"%: "+str(VaR_mean))
print("l'intervalle de confiance de 95% avec "+str(num_iter)+" interations de VaR: ["+ 
     str(VaR_mean - 1.96*VaR_std/np.sqrt(num_iter))+", "+str(VaR_mean + 1.96*VaR_std/np.sqrt(num_iter))+"]")
print("l'erreur relative: "+str(100*2*1.96*VaR_std/np.sqrt(num_iter)/VaR_mean)+"%")

#probability associated with VaR
indicator_Q = (Q > (VaR_mean-a0)) * np.exp(psi_theta-theta*Q)
probs = np.mean(indicator_Q,axis = 0)
mean = np.mean(probs)
std = np.std(probs)
print("")
print("P(L> VaR): " + str(mean))
print("P(L <= VaR) = "+ str((1- mean)*100.)+"%")
print("l'intervalle de confiance de 95% avec "+str(num_iter)+" interations: ["+ 
     str(mean - 1.96*std/np.sqrt(num_iter))+", "+str(mean + 1.96*std/np.sqrt(num_iter))+"]") 
print("l'erreur relative: "+str(100*2*1.96*std/np.sqrt(num_iter)/mean)+"%")


Conditional_Var = []
for i in range(len(probs)):
    if probs[i] != 0:
        Conditional_Var.append(np.mean(indicator_Q[:,i] * (Q[:,i]+a0))/probs[i])
Conditional_Var = np.array(Conditional_Var)
    

mean_condi = np.mean(Conditional_Var)
std_condi = np.std(Conditional_Var)
print("")
print("Conditional Var E[L|L>=VaR] avec P(L>VaR)="+str(mean)+": "+ str(mean_condi))
print("l'intervalle de confiance de 95% avec "+str(num_iter)+" interations: ["+ 
     str(mean_condi - 1.96*std_condi/np.sqrt(num_iter))+", "+str(mean_condi + 1.96*std_condi/np.sqrt(num_iter))+"]") 
print("l'erreur relative: "+str(100*2*1.96*std_condi/np.sqrt(num_iter)/mean_condi)+"%")


# Conditional Distribution 
distribu = Q[Q>(VaR_mean-a0)] + a0
P = []
integral = 0.
inter = np.linspace(VaR_mean + 0.01*VaR_mean, min(np.amax(distribu),2*VaR_mean))
for i in range(len(inter)-1):
    P.append(np.mean((Q>(inter[i]-a0))*(Q<(inter[i+1]-a0))*indicator_Q)/mean/(inter[i+1]-inter[i]))
    integral += P[-1]*(inter[i+1]-inter[i])

P.append(0)
print("integral: " + str(integral))

plt.figure(dpi = 150)
plt.plot(inter,P)
plt.title("Distribution conditionelle au dela de la VaR(mesure P)")
plt.xlabel("Loss")
plt.ylabel("density")
