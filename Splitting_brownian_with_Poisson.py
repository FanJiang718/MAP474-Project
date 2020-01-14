#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:44:34 2018

@author: shen chali, jiang fan
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.stats as sps
import time

#the overall structure of the code remains the same, but several modifications have been made in order to 
#adapt the code to the case with market chocs. 

I0 = 10
t=10/360
T=1 #the number of days of a year is approximated by 360
sigma_volatility = 0.2
Alpha=np.array([10]*5+[10]*5)
Beta=np.array([5]*5+[5]*5)
S_0 = np.ones(I0)*10
K=15
alpha=1e-4
rho_correlation = 0.4
#Basic functions
def d_minus(t,x,y):
    return 1/(sigma_volatility*np.sqrt(t))*np.log(x/y)-1/2*sigma_volatility*np.sqrt(t)
           
def d_plus(t,x,y):
    return 1/(sigma_volatility*np.sqrt(t))*np.log(x/y)+1/2*sigma_volatility*np.sqrt(t)

def call(t,S_t):
    return(S_t*sps.norm.cdf(d_plus(T-t,S_t,K))-K*sps.norm.cdf(d_minus(T-t,S_t,K)))
    
def put(t,S_t):
    return(K*sps.norm.cdf(d_plus(T-t,K,S_t))-S_t*sps.norm.cdf(d_minus(T-t,K,S_t)))

M_cov=np.ones((I0,I0)) 
#fill the covariance matrix of I0 Brownian movements
for i in range(I0):
    for j in range(I0):
        if(i==j):
            M_cov[i][j]=t
        else:
            M_cov[i][j]=rho_correlation*t

initial_value = Alpha.dot(call(0,S_0))+Beta.dot(put(0,S_0))

def loss(X):
    S_t = S_0*np.exp(-1/2*np.square(sigma_volatility)*t+sigma_volatility*X)
    Call_value = call(t,S_t)
    Put_value = put(t,S_t)
    value = Call_value.dot(Alpha) + Put_value.dot(Beta)
    return (initial_value - value)

lambd=5 #the parameter of the homogeneous Poisson process
#this function generate a homogeneous Poisson process of parameter lambda at time t
def Poisson(lambd,t):
    jump=[]
    value=[]
    s=0
    counter=0
    while(s<t):
       s += npr.exponential(scale=1/lambd)
       if (s<=t):
          counter+=1
          jump.append(s)
          value.append(counter)
    return(np.array(jump),np.array(value))     

#reversible transformtion of the Poisson process
p_poisson = 0.4 #the probability of keeping a jump in the Poisson process
#this function returns a reversible transformation of the homogeneous Poisson process
def rev_transf_Poisson(poi):
    jump=poi[0]
    value=[]
    n=len(jump)
    keep=npr.random(size=n)
    keep=(keep<p_poisson)
    poisson_colored=jump[keep]
    poisson_duplic = Poisson((1-p_poisson)*lambd,t)[0]
    jump_new = np.sort(np.append(poisson_colored,poisson_duplic))
    counter=0
    for i in range(len(jump_new)):
        counter+=1
        value.append(counter)
    return (jump_new,np.array(value),keep)

#Assume that the collapses arrive with a gaussian distribution N(mean,1) (Therefore a typical jump will be between mean-3 and mean+3)
mean=2
rho_rev_gauss = 0.7 
mean_transition = (1-rho_rev_gauss)*mean/np.sqrt(1-rho_rev_gauss**2)
#this function returns a reversible transformation for the composed Poisson process of parameter (lambda,N(mean,1))
def rev_transf_composed_Poisson(X):
    (poi,Y)=X
    (jump,value)=poi
    (jump_new,value_new,keep)=rev_transf_Poisson((jump,value))
    N_t = len(jump)
    N_t_prime = len(jump_new)
    x_prime=0
    Y_prime=[]
    for i in range(N_t_prime):
        if(i < (N_t-1)):
            if (keep[i]==True):
                #a reversible transformation of the distribution of collapses 
                y_i_prime = rho_rev_gauss*Y[i] + np.sqrt(1-rho_rev_gauss**2)*npr.normal(loc=mean_transition,scale=1) 
                x_prime += y_i_prime
                Y_prime.append(y_i_prime)
            else:
                y_i_prime = npr.normal(loc=mean,scale=1)
                x_prime += y_i_prime
                Y_prime.append(y_i_prime)
        else:
             y_i_prime = npr.normal(loc=mean,scale=1)
             x_prime += y_i_prime
             Y_prime.append(y_i_prime)
    return(x_prime,jump_new,value_new,np.array(Y_prime))        
            
    
N = int(1e4) #the length of the markov chain used in MCMC
#MCMC has been adapted to this specfic case
def MCMC(N,lower_threshold,upper_threshold,rho_Gaussian_Process):
    markov_chain = np.array([])
    #the usual fluctuation in markets
    B_0=npr.normal(loc=0,scale=np.sqrt(t))
    B = npr.normal(loc=0,scale=np.sqrt(t),size=I0)
    W_t = np.sqrt(rho_correlation)*B_0 + np.sqrt(1-rho_correlation)*B  
    #the unexpected collapse
    p_poi = Poisson(lambd,t)
    collapses = npr.normal(loc=mean,scale=1,size=len(p_poi[0])) 
    total_collapse = np.sum(collapses)    
    X = W_t+total_collapse
    while (loss(X)<=lower_threshold):
        B_0=npr.normal(loc=0,scale=np.sqrt(t))
        B = npr.normal(loc=0,scale=np.sqrt(t),size=I0)
        W_t = np.sqrt(rho_correlation)*B_0 + np.sqrt(1-rho_correlation)*B  
        p_poi = Poisson(lambd,t)
        collapses = npr.normal(loc=mean,scale=1,size=len(p_poi[0])) 
        total_collapse = np.sum(collapses)    
        X = W_t+total_collapse
    for i in range(N):
            Y = npr.multivariate_normal(mean=np.zeros(I0),cov=M_cov)
            W_t_intermediate = rho_Gaussian_Process*W_t+np.sqrt(1-rho_Gaussian_Process**2)*Y
            (total_collapse_inter,jump_inter,value_inter,collapses_inter) = rev_transf_composed_Poisson((p_poi,collapses))
            X_intermediate = W_t_intermediate + total_collapse_inter 
            l=loss(X_intermediate)
            #print("loss {}".format(l))
            if (l>lower_threshold):                
                W_t=W_t_intermediate
                p_poi = (jump_inter,value_inter)
                collapses=collapses_inter
                total_collapse = total_collapse_inter
                X = X_intermediate
                markov_chain=np.append(markov_chain,l)
            else:
                markov_chain=np.append(markov_chain,loss(X))
                #print("loss {}".format(loss(X)))
    count=len(markov_chain[markov_chain>upper_threshold])
    return (markov_chain,count/N)

#now we use the chain rule to compute the quantity P(x>x) with x a given threshold, and then we calibrate x according to the relationship between
#P(X>x) and alpha in order to evaluate the VaR at alpha
M=5 #number of events we use in the splitting 
rho_Gaussian_Process=0.7 #paramater in the Gaussian process
calibration_rate=0
calibration_increment=0.5
x=10 #initialize a threshold
old_x=x #stock the old value of x
eps=0.1*alpha #the imprecision that we allow
counter=1


#an iteration performs one single splitting process    
def iteration(N,x):
#    splitting=np.array([-np.inf])
#    splitting=np.append(splitting,np.linspace(0,x,M))
    splitting = x*(1-((np.arange(M,0,-1,dtype=float))**2/M**2))
    splitting[0]=-np.inf
    splitting = np.append(splitting,x)
    proba=1 #the estimation of P(X>x)
    itr=0
    for i in range(len(splitting)-1):
        conditional_proba = MCMC(N,splitting[i],splitting[i+1],rho_Gaussian_Process)[1]
        proba *=conditional_proba
        itr+=1
        print("at the {}-th calculation, we have P(X>{}|X>{}) = {}" .format(itr,splitting[i+1],splitting[i],proba))
    print("-"*50)
    print("We obtain {} as the approximated value of P(X>x), with x={}".format(proba,x))
    return(proba)

start = time.time()
print("This is the {}-th iteration:".format(counter)) 
proba=iteration(N,x)
if ((np.abs(proba-alpha)/alpha>eps) & ((proba-alpha)>0)):
    while((proba-alpha)>0):
         print("With such a threshold, P(X>x) is greater than {}, so we need to increase x to approach the VaR".format(alpha))
         if ((proba-alpha)/alpha<100):
            calibration_rate = calibration_increment
         else:
            calibration_rate += calibration_increment
         old_x=x
         x *= (1+calibration_rate)
         print ("the threshold x is increased from {} to {},with a calibration rate of {}%".format(old_x,x,calibration_rate*100))
         print("*"*80)
         counter+=1
         print("This is the {}-th iteration:".format(counter))
         proba=iteration(N,x)
    print("*"*80)
    print("We will then proceed the dichotomy within [{},{}]".format(old_x,x))
    print("*"*80)
elif ((np.abs(proba-alpha)>eps) & ((proba-alpha)<0)):
     while((proba-alpha)<0):
         print("With such a threshold, P(X>x) is smaller than {}, so we need to decrease x to approach the VaR".format(alpha))
         if ((alpha-proba)/alpha<100):
            calibration_rate = calibration_increment
         else:
            calibration_rate += calibration_increment
         old_x=x
         x *= (1-calibration_rate)
         print ("the threshold x is decreased from {} to {},with a calibration rate of {}%".format(old_x,x,calibration_rate*100))
         print("*"*80)
         counter+=1
         print("This is the {}-th iteration:".format(counter))
         proba=iteration(N,x)
         old_x,x=x,old_x
     print("*"*80)
     print("We will then proceed dichotomy within [{},{}]".format(old_x,x))
     print("*"*80)
else: #We are so lucky that we directly have the good x at the first guess
    print("the Var at {}% is {}".format((1-alpha)*100,x))
    print("*"*80)


#Proceed the dichotomy now
length=1e-2 #the convergence criterion we impose on the dichotomy 
while(np.abs(x-old_x)>length):
    counter += 1
    print("This is the {}-th iteration:".format(counter)) 
    median = (old_x+x)/2
    proba=iteration(N,median)
    if (proba>alpha):
        old_x=median
        print("-"*30)
        print("continue the process with [{},{}]".format(median,x))
        print("-"*30)
    else:
        x=median
        print("-"*30)
        print("continue the process with [{},{}]".format(old_x,median))
        print("-"*30)
    print("*"*80)

VaR=(old_x+x)/2
end=time.time()
print("the VaR at {}% is {}".format((1-alpha)*100,VaR))
print("The program took {} seconds to compute the VaR".format(end-start))

#evaluate the conditional VaR at level alpha
#In order to improve the precision of the estimation, we will increase the size of the
#sample set
N=int(5e4)
final_chain=MCMC(N,VaR,-np.inf,rho_Gaussian_Process)[0]
conditional_var=np.mean(final_chain)
print("the conditional Var at {}% is {}".format((1-alpha)*100,conditional_var))
print("*"*80)
 
plt.hist(final_chain,bins=int(3*N**(1/3)),normed=True)
plt.xlabel("estimated loss on the {}-th day".format(t))
plt.ylabel("the density")
plt.title("The conditional distribution of L|L>Var")
plt.show()

#We now exhibit the brownian movements and chocs leading to an extreme loss
interval=200 #We divide the interval [0,t] into "interval" equal-size piece    
dt=t/interval
def motion_generator():
    B_0_motion=[0]
    for j in range(interval):
       B_0_motion.append(B_0_motion[j]+npr.normal(loc=0,scale=np.sqrt(dt)))
    B_0=B_0_motion[interval]
    B = npr.normal(loc=0,scale=np.sqrt(t),size=I0)
    (jump,value) = Poisson(lambd,t)
    chocs = npr.normal(loc=mean,scale=1,size=len(jump))
    total_choc=np.sum(chocs)
    poisson_composed=np.cumsum(chocs)
    jump=np.append(0,jump)
    poisson_composed=np.append(0,poisson_composed)
    W_t = np.sqrt(rho_correlation)*B_0 + np.sqrt(1-rho_correlation)*B  
    return(B_0_motion,W_t,jump,poisson_composed,total_choc)

time=np.linspace(0,t,interval+1)
brownian_motion=[]
com_poisson=[]
for i in range(int (5e4)):
    (B_0_motion,W_t,jump,poisson_composed,total_choc)=motion_generator()
    X=W_t+total_choc
    if (loss(X)>VaR):
        print ("Extreme loss {}".format(loss(X)))
        brownian_motion.append(B_0_motion)
        com_poisson.append([jump,poisson_composed])

for i in range(len(brownian_motion)):
    plt.plot(time,brownian_motion[i],label="Mouvement {}".format(i))
    plt.legend(loc="best")
    
plt.figure()

for i in range(len(com_poisson)):
    plt.step(com_poisson[i][0],com_poisson[i][1],where="post",label="choc {}".format(i))
    plt.legend(loc="best")

#Now we have the VaR, we will simulate the distribution of estimators of P(X>Var)
#this process is very long (it takes roughly 30 minutes to finish), so please do not uncomment the following
#code if you only want to get the estimations of the VaR and the conditonal VaR
#estimators=[]
#estNum = 1000
#M=3
#for i in range(estNum):
#    print("round {}".format(i+1))
#    p=iteration(1000,VaR)
#    estimators.append(p)
#    print("estimator {}".format(p))
#    
#plt.hist(estimators,bins=3*int(estNum**(1/3)),normed=True)
#plt.show()
#plt.boxplot(estimators)