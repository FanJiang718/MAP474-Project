#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created on thu may 24 00:03:03 2018

@author: shen chali, jiang fan
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import scipy.stats as sps
import time 

I0 = 10
t=10/360
T=1 #the number of days of a year is approximated by 360
sigma_volatility = 0.2
Alpha=np.array([10]*5+[10]*5)
Beta=np.array([5]*5+[5]*5)
S_0 = np.ones(I0)*10
K=15
alpha=1e-4

def d_minus(t,x,y):
    return 1/(sigma_volatility*np.sqrt(t))*np.log(x/y)-1/2*sigma_volatility*np.sqrt(t)
           
def d_plus(t,x,y):
    return 1/(sigma_volatility*np.sqrt(t))*np.log(x/y)+1/2*sigma_volatility*np.sqrt(t)

def call(t,S_t):
    return(S_t*sps.norm.cdf(d_plus(T-t,S_t,K))-K*sps.norm.cdf(d_minus(T-t,S_t,K)))
    
def put(t,S_t):
    return(K*sps.norm.cdf(d_plus(T-t,K,S_t))-S_t*sps.norm.cdf(d_minus(T-t,K,S_t)))

initial_value = Alpha.dot(call(0,S_0))+Beta.dot(put(0,S_0))
#mcmc in order to estimate the conditional probability
N = int(1e4) #the length of the Markov chain that we will use in the MCMC

def loss(X):
    S_t = S_0*np.exp(-1/2*np.square(sigma_volatility)*t+sigma_volatility*X)
    Call_value = call(t,S_t)
    Put_value = put(t,S_t)
    value = Call_value.dot(Alpha) + Put_value.dot(Beta)
    return (initial_value - value)
#this function generates a Markov chain of the length N, and returns the estimation of the probabilty 
#P(loss(x)>upper|loss(X)>lower)
def MCMC(N,lower_threshold,upper_threshold,rho):
    markov_chain = np.array([])
    X = npr.normal(loc=0,scale=np.sqrt(t),size=I0) #multivariate gaussian vector with mean=0 and covariance matrix=t*id
    #initialize a good x0
    while (loss(X)<=lower_threshold):
        X = npr.normal(loc=0,scale=np.sqrt(t),size=I0)
    for i in range(N):
            Y = npr.normal(loc=0,scale=np.sqrt(t),size=I0)
            X_intermediate=rho*X+np.sqrt(1-rho**2)*Y
            l=loss(X_intermediate)
            if (l>lower_threshold):
                X=X_intermediate
                markov_chain=np.append(markov_chain,l)
            else:
                markov_chain=np.append(markov_chain,loss(X))
    count=len(markov_chain[markov_chain>upper_threshold])
    return (markov_chain,count/N)

#now we use the chain rule to compute the quantity P(loss(X)>x) with x a given threshold, and then we 
#calibrate x according to the relationship between P(loss(X)>x) and alpha in order to evaluate the VaR at level alpha
M=5 #number of sub-events we use in the splitting 
rho=0.7 #paramater of the Gaussian process
calibration_rate=0
calibration_increment=0.3 #calibrate the threshold x
x=8 #initialize the threshold
old_x=x #stock the old value of x
eps=0.1*alpha #the imprecision that we allow
counter=1 #this variable counts the number of iterations permormed during the execution

#the function estimated the quantity P(loss(X)>x) for a given x basing on the splitting principle 
def iteration(N,x):
    #splitting=np.append(splitting,np.linspace(0,x,M))
    splitting = x*(1-((np.arange(M,0,-1,dtype=float))**2/M**2))
    splitting[0]=-np.inf
    splitting = np.append(splitting,x)
    proba=1 #the estimation of P(X>x)
    itr=0
    for i in range(len(splitting)-1):
        conditional_proba=MCMC(N,splitting[i],splitting[i+1],rho)[1]
        proba *= conditional_proba
        itr+=1
        print("at the {}-th calculation, we have P(X>{}|X>{}) = {}" .format(itr,splitting[i+1],splitting[i],proba))
    print("-"*50)
    print("We obtain {} as the approximated value of P(X>x), with x={}".format(proba,x))
    return(proba)

start=time.time()
#find a good interval containing the VaR, once we find it we will be able to start the dichotomy
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
         print ("the threshold x is decreased from{} to {},with a calibration rate of {}%".format(old_x,x,calibration_rate*100))
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


#an appropriate interval has been found, proceed the dichotomy now
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

VaR=(old_x+x)/2 #the estimated VaR at leval alpha
end =time.time()
print(Alpha)
print(Beta)
print("the VaR at {}% is {}".format((1-alpha)*100,VaR))
print("The program took {} seconds to compute the VaR".format(end-start))
#evaluate the conditional VaR at level alpha
#In order to improve the precision of the estimation, we will increase the length of the markov chain 
N=int(5e4)
final_chain=MCMC(N,VaR,-np.inf,rho)[0]
conditional_var=np.mean(final_chain)
print("the conditional Var at {}% is {}".format((1-alpha)*100,conditional_var))
print("*"*80)
plt.hist(final_chain,bins=int(3*N**(1/3)),normed=True)
plt.xlabel("estimated loss on the {}-th day".format(t))
plt.ylabel("the density")
plt.title("The conditional distribution of L|L>Var")
plt.show()
                
#Now we have the VaR, we will simulate the distribution of the estimators of P(X>Var)
#this process is very long (it takes roughly 30 minutes to finish), so please do not uncomment the following
#code if you only want to get the estimations of the VaR and the conditonal VaR
#estimators=[]
#estNum = 1000
#M=3
#for i in range(estNum):
#    print("round {}".format(i+1))
#    p=iteration(500,VaR)
#    estimators.append(p)
#    print("estimator {}".format(p))
#
#print("the standard deviation of estimators {}".format(np.std(estimators)))
#print("relative error on P(X>Var) {}".format(2*1.96*np.std(estimators)/(alpha*np.sqrt(estNum))))
#plt.hist(estimators,bins=3*int(estNum**(1/3)),normed=True)
#plt.show()
#plt.boxplot(estimators)
