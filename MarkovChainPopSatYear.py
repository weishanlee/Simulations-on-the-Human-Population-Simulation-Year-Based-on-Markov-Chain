# -*- coding: utf-8 -*-
"""
@author: Wei-Shan Lee
email: weishan_lee@yahoo.com

Description
---------------------------
This code simulates the global human population from Year 2019 to the saturated 
year with the Discrete-time Markov Chain (after imposing the maximum number of 
population), and compares the results calculated with T-Function in Ref[1].
Data from Ref[2] is also shown for comparison reasons.

References:
[1] Rein Taagepera,"A world population growth model: Interaction with Earth's 
    carrying capability and technology in limited space", Technological 
    Forecasting & Social Change 82 (2014) 34-41.
[2] https://www.kaggle.com/theworldbank/global-population-estimates
"""
#print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

#%% Markov Chain
nsteps = 2500    # time step (month)

timeStep = np.arange(0, nsteps, 1)
for i, val in enumerate(timeStep):
    timeStep[i] = 2019 + val/12.0

def MarkovChain(maxN, yearlyBirthRate, yearlyDeathRate, initialPopulation):
    N = maxN  # 1.02e10 #maximum population size
    birthRate = yearlyBirthRate  #19.349/1000. # yearly birth rate
    deathRate = yearlyDeathRate  #7.748/1000. # yearly death rate
    birthRate = birthRate/12.0
    deathRate = deathRate/12.0
    a = 0.4 / N
    b = 0.4 / N
    x = np.zeros(nsteps)
    x = []                 
    x += [initialPopulation] # 7.6e9 # initial population 

    for t in range(nsteps - 1): 
        t = int(t)
        if 0 < x[t] < N - 1:
            # Is there a birth?
            birth = np.random.rand() <= a * x[t]
            # Is there a death?
            death = np.random.rand() <= b * x[t]
            # We update the population size.
            if (birth == True) and (death == True):
                x += [ x[t] + birthRate * N - deathRate * N ]
                #x[t + 1] = x[t] + birthRate * N - deathRate * N
            elif (birth == True) and (death == False):
                x += [ x[t] + birthRate * N ]
                #x[t + 1] = x[t] + birthRate * N
            elif (birth == False) and (death == True):
                x += [ x[t] -  deathRate * N ]  
                #x[t + 1] = x[t] -  deathRate * N
            else:
                x += [x[t]]
                #x[t + 1] = x[t]
        # The evolution stops if we reach $0$ or $N$.
        else:
            x += [x[t]]
            #x[t + 1] = x[t]
    return x

#%% Fitting parameters for the t-function
A = 3.83e9
B = 1.28
D = 1980.0
tau = 22.9 # years
M = 0.70

def EFunc(t):
    value = np.exp( (D-t)/tau )
    return value

def tFunc(t): # t-function
    value = A / ( np.log( (B + EFunc(t)) ) )**M
    return value

# plot all together
plt.figure("Population Size vs. Time (year)")
ax = plt.gca()
ax.set_xlabel("Time (year)",size = 16)
ax.set_ylabel("Population Size",size = 16)
plt.grid()

xvalsRef1 = [400,600,800,1000,1100,1200,1300,1400,1500,1600,1700,1750,1800,
             1850,1900,1920,1940,1950,1960,1970,1980,1990,2000,2010]
yvalsRef1 = [198,214,235,281,310,398,396,362,457,544,635,771,941,1242,1639,
             1905,2313,2526,3035,3667,4442,5278,6021,6861] 
for i, val in enumerate(yvalsRef1):
    yvalsRef1[i] = val * 1e6

xvalsTimeStep = np.arange(-2,2500, 1)
yvalsx1 = tFunc(xvalsTimeStep)

xvalsRef2 = np.arange(1962,2051,10)

yvalsRef2 = [3127961482,3192794384,3258201476,3324951621,3394864530,3464439525,
3534821115,3609383725,3684765870,3762198347,3838924951,3914857611,	
3991430917,4066267984,4139151082,4211781677,4285609387,4361295248,	
4437690434,4515764583,4596813158,4678525765,4759982360,4843067309,	
4928822143,5016798785,5105701987,5194731380,5284886348,5372078249,
5456141249,5541075501,5624840414,5709757338,5792568347,5875398158,
5957237460,6038067278,6118075293,6197638117,6276824418,6356259574,	
6436346998,6517020798,6598421257,6680423047,6763745673,6847214549,	
6930656699,7012843635,7097400665,7182860115,7268986176,7355220412,	
7442135578,7524453000,7606102000,7686852000,7766687000,7845612000,	
7923611000,8000663000,8076739000,8151786000,8225788000,8298763000,	
8370716000,8441688000,8511707000,8580819000,8649017000,8716283000,	
8782656000,8848138000,8912769000,8976588000,9039596000,9101784000,
9163183000,9223737000,9283410000,9342117000,9399793000,9456429000,	
9511938000,9566349000,9619629000,9671774000,9722319000]

yvalsRef2 = [yvalsRef2[i] for i in range(len(yvalsRef2)) if i%10 == 0]

MC = MarkovChain(1.02e10,19.349/1000.,7.748/1000.,7.6e9)

plt.plot(xvalsRef1, yvalsRef1,'ro',
         xvalsRef2, yvalsRef2,'mo',
         xvalsTimeStep, yvalsx1,'k-',
         timeStep, MC ,'b-.',lw=1)

plt.minorticks_on()
minorLocatorX = AutoMinorLocator(5)#number of minor intervals per major inteval
minorLocatorY = AutoMinorLocator(5)
ax.xaxis.set_minor_locator(minorLocatorX) # add minor ticks on x axis
ax.yaxis.set_minor_locator(minorLocatorY) # add minor ticks on y axis
lines=ax.get_lines()
lines[0].set_label("Data from Table 1 in Ref[1]")
lines[1].set_label("Data from Ref[2]")
lines[2].set_label("T-function")
lines[3].set_label("Markov Chain Prediction")
ax.legend(loc='center left', shadow=True, fontsize='large')
plt.xlim(1500, 2200)
plt.show()
plt.savefig("./allTogether.png")
