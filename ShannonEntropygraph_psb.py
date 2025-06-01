import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
from entropy_and_mutual_information_estimators import entropy,pmf_single_var
# Each row corresponds to a single initial condition and shows its evolution over time.
# Each column represents the system state at a specific time step across all initial conditions.
#       t = 0      t = 1      t = 2           t = numTimeSteps
# x_0  x_0(t=0)   x_0(t=1)   x_0(t=2)   ...  x_0(t=n)
# y_1  y_1(t=0)   y_1(t=1)   y_1(t=2)   ...  y_1(t=n)
# z_2  z_2(t=0)   z_2(t=1)   z_2(t=2)   ...  z_2(t=n)
# ...    ...        ...        ...      ...     ...
# x_m  x_m(t=0)   x_m(t=1)   x_m(t=2)   ...  x_m(t=n) 
# x_m = x_numConditions

numConditions = int(5) # Number of initial conditions
numTimeSteps = 100     # Number of time steps (columns of matrix)
lengthTimeSteps = 1    # Length of each time step t{n+1} - t{n}
timeSteps = np.linspace(0,lengthTimeSteps, numTimeSteps) 

def gaussian_initial_condition_generator(numConditions, numTimeSteps): 
    """
    Generates matrix of Gaussian-distributed initial conditions with numTimeSteps columns and numConditions rows
    Creates a `numConditions`-dimensional dataset, with each condition 
    represented as a row and each time step as a column. Values are sampled from 
    a multivariate normal distribution with zero mean and identity covariance 
    """
    mean = np.zeros(numConditions)
    cov = np.eye(numConditions)  # Identity matrix = no correlation
    initial_conditions = np.random.multivariate_normal(mean, cov, numTimeSteps)
    return initial_conditions.T

gaussian_data = gaussian_initial_condition_generator(numConditions, numTimeSteps)
# PMF: numBins columns and numTimeSteps rows
# PMF taken for all initial conditions at a single time step 
pmf = [] 
for i in range (numTimeSteps): # Iterate columns 
    pmf.append(pmf_single_var(gaussian_data[:,i],numConditions // 2,-5,5)) # 'numConditions // 2' chosen as arbitrary bin size

entropy_results = [] # Entropy results taken across the column of PMF
for i in range(numTimeSteps):    
    pmf = np.array(pmf)
    entropy_results.append(entropy(pmf[i:, ]))

# Entropy vs time figure
plt.figure(figsize=(8, 5))
plt.plot(timeSteps, entropy_results, marker='o', linestyle='-', color='blue', label='Entropy')
plt.title("Entropy over Time Steps")
plt.xlabel("Time")
plt.ylabel("Entropy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# ----------------------------- FUNCTIONS NOT USED -----------------------------
def initial_condition_generator(numConditions):
    initial_conditions = []
    for i in range(numConditions):
        initial_conditions.append([random.randint(0, 5), random.randint(0, 5), random.randint(0, 5)])  
    return initial_conditions # numConditions rows and 3 columns

def rossler_system_generator(numTimeSteps, lengthTimeSteps, initial_conditions):
    a=0.2
    b=0.2
    c=5.7 # rossler dynamics
    dt = lengthTimeSteps
    rossler_system = []
    
    for i in range (len(initial_conditions)):
        row = []
        for t in timeSteps:
            x = initial_conditions[i][0]
            y = initial_conditions[i][1]
            z = initial_conditions[i][2]
            
            dxdt = - y - z
            dydt = x + a * y
            dzdt = b + z * (x - c)
            
            x = x + dt * dxdt
            y = y + dt * dydt
            z = z + dt * dzdt
            row.append([x, y, z])
        rossler_system.append(row) 
    return rossler_system

    ''' for rows in range(numConditions):
        mean = random.uniform(-0.2, 0.2)        
        std_dev = random.uniform(0.8, 1.2)
        initial_conditions = np.clip(np.random.multivariate_normal(mean, std_dev, size=numTimeSteps), 0, 1)
    return initial_conditions '''
    
# row: time steps columns: different initial conditions
# data = rossler_system_generator(numTimeSteps, lengthTimeSteps, initial_condition_generator(numConditions))