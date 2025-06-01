import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random
import multiprocessing
from entropy_and_mutual_information_estimators import entropy,pmf_single_var
import EntropyParralelProcessing

numConditions = int(1000)     # Number of initial conditions
timeFinal = 20                # Final Time
dt = 0.01                    # Length of each time step t{n+1} - t{n}
timeSteps = int(timeFinal/dt) 
nth_value_entropy = int(2)   # Get every nth value of entropy

def gaussian_initial_condition_generator_old(numConditions): 
    """
    Generates matrix of Gaussian-distributed initial conditions with numTimeSteps columns and numConditions rows
    Creates a `numConditions`-dimensional dataset, with each condition 
    represented as a row and each time step as a column. Values are sampled from 
    a multivariate normal distribution with zero mean and identity covariance 
    """
    # Each row corresponds to a single initial condition and shows its evolution over time.
    # Each column represents the system state at a specific time step across all initial conditions.
    # [x_0      y_0     z_0 ]
    # [x_1      y_1     z_1 ]
    #   .        .       .
    #   .        .       .
    # [x_{numConditions-1} y_{numConditions-1} z_{numConditions-1} ]
    mean = np.zeros(numConditions)
    cov = np.eye(numConditions)  # Identity matrix = no correlation
    initial_conditions = np.random.multivariate_normal(mean, cov, 3) 
    return initial_conditions.T

def gaussian_initial_condition_generator(numConditions): 
    """
    Generates `numConditions` initial conditions, each in 3D (x, y, z),
    sampled independently from a standard normal distribution.
    Returns a (numConditions, 3) matrix.
    """
    return np.random.normal(loc=0.0, scale=1.0, size=(numConditions, 3))

print("Starting Execution....")
# numConditions rows and 3 columns (x, y, z)
gaussian_initial_conditions = np.array(gaussian_initial_condition_generator(numConditions))
# gaussian_initial_conditions_old = np.array(gaussian_initial_condition_generator(numConditions))
print("Gaussian initial conditions generated")

# PMF: numBins columns and 1 row
# PMF taken for all (x,y,z) separately across all initial conditions 
pmf_x = np.array(pmf_single_var(gaussian_initial_conditions[:,0],numConditions//40,-5,5)) 
entropy_x = [] # Singular entropy value of x initial conditions
entropy_x.append(entropy(np.array(pmf_x)))

pmf_y = []
pmf_y.append(pmf_single_var(gaussian_initial_conditions[:,1],numConditions//40,-5,5)) 
entropy_y = []
entropy_y.append(entropy(np.array(pmf_y)))

pmf_z = [] 
pmf_z.append(pmf_single_var(gaussian_initial_conditions[:,2],numConditions//40,-5,5)) 
entropy_z = []
entropy_z.append(entropy(np.array(pmf_z)))

print("PMFs and entropies for gaussian initial conditions calculated")
# take in an input with 
    # [x_0      y_0     z_0 ]
    # [x_1      y_1     z_1 ]
    #   .        .       .
    #   .        .       .
# generate [x_0(t = 0)  x_0(t = 1)  x_0(t = 2)  x_0(t = 3) .. 
#           y_0(t = 0)  y_0(t = 1)
# ]
# and run numConditions times 

# initial conditions = [x_0      y_0     z_0 ] 
def rossler_system_generator(initial_conditions, dt):
    # Rossler dynamics variables
    a=0.2
    b=0.2
    c=5.7
    
    trajectory_x = []
    trajectory_y = []
    trajectory_z = []
    
    x = initial_conditions[0]
    y = initial_conditions[1]
    z = initial_conditions[2]
            
    dxdt = - y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
        
    for t in range(timeSteps): # does for each time step
        x = x + (t * dt * dxdt)
        y = y + (t * dt * dydt)
        z = z + (t * dt * dzdt)
        trajectory_x.append(x) 
        trajectory_y.append(y)
        trajectory_z.append(z) # add to columns
    
    # returns three arrays each are 1 row by timeSteps columns = 1 by timeFinal/dt
    return np.array(trajectory_x).T, np.array(trajectory_y).T, np.array(trajectory_z).T # columns (x, y, z) and timeFinal/dt rows 

print("Rossler dynamics generated")

rossler_dynamics_x = []
rossler_dynamics_y = []
rossler_dynamics_z = []

for i in range(numConditions):
    x_arr, y_arr, z_arr = np.array(rossler_system_generator(gaussian_initial_conditions[i], dt)) # add to new row each time
    rossler_dynamics_x.append(x_arr)
    rossler_dynamics_y.append(y_arr)
    rossler_dynamics_z.append(z_arr) # num time steps columns and num initial conditions rows

if __name__ == "__main__":
    EntropyParralelProcessing.main()


"""
print("Entropy analysis started")
# ============================================================================ 
# Entropy Analysis
# get pmf of each column which is at the same time step you sum you the initial condition x, y, and z separately
rossler_pmf_x = []
rossler_entropy_x = []
rossler_pmf_y = []
rossler_entropy_y = []
rossler_pmf_z = []
rossler_entropy_z = []

for i in range(timeSteps):  # iter # iterate through each columns which is number of time steps columns
    rossler_pmf_x.append(pmf_single_var(np.array(rossler_dynamics_x)[:,i],numConditions//40,-50,50)) # 100 bins ??
    rossler_entropy_x.append(entropy(np.array(rossler_pmf_x)))

    rossler_pmf_y.append(pmf_single_var(np.array(rossler_dynamics_y)[:,i],numConditions//40,-50,50)) # 100 bins ??
    rossler_entropy_y.append(entropy(np.array(rossler_pmf_y)))
    
    rossler_pmf_z.append(pmf_single_var(np.array(rossler_dynamics_z)[:,i],numConditions//40,-50,50)) # 100 bins ??
    rossler_entropy_z.append(entropy(np.array(rossler_pmf_z)))

print("PMFs and entropies for rossler full system calculated")

timeSteps = np.linspace(0, timeFinal, int(timeFinal/(dt)))
textToDisplay = f"Number of initial conditions: {numConditions}   Time: {timeFinal}   dt: {dt}"

figure = plt.figure(figsize=(8, 5))
plt.plot(timeSteps, rossler_entropy_z, color='blue', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Entropy (Z)')
plt.title('Rössler System Entropy (Z) vs Time\n' + textToDisplay)
plt.grid(True)
plt.tight_layout()
plt.show()

timeSteps = np.linspace(0, timeFinal, int(timeFinal/dt))
plt.figure(figsize=(8, 5))
plt.plot(timeSteps, rossler_entropy_y, color='blue', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Entropy (Y)')
plt.title('Rössler System Entropy (Y) vs Time\n' + textToDisplay)
plt.grid(True)
plt.tight_layout()
plt.show()

timeSteps = np.linspace(0, timeFinal, int(timeFinal/dt ))
plt.figure(figsize=(8, 5))
plt.plot(timeSteps, rossler_entropy_x, color='blue', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Entropy (X)')
plt.title('Rössler System Entropy (X) vs Time\n' + textToDisplay)
plt.grid(True)
plt.tight_layout()
plt.show()
"""