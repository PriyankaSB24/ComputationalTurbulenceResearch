import numpy as np
import multiprocessing
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Optimized_EntropyModelingLorenzSystem import pmf_single_var, rossler_dynamics_x, rossler_dynamics_y, rossler_dynamics_z, numConditions, entropy
from Optimized_EntropyModelingLorenzSystem import rossler_pmf_x, rossler_entropy_x,rossler_pmf_y,rossler_entropy_y, rossler_pmf_z,rossler_entropy_z, timeSteps, timeFinal, dt

# ============================================================================ 
# Entropy Analysis
# get pmf of each column which is at the same time step you sum you the initial condition x, y, and z separately
rossler_pmf_x = []
rossler_entropy_x = []
rossler_pmf_y = []
rossler_entropy_y = []
rossler_pmf_z = []
rossler_entropy_z = []
numtimeSteps = timeSteps
timeSteps = np.linspace(0, timeFinal, int(timeFinal/(dt)))

for i in range(numtimeSteps):  # iter # iterate through each columns which is number of time steps columns
    rossler_pmf_x.append(pmf_single_var(np.array(rossler_dynamics_x)[:,i],numConditions//40,-50,50)) # 100 bins ??
    rossler_entropy_x.append(entropy(np.array(rossler_pmf_x)))

    rossler_pmf_y.append(pmf_single_var(np.array(rossler_dynamics_y)[:,i],numConditions//40,-50,50)) # 100 bins ??
    rossler_entropy_y.append(entropy(np.array(rossler_pmf_y)))
    
    rossler_pmf_z.append(pmf_single_var(np.array(rossler_dynamics_z)[:,i],numConditions//40,-50,50)) # 100 bins ??
    rossler_entropy_z.append(entropy(np.array(rossler_pmf_z)))

print("PMFs and entropies for rossler full system calculated")

textToDisplay = f"Number of initial conditions: {numConditions}   Time: {timeFinal}   dt: {dt}    Every {nth_value_entropy}th entropy value skipped"

figure = plt.figure(figsize=(8, 5))
plt.plot(timeSteps, rossler_entropy_z, color='blue', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Entropy (Z)')
plt.title('Rössler System Entropy (Z) vs Time\n' + textToDisplay)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(timeSteps, rossler_entropy_y, color='blue', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Entropy (Y)')
plt.title('Rössler System Entropy (Y) vs Time\n' + textToDisplay)
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(timeSteps, rossler_entropy_x, color='blue', linewidth=1.5)
plt.xlabel('Time')
plt.ylabel('Entropy (X)')
plt.title('Rössler System Entropy (X) vs Time\n' + textToDisplay)
plt.grid(True)
plt.tight_layout()
plt.show()



def compute_pmf_entropy_at_timestep(i):
    """
    Compute PMFs and entropies for x, y, z at a given time step i.
    """
    # Extract time slice
    x_col = np.array(rossler_dynamics_x)[:, i]
    y_col = np.array(rossler_dynamics_y)[:, i]
    z_col = np.array(rossler_dynamics_z)[:, i]

    # Compute PMFs
    pmf_x = pmf_single_var(x_col, numConditions // 40, -50, 50)
    pmf_y = pmf_single_var(y_col, numConditions // 40, -50, 50)
    pmf_z = pmf_single_var(z_col, numConditions // 40, -50, 50)

    # Compute entropies (note: pass lists of PMFs if entropy() expects sequence)
    entropy_x = entropy([pmf_x])
    entropy_y = entropy([pmf_y])
    entropy_z = entropy([pmf_z])

    return (pmf_x, entropy_x, pmf_y, entropy_y, pmf_z, entropy_z)

def main_parallel_entropy(timeSteps):
    # List of indices
    time_indices = list(range(timeSteps))

    with multiprocessing.Pool() as pool:
        results = pool.map(compute_pmf_entropy_at_timestep, time_indices)

    # Unpack results
    for (pmf_x, entropy_x, pmf_y, entropy_y, pmf_z, entropy_z) in results:
        rossler_pmf_x.append(pmf_x)
        rossler_entropy_x.append(entropy_x)
        rossler_pmf_y.append(pmf_y)
        rossler_entropy_y.append(entropy_y)
        rossler_pmf_z.append(pmf_z)
        rossler_entropy_z.append(entropy_z)

    print("All PMFs and entropies for Rossler system computed in parallel!")

if __name__ == "__main__":
    print(f"Using {multiprocessing.cpu_count()} CPU cores for PMF + entropy computation")
    main_parallel_entropy(timeSteps)







print(f"Using {multiprocessing.cpu_count()} CPU cores for PMF + entropy computation")

# PMF
with multiprocessing.Pool() as pool:
    computed_pmf_x = pool.map(pmf_single_var(np.array(rossler_dynamics_x)[:,i],numConditions//40,-50,50), list(range(timeSteps)))
    rossler_pmf_x.append(computed_pmf_x)
    computed_pmf_y = pool.map(pmf_single_var(np.array(rossler_dynamics_x)[:,i],numConditions//40,-50,50), list(range(timeSteps)))
    rossler_pmf_y.append(computed_pmf_x)
    computed_pmf_z = pool.map(pmf_single_var(np.array(rossler_dynamics_x)[:,i],numConditions//40,-50,50), list(range(timeSteps)))
    rossler_pmf_z.append(computed_pmf_x)

# Entropy
with multiprocessing.Pool() as pool:
    computed_entropy_x = pool.map(entropy(np.array(rossler_pmf_x)), list(range(timeSteps)))
    rossler_entropy_x.append(computed_entropy_x)
    computed_entropy_y = pool.map(entropy(np.array(rossler_pmf_y)), list(range(timeSteps)))
    rossler_entropy_y.append(computed_entropy_y)
    computed_entropy_z = pool.map(entropy(np.array(rossler_pmf_z)), list(range(timeSteps)))
    rossler_entropy_z.append(computed_entropy_z)