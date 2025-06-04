import numpy as np
import multiprocessing
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from entropy_and_mutual_information_estimators import entropy,pmf_single_var
from Optimized_EntropyModelingLorenzSystem import pmf_single_var, rossler_dynamics_x, rossler_dynamics_y, rossler_dynamics_z, numConditions, entropy
from Optimized_EntropyModelingLorenzSystem import numConditions, timeFinal, timeSteps, dt, nth_value_entropy
import pdb

def compute_entropy_at_timestep(i):
    pmf_x = pmf_single_var(np.array(rossler_dynamics_x)[:, i], numConditions //40, -50, 50)
    entropy_x = entropy(np.array(pmf_x))
    
    pmf_y = pmf_single_var(np.array(rossler_dynamics_y)[:, i], numConditions //40, -50, 50)
    entropy_y = entropy(np.array(pmf_y))

    pmf_z = pmf_single_var(np.array(rossler_dynamics_z)[:, i], numConditions //40, -50, 50)
    entropy_z = entropy(np.array(pmf_z))

    return pmf_x, entropy_x, pmf_y, entropy_y, pmf_z, entropy_z

# ---------------
# Main block for multiprocessing
# ---------------
def main():
    multiprocessing.freeze_support()
    # 6/3 time_indices = range(0, timeSteps, nth_value_entropy)
    float_time_indices = np.linspace(0, timeFinal, int(timeFinal/dt))
    time_indices = range(int(timeFinal / dt))

    # Run parallel computations
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_entropy_at_timestep, time_indices)

    rossler_pmf_x, rossler_entropy_x = [], []
    rossler_pmf_y, rossler_entropy_y = [], []
    rossler_pmf_z, rossler_entropy_z = [], []

    # Unpack results into arrays
    for pmf_x, ent_x, pmf_y, ent_y, pmf_z, ent_z in results:
        rossler_pmf_x.append(pmf_x)
        rossler_entropy_x.append(ent_x)
        rossler_pmf_y.append(pmf_y)
        rossler_entropy_y.append(ent_y)
        rossler_pmf_z.append(pmf_z)
        rossler_entropy_z.append(ent_z)

    print("PMFs and entropies for Rössler full system calculated with parallel system")

    # Build time array and match to selected time indices
    # 6/3 timeSteps_Arr = np.linspace(0, timeFinal, timeSteps)
    # 6/3 selected_times = timeSteps_Arr[::nth_value_entropy]

    textToDisplay = f"Number of initial conditions: {numConditions}   Time: {timeFinal}   dt: {dt}    Every {nth_value_entropy}th entropy value computed"

    # Plot Z
    plt.figure(figsize=(8, 5))
    plt.plot(float_time_indices, rossler_entropy_z, color='blue', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Entropy (Z)')
    plt.title('Rössler System Entropy (Z) vs Time\n' + textToDisplay)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Y
    plt.figure(figsize=(8, 5))
    plt.plot(float_time_indices, rossler_entropy_y, color='green', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Entropy (Y)')
    plt.title('Rössler System Entropy (Y) vs Time\n' + textToDisplay)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot X
    plt.figure(figsize=(8, 5))
    plt.plot(float_time_indices, rossler_entropy_x, color='red', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Entropy (X)')
    plt.title('Rössler System Entropy (X) vs Time\n' + textToDisplay)
    plt.grid(True)
    plt.tight_layout()
    plt.show()




"""
def main():
    multiprocessing.freeze_support()

    # Run parallel computations
    with multiprocessing.Pool() as pool:
        results = pool.map(compute_entropy_at_timestep, range(0, timeSteps, nth_value_entropy))

    # Unpack results
    rossler_pmf_x, rossler_entropy_x = [], []
    rossler_pmf_y, rossler_entropy_y = [], []
    rossler_pmf_z, rossler_entropy_z = [], []

    for res in results:
        pmf_x, ent_x, pmf_y, ent_y, pmf_z, ent_z = res
        rossler_pmf_x.append(pmf_x)
        rossler_entropy_x.append(ent_x)

        rossler_pmf_y.append(pmf_y)
        rossler_entropy_y.append(ent_y)

        rossler_pmf_z.append(pmf_z)
        rossler_entropy_z.append(ent_z)

    print("PMFs and entropies for rossler full system calculated with parralel system")

    timeSteps_Arr = np.linspace(0, timeFinal, int(timeFinal/(dt)))
    textToDisplay = f"Number of initial conditions: {numConditions}   Time: {timeFinal}   dt: {dt}    Every {nth_value_entropy}th entropy value skipped"

    figure = plt.figure(figsize=(8, 5))
    plt.plot(timeSteps_Arr, rossler_entropy_z, color='blue', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Entropy (Z)')
    plt.title('Rössler System Entropy (Z) vs Time\n' + textToDisplay)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(timeSteps_Arr, rossler_entropy_y, color='blue', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Entropy (Y)')
    plt.title('Rössler System Entropy (Y) vs Time\n' + textToDisplay)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(timeSteps_Arr, rossler_entropy_x, color='blue', linewidth=1.5)
    plt.xlabel('Time')
    plt.ylabel('Entropy (X)')
    plt.title('Rössler System Entropy (X) vs Time\n' + textToDisplay)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
    """