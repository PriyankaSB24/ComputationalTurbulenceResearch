import numpy as np
from scipy.integrate import solve_ivp 
import multiprocessing # This is the package that does the heavy lifting for parallel processing; you might need to install it via pip or conda 
import time


### This script demonstrates how to run multiple ODE simulations; it will run 1 ode per cpu core. You can replace the ODE part here with the process you want to run in parallel for instance computing entropy across different time steps ####
# ============================================================================
# STEP 1: Define Your ODE System
# ============================================================================

def simple_ode(t, y):
    """A simple exponential decay: dy/dt = -y"""
    return -y

# ============================================================================
# STEP 2: Function to Run ONE Simulation
# ============================================================================

def run_single_simulation(initial_value):
    """
    Run a single ODE simulation with a given initial value.
    
    Args:
        initial_value: Starting value for the ODE
    
    Returns:
        Result of the simulation
    """
    # Time points from 0 to 5
    t_span = [0, 5]
    t_eval = np.linspace(0, 5, 100)
    
    # Solve the ODE
    result = solve_ivp(simple_ode, t_span, [initial_value], t_eval=t_eval)
    
    # Return the final value
    return result.y[0, -1]

# ============================================================================
# STEP 3: Run Multiple Simulations in Parallel
# ============================================================================

def main(): 
    # timesteps is number of times the for loop should run
    # 
    """Main function to demonstrate parallel processing"""
    
    # SEQUENTIAL (normal) way - runs one at a time
    print("Running simulations one at a time (sequential)...")
    initial_values = [1, 2, 3, 4, 5, 6, 7, 8]
    
    start_time = time.time()
    sequential_results = []
    for value in initial_values:
        result = run_single_simulation(value)
        sequential_results.append(result)
    sequential_time = time.time() - start_time
    
    print(f"Sequential results: {sequential_results}")
    print(f"Time taken: {sequential_time:.2f} seconds")
    print()
    
    # PARALLEL way - runs multiple simulations at once; in this case it will run 8 simulations at once with 1 simulation per CPU core (this would use 8 cores)
    print("Running simulations in parallel...")
    
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        parallel_results = pool.map(run_single_simulation, initial_values)
    parallel_time = time.time() - start_time
    
    print(f"Parallel results: {parallel_results}")
    print(f"Time taken: {parallel_time:.2f} seconds")
    print(f"Speedup: {sequential_time/parallel_time:.1f}x faster!")



if __name__ == "__main__":
    # Check how many CPU cores you have
    print(f"Your computer has {multiprocessing.cpu_count()} CPU cores")
    print("Each simulation will run on a different core")
    print()
    
    main()
