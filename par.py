import numpy as np
import multiprocessing
import time
from opt import (
    gaussian_initial_condition_generator,
    generate_all_trajectories,
    calculate_joint_pmf,
    plot_entropy_evolution_comparison
)
from entropy_and_mutual_information_estimators import entropy, pmf_single_var, entropy_nvars

# Global variables to be shared across processes
trajectories_x = None
trajectories_y = None
trajectories_z = None
z_min = None
z_max = None
num_conditions = None
num_bins = None

def init_worker(traj_x, traj_y, traj_z, z_min_val, z_max_val, num_cond, bins):
    """Initialize worker processes with shared data."""
    global trajectories_x, trajectories_y, trajectories_z, z_min, z_max, num_conditions, num_bins
    trajectories_x = traj_x
    trajectories_y = traj_y
    trajectories_z = traj_z
    z_min = z_min_val
    z_max = z_max_val
    num_conditions = num_cond
    num_bins = bins

def compute_entropy_at_timestep(time_indices):
    """
    Compute entropy for a chunk of time steps.
    This function will be called by worker processes.
    """
    global trajectories_x, trajectories_y, trajectories_z, z_min, z_max, num_conditions, num_bins
    
    # Convert trajectories to numpy arrays for efficient indexing
    traj_x = np.array(trajectories_x)
    traj_y = np.array(trajectories_y)
    traj_z = np.array(trajectories_z)
    
    results = []
    
    for i in time_indices:
        # Method 1: Individual PMFs
        pmf_x = pmf_single_var(traj_x[:, i], num_bins, -50, 50)
        entropy_x_ind = entropy(np.array(pmf_x))
        
        pmf_y = pmf_single_var(traj_y[:, i], num_bins, -50, 50)
        entropy_y_ind = entropy(np.array(pmf_y))
        
        pmf_z = pmf_single_var(traj_z[:, i], num_bins, z_min, z_max)
        entropy_z_ind = entropy(np.array(pmf_z))
        
        # Method 2: Joint PMF and marginalization
        joint_pmf = calculate_joint_pmf(traj_x[:, i], traj_y[:, i], traj_z[:, i], 
                                       num_bins, z_min, z_max)
        
        entropy_x_joint = entropy_nvars(joint_pmf, (0,))
        entropy_y_joint = entropy_nvars(joint_pmf, (1,))
        entropy_z_joint = entropy_nvars(joint_pmf, (2,))
        entropy_xyz_joint = entropy_nvars(joint_pmf, (0, 1, 2))
        
        results.append((i, entropy_x_ind, entropy_y_ind, entropy_z_ind,
                       entropy_x_joint, entropy_y_joint, entropy_z_joint,
                       entropy_xyz_joint))
    
    return results

def chunk_time_indices(time_steps, num_cores, skip=1):
    """Divide time indices into chunks for parallel processing."""
    indices = list(range(0, time_steps, skip))  # Skip every 'skip' time steps
    chunk_size = len(indices) // num_cores
    chunks = []
    
    for i in range(num_cores):
        start_idx = i * chunk_size
        if i == num_cores - 1:  # Last chunk gets remaining indices
            end_idx = len(indices)
        else:
            end_idx = (i + 1) * chunk_size
        chunks.append(indices[start_idx:end_idx])
    
    return chunks

def plot_results(entropy_results, time_final, dt, num_conditions):
    """Plot entropy evolution comparison using single CPU."""
    print("Starting plotting on single CPU...")
    start_time = time.time()
    
    # Sort results by time index and extract entropy arrays
    entropy_results.sort(key=lambda x: x[0])
    
    entropy_x_ind = [result[1] for result in entropy_results]
    entropy_y_ind = [result[2] for result in entropy_results]
    entropy_z_ind = [result[3] for result in entropy_results]
    entropy_x_joint = [result[4] for result in entropy_results]
    entropy_y_joint = [result[5] for result in entropy_results]
    entropy_z_joint = [result[6] for result in entropy_results]
    entropy_xyz_joint = [result[7] for result in entropy_results]
    
    # Use the plotting function from Opt_v2.py
    plot_entropy_evolution_comparison(
        entropy_x_ind, entropy_y_ind, entropy_z_ind,
        entropy_x_joint, entropy_y_joint, entropy_z_joint,
        entropy_xyz_joint, time_final, dt, num_conditions
    )
    
    end_time = time.time()
    print(f"Plotting completed in {end_time - start_time:.2f} seconds")
    print("All plots saved!")

def main():
    """Main execution function with parallelization."""
    # Parameters
    num_conditions_param = 1000
    time_final = 200
    dt = 1e-3
    time_steps = int(time_final / dt)
    skip = 100  # Compute entropy every 'skip' time steps (1 = every step, 2 = every other step, etc.)
    
    # Get number of CPU cores
    num_cores = 10 
    print(f"Detected {num_cores} CPU cores")
    print(f"Using {num_cores} cores for entropy computation")
    
    total_start_time = time.time()
    
    print("Starting Parallel Rossler Entropy Analysis...")
    
    # Step 1: Generate initial conditions (single CPU)
    print("\n=== STEP 1: Generating Initial Conditions ===")
    initial_conditions = gaussian_initial_condition_generator(num_conditions_param)
    print("Gaussian initial conditions generated")
    
    # Step 2: Generate trajectories (single CPU)
    print("\n=== STEP 2: Generating Trajectories ===")
    trajectories_x, trajectories_y, trajectories_z = generate_all_trajectories(
        initial_conditions, time_final, time_steps
    )
    
    # Get global z range for consistent binning
    traj_z_array = np.array(trajectories_z)
    z_min_val = np.min(traj_z_array)
    z_max_val = np.max(traj_z_array)
    num_bins_val = num_conditions_param // 40
    
    print(f"Z range: [{z_min_val:.3f}, {z_max_val:.3f}]")
    print(f"Number of bins: {num_bins_val}")
    
    # Step 3: Parallel entropy computation
    print(f"\n=== STEP 3: Computing Entropies in Parallel ({num_cores} cores) ===")
    print(f"Skip parameter: {skip} (computing entropy every {skip} time step{'s' if skip > 1 else ''})")
    entropy_start_time = time.time()
    
    # Divide time steps among cores (with skip)
    time_chunks = chunk_time_indices(time_steps, num_cores, skip)
    total_entropy_computations = sum(len(chunk) for chunk in time_chunks)
    print(f"Divided {total_entropy_computations} entropy computations into {len(time_chunks)} chunks")
    for i, chunk in enumerate(time_chunks):
        if len(chunk) > 0:
            print(f"Core {i}: {len(chunk)} time steps (t={chunk[0]} to t={chunk[-1]})")
        else:
            print(f"Core {i}: 0 time steps")
    
    # Parallel entropy computation
    multiprocessing.freeze_support()
    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(trajectories_x, trajectories_y, trajectories_z, 
                 z_min_val, z_max_val, num_conditions_param, num_bins_val)
    ) as pool:
        chunk_results = pool.map(compute_entropy_at_timestep, time_chunks)
    
    # Flatten results from all chunks
    all_results = []
    for chunk_result in chunk_results:
        all_results.extend(chunk_result)
    
    entropy_end_time = time.time()
    print(f"Parallel entropy computation completed in {entropy_end_time - entropy_start_time:.2f} seconds")
    print(f"Computed entropies for {len(all_results)} time steps (skip={skip})")
    
    # Step 4: Plot results (single CPU)
    print("\n=== STEP 4: Plotting Results ===")
    # Adjust time_final for plotting based on skip to get correct time axis
    effective_dt = dt * skip
    plot_results(all_results, time_final, effective_dt, num_conditions_param)
    
    total_end_time = time.time()
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Total execution time: {total_end_time - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
