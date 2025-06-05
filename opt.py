import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from entropy_and_mutual_information_estimators import entropy, pmf_single_var, entropy_nvars
import pdb

def gaussian_initial_condition_generator(num_conditions):
    """
    Generates matrix of Gaussian-distributed initial conditions.
    
    Args:
        num_conditions: Number of initial conditions to generate
        
    Returns:
        np.array: Shape (num_conditions, 3) with each row being [x0, y0, z0]
    """
    mean = np.zeros(num_conditions)
    cov = np.eye(num_conditions)  # Identity matrix = no correlation
    initial_conditions = np.random.multivariate_normal(mean, cov, 3)
    return initial_conditions.T


def rossler_ode_system(t, state, a, b, c):
    """
    Rossler system differential equations.
    
    Args:
        t: Time (not used in autonomous system)
        state: [x, y, z] current state
        a, b, c: Rossler parameters
        
    Returns:
        List of derivatives [dx/dt, dy/dt, dz/dt]
    """
    x, y, z = state
    dxdt = -y - z
    dydt = x + a * y
    dzdt = b + z * (x - c)
    return [dxdt, dydt, dzdt]


def rossler_system_generator(initial_conditions, time_final, num_time_steps):
    """
    Generate Rossler system trajectory using scipy's solve_ivp.
    
    Args:
        initial_conditions: [x0, y0, z0] starting point
        time_final: Final integration time
        num_time_steps: Number of time points to evaluate
        
    Returns:
        Tuple of (trajectory_x, trajectory_y, trajectory_z) arrays
    """
    # Rossler parameters
    a, b, c = 0.2, 0.2, 5.7
    
    t_span = (0, time_final)
    t_eval_points = np.linspace(0, time_final, num_time_steps)
    
    sol = solve_ivp(
        rossler_ode_system,
        t_span,
        initial_conditions,
        args=(a, b, c),
        t_eval=t_eval_points,
        method='RK45'
    )
    
    return sol.y[0, :], sol.y[1, :], sol.y[2, :]


def calculate_joint_pmf(traj_x_t, traj_y_t, traj_z_t, num_bins, z_min, z_max):
    """
    Calculate joint PMF for X, Y, Z at a given time step.
    
    Args:
        traj_x_t, traj_y_t, traj_z_t: Arrays of trajectory values at time t
        num_bins: Number of bins for histogramdd
        z_min, z_max: Min and max values for Z coordinate
        
    Returns:
        3D numpy array representing the joint PMF
    """
    # Stack the trajectories for histogramdd
    data = np.vstack([traj_x_t, traj_y_t, traj_z_t]).T
    
    # Define ranges for each coordinate
    ranges = [(-50, 50), (-50, 50), (z_min, z_max)]
    
    # Compute 3D histogram
    hist, _ = np.histogramdd(data, bins=num_bins, range=ranges)
    
    # Normalize to get PMF
    joint_pmf = hist / hist.sum()
    
    return joint_pmf


def calculate_entropy_evolution_with_joint(trajectories_x, trajectories_y, trajectories_z, 
                                         num_conditions, num_time_steps):
    """
    Calculate entropy evolution over time using both individual and joint PMF approaches.
    
    Args:
        trajectories_x, trajectories_y, trajectories_z: Lists of trajectory arrays
        num_conditions: Number of initial conditions
        num_time_steps: Number of time steps
        
    Returns:
        Tuple of entropy arrays for individual and joint methods, plus joint entropy
    """
    print("Entropy analysis with joint PMF started")
    
    # Initialize entropy lists for individual PMF method
    entropy_x_individual, entropy_y_individual, entropy_z_individual = [], [], []
    
    # Initialize entropy lists for joint PMF marginal method
    entropy_x_joint_marg, entropy_y_joint_marg, entropy_z_joint_marg = [], [], []
    
    # Initialize joint entropy list
    entropy_xyz_joint = []
    
    num_bins = num_conditions // 40
    
    # Convert to numpy arrays for easier indexing
    traj_x = np.array(trajectories_x)
    traj_y = np.array(trajectories_y) 
    traj_z = np.array(trajectories_z)
    
    # Get global z range for consistent binning
    z_min = np.min(traj_z)
    z_max = np.max(traj_z)
    
    for i in range(num_time_steps):
        if i % 1000 == 0:
            print(f"Processing time step {i}/{num_time_steps}")
            
        # Method 1: Individual PMFs (original approach)
        pmf_x = pmf_single_var(traj_x[:, i], num_bins, -50, 50)
        entropy_x_individual.append(entropy(np.array(pmf_x)))
        
        pmf_y = pmf_single_var(traj_y[:, i], num_bins, -50, 50)
        entropy_y_individual.append(entropy(np.array(pmf_y)))
        
        pmf_z = pmf_single_var(traj_z[:, i], num_bins, z_min, z_max)
        entropy_z_individual.append(entropy(np.array(pmf_z)))
        
        # Method 2: Joint PMF and marginalization
        joint_pmf = calculate_joint_pmf(traj_x[:, i], traj_y[:, i], traj_z[:, i], 
                                       num_bins, z_min, z_max)
        
        # Calculate marginal entropies from joint PMF
        entropy_x_joint_marg.append(entropy_nvars(joint_pmf, (0,)))  # X is axis 0
        entropy_y_joint_marg.append(entropy_nvars(joint_pmf, (1,)))  # Y is axis 1
        entropy_z_joint_marg.append(entropy_nvars(joint_pmf, (2,)))  # Z is axis 2
        
        # Calculate joint entropy
        entropy_xyz_joint.append(entropy_nvars(joint_pmf, (0, 1, 2)))
    
    print("PMFs and entropies for Rossler system calculated")
    return (entropy_x_individual, entropy_y_individual, entropy_z_individual,
            entropy_x_joint_marg, entropy_y_joint_marg, entropy_z_joint_marg,
            entropy_xyz_joint)


def plot_entropy_evolution_comparison(entropy_x_ind, entropy_y_ind, entropy_z_ind,
                                    entropy_x_joint, entropy_y_joint, entropy_z_joint,
                                    entropy_xyz_joint, time_final, dt, num_conditions):
    """
    Plot entropy evolution comparing individual and joint PMF methods.
    
    Args:
        entropy_*_ind: Individual PMF entropy time series
        entropy_*_joint: Joint PMF marginal entropy time series  
        entropy_xyz_joint: Joint entropy time series
        time_final: Final time
        dt: Time step
        num_conditions: Number of initial conditions
    """
    # Enable LaTeX rendering
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    
    fs = 30  # Font size parameter
    lw = 5   # Line width parameter
    time_points = np.linspace(0, time_final, len(entropy_x_ind))
    text_display = f"Number of initial conditions: {num_conditions}   $dt$: {dt}"
    
    # Plot entropy for X coordinate - comparison
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, entropy_x_ind, color='red', linewidth=lw, 
             label='Individual PMF', linestyle='-')
    plt.plot(time_points, entropy_x_joint, color='darkred', linewidth=lw, 
             label='Joint PMF Marginal', linestyle='--')
    plt.xlabel(r'$t$', fontsize=fs)
    plt.ylabel(r'$H\left[X(t)\right]$', fontsize=fs)
    plt.title(text_display, fontsize=fs-5)
    plt.legend(fontsize=fs-5)
    plt.tick_params(labelsize=fs-5)
    plt.tight_layout()
    plt.savefig('rossler_entropy_x_comparison.png', dpi=300)
    
    # Plot entropy for Y coordinate - comparison
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, entropy_y_ind, color='green', linewidth=lw, 
             label='Individual PMF', linestyle='-')
    plt.plot(time_points, entropy_y_joint, color='darkgreen', linewidth=lw, 
             label='Joint PMF Marginal', linestyle='--')
    plt.xlabel(r'$t$', fontsize=fs)
    plt.ylabel(r'$H\left[Y(t)\right]$', fontsize=fs)
    plt.title(text_display, fontsize=fs-5)
    plt.legend(fontsize=fs-5)
    plt.tick_params(labelsize=fs-5)
    plt.tight_layout()
    plt.savefig('rossler_entropy_y_comparison.png', dpi=300)
    
    # Plot entropy for Z coordinate - comparison
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, entropy_z_ind, color='blue', linewidth=lw, 
             label='Individual PMF', linestyle='-')
    plt.plot(time_points, entropy_z_joint, color='darkblue', linewidth=lw, 
             label='Joint PMF Marginal', linestyle='--')
    plt.xlabel(r'$t$', fontsize=fs)
    plt.ylabel(r'$H\left[Z(t)\right]$', fontsize=fs)
    plt.title(text_display, fontsize=fs-5)
    plt.legend(fontsize=fs-5)
    plt.tick_params(labelsize=fs-5)
    plt.tight_layout()
    plt.savefig('rossler_entropy_z_comparison.png', dpi=300)
    
    # Plot joint entropy H(X,Y,Z)
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, entropy_xyz_joint, color='purple', linewidth=lw)
    plt.xlabel(r'$t$', fontsize=fs)
    plt.ylabel(r'$H\left[X(t),Y(t),Z(t)\right]$', fontsize=fs)
    plt.title(text_display, fontsize=fs-5)
    plt.tick_params(labelsize=fs-5)
    plt.tight_layout()
    plt.savefig('rossler_entropy_xyz_joint.png', dpi=300)


def calculate_initial_entropies(initial_conditions, num_conditions):
    """
    Calculate entropies for the initial condition distributions.
    
    Args:
        initial_conditions: Array of initial conditions
        num_conditions: Number of initial conditions
        
    Returns:
        Tuple of (entropy_x, entropy_y, entropy_z) for initial conditions
    """
    num_bins = num_conditions // 40
    
    pmf_x = np.array(pmf_single_var(initial_conditions[:, 0], num_bins, -5, 5))
    entropy_x = entropy(pmf_x)
    
    pmf_y = np.array(pmf_single_var(initial_conditions[:, 1], num_bins, -5, 5))
    entropy_y = entropy(pmf_y)
    
    pmf_z = np.array(pmf_single_var(initial_conditions[:, 2], num_bins, -5, 5))
    entropy_z = entropy(pmf_z)
    
    return entropy_x, entropy_y, entropy_z


def generate_all_trajectories(initial_conditions, time_final, time_steps):
    """
    Generate Rossler trajectories for all initial conditions.
    
    Args:
        initial_conditions: Array of initial conditions (num_conditions, 3)
        time_final: Final integration time
        time_steps: Number of time steps
        
    Returns:
        Tuple of (trajectories_x, trajectories_y, trajectories_z) lists
    """
    num_conditions = len(initial_conditions)
    trajectories_x, trajectories_y, trajectories_z = [], [], []
    
    for i in range(num_conditions):
        x_traj, y_traj, z_traj = rossler_system_generator(
            initial_conditions[i], time_final, time_steps
        )
        trajectories_x.append(x_traj)
        trajectories_y.append(y_traj)
        trajectories_z.append(z_traj)
    
    print(f"Generated {num_conditions} Rossler trajectories")
    return trajectories_x, trajectories_y, trajectories_z


def main():
    """Main execution function."""
    # Parameters
    num_conditions = 1000
    time_final = 50
    dt = 1e-2
    time_steps = int(time_final / dt)
    
    print("Starting Execution....")
    
    # Generate initial conditions
    initial_conditions = gaussian_initial_condition_generator(num_conditions)
    print("Gaussian initial conditions generated")
    
    # Calculate initial entropies
    init_entropy_x, init_entropy_y, init_entropy_z = calculate_initial_entropies(
        initial_conditions, num_conditions
    )
    print("PMFs and entropies for gaussian initial conditions calculated")
    
    # Generate trajectories for all initial conditions
    trajectories_x, trajectories_y, trajectories_z = generate_all_trajectories(
        initial_conditions, time_final, time_steps
    )
    
    # Calculate entropy evolution with both methods
    (entropy_x_ind, entropy_y_ind, entropy_z_ind,
     entropy_x_joint, entropy_y_joint, entropy_z_joint,
     entropy_xyz_joint) = calculate_entropy_evolution_with_joint(
        trajectories_x, trajectories_y, trajectories_z, num_conditions, time_steps
    )
    
    print("Plotting Results") 
    # Plot comparison results
    plot_entropy_evolution_comparison(
        entropy_x_ind, entropy_y_ind, entropy_z_ind,
        entropy_x_joint, entropy_y_joint, entropy_z_joint,
        entropy_xyz_joint, time_final, dt, num_conditions
    )
    
    print("Analysis complete! Figures saved.")


if __name__ == "__main__":
    main()
