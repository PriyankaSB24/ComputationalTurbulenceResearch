## **Computational Turbulence: Entropy Modeling of Rössler Dynamical Systems**
This work investigates the evolution of **Shannon entropy** in the Rössler dynamical system using computational methods. It examines how entropy evolves over time for each state variable (X, Y, Z), offering insight into the system's complexity and predictability under chaotic dynamics. The file descriptions below are organized from most recent to earliest.

#### **1. `Optimized_EntropyModelingLorenzSystem.py`**
* **Purpose:** Optimized version of entropy modeling that prepares the system’s state evolution (X, Y, Z arrays) for multiprocessing.
* **Usage:** Run this before parallel entropy calculations to generate necessary data structures.

#### **2. `EntropyParralelProcessing.py`**
* **Purpose:** Core script that uses Python's multiprocessing library to compute entropy values at selected time steps in parallel.
* **Functions:**
  * `compute_entropy_at_timestep()`: Computes PMF and entropy for a specific time index.
  * `main()`: Runs the multiprocessing pipeline and plots entropy vs. time for each dimension.
  
#### **3. `entropy_computation_psb.py`**
* **Purpose:** Standalone utility for entropy computation. Can be integrated into other scripts or tested independently.
* **Content:** Implements entropy-specific calculations, usable outside of dynamical system modeling.

NOTE : RUN FILES 1 AND 2 TOGETHER while importing the entropy functions from FILE 3

#### **4. `EntropyModelingLorenzSystem.py`**
* **Purpose:** Computes entropy values of the Rössler system over time without parallel processing.
* **Usage:** Provides a reference implementation of the entropy computation pipeline.

#### **5. `Multiprocessing.py`**
* **Purpose:** A minimal, general-purpose example demonstrating how Python’s `multiprocessing` module works.
* **Usage:** Educational script to understand and debug multiprocessing before applying it to entropy tasks.

#### **6. `entropy_and_mutual_information_estimators.py`**
* **Purpose:** Collection of helper functions, including:
  * Shannon entropy estimator
  * Probability mass function (PMF) generators
* **Usage:** Shared library imported across the project for entropy and PMF calculations.
  
#### **7. `ShannonEntropygraph_psb.py`**
* **Purpose:** Standalone script to visualize entropy graphs using matplotlib.
* **Usage:** Useful for plotting entropy from saved arrays or external data.
