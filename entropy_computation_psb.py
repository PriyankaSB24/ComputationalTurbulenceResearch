# A PMF is  a list of probabilities that tell us how likely it is to observe each possible state in the system. 
# PMF tells us the probability of the particle being at each possible position in a distribution
# Each simulation produces a series of "states" at each time step
# reak the possible outcomes into "bins" — small intervals over which you will count how often certain outcomes occur
# Calculate PMF: count how many times each bin is "hit" and then divide by the total number of data points
# Calculate entropy: If all outcomes are equally likely, the entropy is high because there's maximum uncertainty. (The system could be in any state with equal probability.)
# If one outcome is much more likely than others, entropy is low because the system is very predictable.

import numpy as np
import matplotlib.pyplot as plt
from entropy_and_mutual_information_estimators import entropy,pmf_single_var
import pdb
nbins = 10
lowerBound = 0
upperBound = 1
numPoints = 150
numSimulations = 50
data=[]
pmf = []
entropy_results = []

for i in range (numSimulations):
    # random generation of 'numPoints' points between 0 and 1 :
    data.append(np.random.rand(numPoints)) 
    pmf.append(pmf_single_var(data,nbins,lowerBound,upperBound))
    entropy_results.append(entropy(pmf[i]))
    pdb.set_trace()

# Plot entropy vs. num simulation
# entropy levels out at log_2(10) due to random uniform distribution = max entropy 
# = H = -∑(i=1 to 10) pᵢ log(pᵢ) = -10 × (1/10) log(1/10) = log(10)
plt.figure(figsize=(25, 6))
plt.plot(range(len(entropy_results)), entropy_results, marker='o') 
plt.xlabel('Number of Simulation')
plt.ylabel('Entropy')
plt.title('Entropy vs. Number of Simulations')
plt.xticks(np.arange(0, len(entropy_results), 1))
plt.grid(True)
plt.show()

# Plot histogram of data values
plt.hist(data[0], bins=nbins, range=(lowerBound, upperBound),
         edgecolor='black', color='skyblue')
plt.xlabel('Data Values')
plt.ylabel('Number of Data Values within Bins')
plt.title('Histogram of Simulation 1 with Entropy')
plt.grid(True)
plt.tight_layout()
plt.show()