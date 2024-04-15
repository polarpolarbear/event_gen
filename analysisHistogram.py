import numpy as np
import platform
from matplotlib import pyplot as plt

data = np.load("nEvent_list.npy")
 
std = np.std(data)
mean = np.mean(data)

# Plotting a basic histogram
plt.hist(data)
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')
 
# Display the plot
plt.show()

plt.savefig('nEvent_list.png')

print(mean-std)
#print(mean+2*std)
print(np.min(data))
print(np.max(data))