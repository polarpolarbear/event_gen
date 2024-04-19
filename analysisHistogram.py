import numpy as np
import platform
from matplotlib import pyplot as plt

data = np.load("nEvent_list.npy")
mean = np.mean(data)
max = np.max(data)
print("std:",np.std(data))
print("mean:",np.mean(data))
print("max:",np.max(data))

'''
# Plotting a basic histogram
plt.hist(data)
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')
 
# Display the plot
plt.show()
plt.savefig('nEvent_list.png')
'''