import numpy as np
import platform
from matplotlib import pyplot as plt
import tonic 
import pickle


path = './caltech_nEvent.pickle'

with open(path, 'rb') as handle:
    data = pickle.load(handle)
    
with open('./label.pickle', 'rb') as handle:
    labels = pickle.load(handle)

list = []
print(path)
for idx,classData in enumerate(data):
    #print("std:",np.std(data))
    print("----------")
    print(labels[idx])
    print("nSample:", len(classData))
    print("median:", np.median(classData))
    print("mean:", np.mean(classData))
    print("max:", np.max(classData))
    print("min:", np.min(classData))

    #mask = np.array(classData) < 8000
    #procent = sum(mask)/len(classData)
    #list.append(procent)
    #print("min:", procent)
    if len(classData) < 50:
         list.append(labels[idx])

# Plotting a basic histogram
plt.hist(data)
 
# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')
 
 
# Display the plot
plt.show()
plt.savefig('nEvent_list.png')

print(list)
print(len(list))