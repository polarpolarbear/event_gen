import numpy as np
import matplotlib.pyplot as plt


def event2histogram_alt(event_stream):
    event_stream = event_stream
    hist = np.zeros((2, 34, 34))
    xx = np.floor((event_stream[:,0]*33)).astype(int)
    yy= np.floor((event_stream[:,1]*33)).astype(int)
    pp = np.floor((event_stream[:,2])).astype(int)
    
    ii = np.where(pp == 2)
    if len(ii[0]) > 0 :
      np.add.at(hist, (pp[:ii[0][0]], yy[:ii[0][0]], xx[:ii[0][0]]), 1)
    else:
      np.add.at(hist, (pp, yy, xx), 1)
    return hist

coord = np.load("predict_event.npy")

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

ax.scatter3D(coord[:,0], coord[:,1], coord[:,2])
plt.xlabel("x")
plt.ylabel("y")
#plt.zlabel("t")
plt.show()



histogram = event2histogram_alt(coord)
image = np.zeros((34, 34, 3)) 
histogram = np.transpose(histogram, (1, 2, 0))
image[:,:,0:2] = histogram
plt.imshow(image, cmap='magma')
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.colorbar()
plt.title("Real")