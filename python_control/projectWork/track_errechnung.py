import numpy as np
import matplotlib.pyplot as plt

def norm(x0,y0,x1,y1):
    return np.sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1))

data = np.load('file.npy')
start=270
end=1950
x_data = data[start:end, 0]
y_data = data[start:end, 1]

Point_data_used=np.zeros((x_data.shape[0],2))
k=0
for i in range(0,x_data.shape[0]):
    if k==0:
        Point_data_used[k, 0] = x_data[i]
        Point_data_used[k, 1] = y_data[i]
        k=k+1
    else:
        if norm(Point_data_used[k-1,0],Point_data_used[k-1, 1],x_data[i],y_data[i])>10:
            Point_data_used[k, 0] = x_data[i]
            Point_data_used[k, 1] = y_data[i]
            k=k+1




Point_data_used=Point_data_used[0:k,]
print(Point_data_used.shape)
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

plt.plot(Point_data_used[0:k,0],Point_data_used[0:k,1])
#print(data.shape)
#plt.xlim([-256, 256])
plt.show()
np.save("track_x_y_pos.npy", Point_data_used)
#print(x_data.shape[0])
#print(y_data.shape)