import numpy as np
import matplotlib.pyplot as plt

def norm(x0,y0,x1,y1):
    return np.sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1))

data = np.load('Track_Saved_b.npy')
print(data.shape)
start=0
end=126210
x_data = data[start:end, 0]
y_data = data[start:end, 1]

#print(data.shape)
#plt.xlim([-256, 256])

Point_data_used = np.zeros((x_data.shape[0],2))
k=0
for i in range(0,x_data.shape[0]):
    print(k)
    if k==0:
        Point_data_used[k, 0] = x_data[i]
        Point_data_used[k, 1] = y_data[i]
        k=k+1
    else:
        if norm(Point_data_used[k-1,0],Point_data_used[k-1, 1],x_data[i],y_data[i])>10 and norm(Point_data_used[k-1,0],Point_data_used[k-1, 1],x_data[i],y_data[i])<100 :
            Point_data_used[k, 0] = x_data[i]
            Point_data_used[k, 1] = y_data[i]
            k=k+1
        if norm(Point_data_used[k-1,0],Point_data_used[k-1, 1],Point_data_used[0,0],Point_data_used[0, 1])<10 and k>5:
            break




Point_data_used=Point_data_used[0:k,]
print(Point_data_used)
#plt.figure()
#plt.title("Grayscale Histogram")
#plt.xlabel("Bins")
#plt.ylabel("# of Pixels")

#plt.plot(Point_data_used[0:k,0],Point_data_used[0:k,1],"bo")
#print(data.shape)
#plt.xlim([-256, 256])
#plt.show()
np.save("track_x_y_pos_b.npy", Point_data_used)
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

plt.plot(Point_data_used[:,0],Point_data_used[:,1],"bo")
plt.show()
#print(x_data.shape[0])
#print(y_data.shape)