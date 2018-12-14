import numpy as np


x_pos=-70
y_pos=-0.3
distance=38.4306602478
x_car=-185.49200439453125
y_car= 33.12493896484375
theta=-1.2071195287943717



alpha=np.arctan((y_pos-y_car)/(x_pos-x_car))-theta


print("alpha:",alpha)
print("scaling",(x_pos-x_car)/np.cos(alpha+theta)/distance)




