import numpy as np
import matplotlib.pyplot as plt

N = 20
a, b = 3, 2

X = np.random.uniform(0, 3*np.pi/2, size=(N,1))
y  = a*np.sin(X) + b

data = np.hstack((X,y))
plt.scatter(X, y)
plt.ylabel("y")
plt.xlabel("X")
plt.savefig("plot_of_training_data", dpi=1000)
np.save("training_data", data)
