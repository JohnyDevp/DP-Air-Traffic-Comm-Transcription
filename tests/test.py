import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

arr = torch.tensor(np.arange(-10, 10,0.001))
m = nn.GELU()

y = m(arr)

plt.plot(arr, y)
plt.show()

