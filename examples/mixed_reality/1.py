# MNIST strokes database loder

from pytorch_mnist_stroke import MNISTStroke
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from skimage.draw import line_aa
# %matplotlib inline
plt.style.use('classic')

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

loader = torch.utils.data.DataLoader(
  MNISTStroke(
    "/tmp/mnist_stroke", train=True
  ),batch_size=5, shuffle=True
)

X, Y = next(iter(loader))
# print(X, Y)

for x in X:
  image = torch.zeros(28, 28)
  pen_x = 0
  pen_y = 0
  for row in x:
    px, py, _, _ = row
    pen_x += int(px.item())
    pen_y += int(py.item())
    image[pen_y, pen_x] = 1

  plt.imshow(image)
  plt.show()

# for batch_X, batch_Y in loader:
#   print(batch_X, batch_Y)

  input('Press ENTER')