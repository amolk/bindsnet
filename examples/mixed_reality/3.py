from bindsnet.datasets import ImageDataset, MNIST
from bindsnet.encoding import bernoulli
from bindsnet.environment import DatasetEnvironment
from bindsnet.learning import PostPre
from torch.nn.modules.utils import _pair
from bindsnet.network import Network
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import Connection, RFConnection
from bindsnet.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
from typing import Optional, Union, Tuple, List, Sequence

def get_dataset():
  images = torch.as_tensor([
    [
      [0, 0, 0, 0,
       1, 1, 1, 1,
       0, 0, 0, 0,
       0, 0, 0, 0],
    ],
    # [
    #   [1, 1, 1, 1,
    #    0, 0, 0, 0,
    #    0, 0, 0, 0,
    #    0, 0, 0, 0],
    # ],
    [
      [1, 0, 0, 0,
       1, 0, 0, 0,
       1, 0, 0, 0,
       1, 0, 0, 0],
    ],
    # [
    #   [0, 0, 0, 1,
    #    0, 0, 0, 1,
    #    0, 0, 0, 1,
    #    0, 0, 0, 1],
    # ],
  ]).float() * 0.5

  labels = torch.as_tensor(range(images.shape[0]))
  dataset = ImageDataset(images, labels)
  return dataset

class MixedRealityNetwork(Network):
    # language=rst
    """
    Implements a mixed reality network
    """

    def __init__(self, source_shape: Union[int, Tuple[int, int]],
                 target_shape: Union[int, Tuple[int, int]],
                 kernel_shape: Union[int, Tuple[int, int]],
                 dt: float = 1.0, wmin: float = 0.0, wmax: float = 1.0,
                 nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
                 norm: float = None) -> None:
        # language=rst
        """
        Constructor for class ``MixedRealityNetwork``.

        :param source_shape: Shape of input neurons. Matches the 2D size of the input data.
        :param target_shape: Shape of neurons in the ``LIFNodes`` population.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization constant.
        """
        super().__init__(dt=dt)
        self.source_shape = _pair(source_shape)
        self.target_shape = _pair(target_shape)
        self.kernel_shape = _pair(kernel_shape)

        self.source_size = int(np.prod(self.source_shape))
        self.target_size = int(np.prod(self.target_shape))
        self.dt = dt

        self.add_layer(Input(n=self.source_size, traces=True, trace_tc=5e-2), name='X')
        self.add_layer(AdaptiveLIFNodes(n=self.target_size, traces=True, sum_input=True,
                                       rest=-65.0, reset=-65.0, thresh=-52.0, refrac=5,
                                       decay=1e-2, trace_tc=5e-2, theta_decay=1e-5,
                                       theta_plus=1e-2), name='Y')

        self.connection = RFConnection(source=self.layers['X'], source_shape=self.source_shape,
                                       target=self.layers['Y'], target_shape=self.target_shape,
                                       update_rule=PostPre, nu=nu, wmin=wmin, wmax=wmax, norm=norm,
                                       kernel_shape=self.kernel_shape)
        self.add_connection(self.connection,
                            source='X', target='Y')

dataset = get_dataset()
source_shape = (4,4)
target_shape = (6,6)
kernel_shape = (3,3)
network = MixedRealityNetwork(source_shape=source_shape, target_shape=target_shape, kernel_shape=kernel_shape, norm=1, nu=(1e-4, 1e-2))

environment = DatasetEnvironment(dataset=dataset, train=True, intensity=1)

# Build pipeline from components.
pipeline = Pipeline(network=network, environment=environment, plot_type='line',
                    encoding=bernoulli, time=400, plot_interval=1)

# Train the network.
# print("w", network.connection.w)
for i in range(20):
    print(f"step {i}")
    pipeline.step()

    # print("w", network.connection.w)
    # plt.imshow(network.connection.w.view(source_shape * source_shape, target_shape * target_shape))
    # plt.imshow(network.layers['Y'].theta.view(target_shape, target_shape))
    plt.imshow(network.layers['Y'].summed.view(target_shape, target_shape))
    # print("theta", network.layers['Y'].theta)
    network.reset_()
print("w", network.connection.w)
input('Press ENTER to exit')