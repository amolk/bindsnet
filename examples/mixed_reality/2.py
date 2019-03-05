from bindsnet.datasets import ImageDataset, MNIST
from bindsnet.encoding import bernoulli
from bindsnet.environment import DatasetEnvironment
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import Connection, LocallyConnectedConnection
from bindsnet.pipeline import Pipeline
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset
from typing import Optional, Union, Tuple, List, Sequence

class MixedRealityNetwork(Network):
    # language=rst
    """
    Implements a mixed reality network
    """

    def __init__(self, n_input: int, n_neurons: int = 100, dt: float = 1.0, wmin: float = 0.0, wmax: float = 1.0,
                 nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2), norm: float = None) -> None:
        # language=rst
        """
        Constructor for class ``MixedRealityNetwork``.

        :param n_input: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of neurons in the ``LIFNodes`` population.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization constant.
        """
        super().__init__(dt=dt)

        self.n_input = n_input
        self.n_neurons = n_neurons
        self.dt = dt

        self.add_layer(Input(n=self.n_input, traces=True, trace_tc=5e-2), name='X')
        self.add_layer(AdaptiveLIFNodes(n=self.n_neurons, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=5,
                                decay=1e-2, trace_tc=5e-2, theta_decay=1e-6, theta_plus=1e-4), name='Y')

        w = 0.3 * torch.rand(self.n_input, self.n_neurons)
        self.connection = Connection(source=self.layers['X'], target=self.layers['Y'], w=w, update_rule=PostPre,
                                       nu=nu, wmin=wmin, wmax=wmax, norm=norm)
        self.add_connection(self.connection,
                            source='X', target='Y')

images = torch.as_tensor([
  [
    [1, 1, 1, 1,
     0, 0, 0, 0,
     0, 0, 0, 0,
     0, 0, 0, 0],
  ],
  # [
  #   [0, 0, 0, 0,
  #    1, 1, 1, 1,
  #    0, 0, 0, 0,
  #    0, 0, 0, 0],
  # ],
]).float()

labels = torch.as_tensor([0])
dataset = ImageDataset(images, labels)
input_size = 4
layer1_size = 1
network = MixedRealityNetwork(n_input=input_size*input_size, n_neurons=layer1_size*layer1_size, norm=1)

environment = DatasetEnvironment(dataset=dataset, train=True, intensity=1)

# Build pipeline from components.
pipeline = Pipeline(network=network, environment=environment, plot_type='line',
                    encoding=bernoulli, time=1000, plot_interval=1)

# Train the network.
print("w", network.connection.w)
for i in range(30):
    print(f"step {i}")
    pipeline.step()
    #network.reset_()

    print("w", network.connection.w)
    plt.imshow(network.connection.w.view(input_size, input_size))
    # plt.imshow(network.layers['Y'].theta.view(layer1_size, layer1_size))
    print("theta", network.layers['Y'].theta)
input('Press ENTER to exit')