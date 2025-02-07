from micrograd.engine import *
import random

# nin: number of inputs


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        activation = self.b
        for wi, xi in zip(self.w, x):
            activation += wi * xi

        out = activation.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

# nin: number of inputs
# nout: number of neurons


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = []
        for n in self.neurons:
            outs.append(n(x))
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

# nin = number of inputs
# nouts = list of define the number of neurons of all the layers


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())

        return params
