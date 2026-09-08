"""
Network Architecture.

Layer      = holds a name, a shape, and current activation values.
Projection = a weighted connection from one layer to another; used to turn the
             sending layer's activation into a contribution to the receiving layer's 
             ge (net input)
Network    = owns a set of layers and projections, and runs one forward pass: compute
             ge for every layer from its incoming projections, then settle each layer's
             activation via FFFB.

Mirrors gridnav's ConfigNet, just with fewer layers: Input (position) + Action -> Hidden.
"""

import numpy as np
from fffb import fffb_settle



class Layer:
    def __init__(self, name: str, shape: tuple[int, ...]):
        self.name = name
        self.shape = shape
        self.size = int(np.prod(shape))
        self.act = np.zeros(shape)  # current activation pattern
        self.ge = np.zeros(shape)  # current net input (excitatory drive)
        self.clamped = False

    def clamp(self, pattern: np.ndarray):
        """
        Force this layer's activation directly (used for Input/Action/
        GoalPos -- layers driven by the environment, not computed)
        """
        assert pattern.shape == self.shape, f"{self.name} : shape mismatch"
        self.act = pattern.copy()
        self.clamped = True

    def __repr__(self):
        return f"Layer({self.name}, shape={self.shape}, clamped={self.clamped})"


class Projection:
    """A weighted connection: send_layer -> recv_layer."""

    def __init__(self, send_layer: Layer, recv_layer: Layer, wt_scale: float = 1.0, seed = None):
        self.send_layer = send_layer
        self.recv_layer = recv_layer
        self.wt_scale = wt_scale
        rng = np.random.default_rng(seed)
        # One weight per (recv_unit, send_unit) pair -- full connectivity.
        # Random uniform [0, 1], mimicking leabra's normalized weight range.
        self.weights = rng.uniform(0.0, 1.0, size=(recv_layer.size, send_layer.size))

    def contribute_ge(self) -> np.ndarray:
        """
        This projection's contribution to the receiving layer's ge,
        as an average over the sending layer's units
        """
        send_act_flat = self.send_layer.act.flatten()
        net = self.wts @ send_act_flat / self.send_layer.size
        return self.wt_scale * net.reshape(self.recv.shape)

    

class Network:
    def __init__(self):
        self.layers: dict[str, Layer] = {}
        self.projections: list[Projection] = []

    def add_layer(self, name: str, shape: tuple[int, ...]) -> Layer:
        layer = Layer(name, shape)
        self.layers[name] = layer
        return layer

    def connect(self, send_name: str, recv_name: str, wt_scale: float = 1.0, seed = None) -> Projection:
        proj = Projection(self.layers[send_name], self.layers[recv_name], wt_scale, seed)
        self.projections.append(proj)
        return proj

    def cycle(self):
        """
        One forward pass: every non-clamped layer collects ge from its incoming,
        then settles its activation via FFFB.
        """
        # First, reset ge for layers we're about to compute (non clamped ones)
        incoming = {name: [] for name in self.layers if not self.layers[name].clamped}
        