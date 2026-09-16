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

Given: ge, gi, gl - how strongly each of the 3 forces is currently pulling on Vm.
Given: E_e, E_l, E_i - where each force is trying to pull toward (reversal potential).
Computed: Vm or geThr.
Computed: act - the unit's final output: 0 if below threshold, a graded 0-1 if above.
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
        self.fbi = 0.0 # persists across cycles

    def clamp(self, pattern: np.ndarray):
        """
        Force this layer's activation directly (used for Input/Action/
        GoalPos -- layers driven by the environment, not computed)
        """
        assert pattern.shape == self.shape, f"{self.name} : shape mismatch"
        self.act = pattern.copy()
        self.clamped = True

    def unclamp(self):
        """
        Release this layer back to being computed, not forced."
        """
        self.clamped = False

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
        # Random uniform [0, 1].
        self.weights = rng.uniform(0.0, 1.0, size=(recv_layer.size, send_layer.size))


    def contribute_ge(self, send_act):
        """
        This projection's contribution to the receiving layer's ge,
        as an average over the sending layer's units
        """
        net = self.weights @ send_act.flatten() / self.send_layer.size
        return self.wt_scale * net.reshape(self.recv_layer.shape)

    def learn(self, send_act_minus: np.ndarray, recv_act_minus: np.ndarray, send_act_plus: np.ndarray, \
              recv_act_plus: np.ndarray, lrate: float = 0.02):
        """
        Apply one CHL weight update from explicit minus/plus phase snapshots of both sender and receiver.
        """
        from learn import chl_dwt
        dwt = chl_dwt(send_act_minus, recv_act_minus, send_act_plus, recv_act_plus, lrate)
        self.weights += dwt
        np.clip(self.weights, 0.0, 1.0, out=self.weights) # keep weights in [0,1]

    

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
        ONE raw cycle: every non-clamped layer computes from a FROZEN
        snapshot of every layer's PREVIOUS state, then all update together.
        Call this repeatedly (e.g. 100x) to run a full trial."""
        from fffb import fffb_step
        snapshot = {name: l.act.copy() for name, l in self.layers.items()}
        for name, l in self.layers.items():
            if l.clamped:
                continue
            incoming = [p for p in self.projections if p.recv_layer.name == name]
            if not incoming:
                continue
            ge = sum(p.contribute_ge(snapshot[p.send_layer.name]) for p in incoming)
            act, gi, l.fbi = fffb_step(ge.flatten(), l.fbi, snapshot[name].flatten(), gi_gain=1.8)
            l.act = act.reshape(l.shape)
            l.ge = ge


if __name__ == "__main__":
    from popcode import PopCode2D
    from world import Pos

    net = Network()
    net.add_layer("Input", shape=(7, 7))  # position, population-coded
    net.add_layer("Action", shape=(4,))   # one-hot: N/E/S/W
    net.add_layer("Hidden", shape=(5,5)) 

    net.connect("Input", "Hidden", wt_scale=15.0, seed=1)
    net.connect("Action", "Hidden", wt_scale=4.0, seed=2)

    # Clamp Input with a real population-coded position
    pc = PopCode2D(rows = 7, cols = 7, sigma = 1.0)
    net.layers["Input"].clamp(pc.encode(Pos(row=3, col=4)))

    # Clamp Action as one-hot: index 1 = East
    action_pattern = np.zeros(4)
    action_pattern[1] = 1.0
    net.layers["Action"].clamp(action_pattern)

    net.cycle()

    hidden = net.layers["Hidden"]
    print("Hidden layer ge (net input):")
    print(np.round(hidden.ge, 2))
    print("\nHidden layer act (after FFFB settling):")
    print(np.round(hidden.act, 2))
    print(f"\nActive Hidden units: {(hidden.act > 0.01).sum()} / {hidden.size}")

        