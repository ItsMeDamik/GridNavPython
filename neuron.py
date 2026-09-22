"""
Rate-Code Neurons.

Implements leabra's point neuron activation function: a unit's membrane potential
is a an equilibrium of excitatory/inhibitory/leak conductances, each pulling
toward its own reversal potential. Activation is a saturating threshold function 
of that membrane (X-over-X-plus-1 or XX1).

Inhibition is computed via FFFB mechanism.
"""

import numpy as np

# Reversal potentials, normalized to a 0...1 scale (leabra convention).
E_E = 1.0   # excitatory reversal potential -- conductance here pulls UP
E_L = 0.3   # leak reversal potential -- resting baseline
E_I = 0.25  # inhibitory reversal potential -- pulls DOWN, below rest
G_L = 0.1   # leak conductance -- constant "pull toward rest" strength
THETA = 0.5 # firing threshold, on the Vm scale
GAIN = 100  # activation function gain (how sharply it saturates)


def ge_thr(gi, gl: float = G_L, theta: float = THETA) -> np.ndarray:
    """
    The ge value that would put a unit exactly at threshold, given the (already-known)
    gi and gl. 
    Activation is driven by (ge - geThr).
    """
    return (gi * (E_I - theta) + gl * (E_L - theta)) / (theta - E_E)


def activation(ge: np.ndarray, gi, gl: float = G_L, theta: float = THETA, gain: float = GAIN) -> np.ndarray:
    """
    Saturating threshold function ('noisy x-over-x-plus-1' style)
    """
    thr = ge_thr(gi, gl, theta)
    x = gain * np.maximum(ge - thr, 0.0)
    return x / (x + 1.0)


if __name__ == "__main__":
    # A toy layer of 10 units receiving varied excitatory drive (imagine this
    # came from a weighted sum of sending-unit activations).
    ge = np.array([0.9, 0.3, 0.7, 0.1, 0.85, 0.2, 0.5, 0.05, 0.6, 0.4])

    for k in [1, 8]:
        act = activation(ge)
        print(f"k={k}:")
        for i, (g, a) in enumerate(zip(ge, act)):
            marker = " <- active" if a > 0.01 else ""
            print(f" unit {i}: ge={g:.2f} act={a:.3f}{marker}")
        print(f" number of active units: {(act > 0.01).sum()}\n")
