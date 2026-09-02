"""
Rate-Code Neurons.

Implements leabra's point neuron activation function: a unit's membrane potential
is a an equilibrium of excitatory/inhibitory/leak conductances, each pulling
toward its own reversal potential. Activation is a saturating threshold function 
of that membrane (X-over-X-plus-1 or XX1).

Inhibition is computed via k-Winners-Take-All (kWTA): find the single inhibitory
conductance for the whole layer such that (approximately) the top-k units by
excitatory drive end up active, and the rest don't.
"""

import numpy as np

# Reversal potentials, normalized to a 0...1 scale (leabra convention).
E_E = 1.0   # excitatory reversal potential -- conductance here pulls UP
E_L = 0.3   # leak reversal potential -- resting baseline
E_I = 0.25  # inhibitory reversal potential -- pulls DOWN, below rest
G_L = 0.1   # leak conductance -- constant "pull toward rest" strength
THETA = 0.5 # firing threshold, on the Vm scale
GAIN = 100  # activation function gain (how sharply it saturates)


def vm_equilibrium(ge: np.ndarray, gi, gl: float = G_L) -> np.ndarray:
    """
    Equilibrium membrane potential: a conductance-weighted average of the three
    reversal potentials. gi may be a scalar (one shared inhibition value for 
    the whole layer) or a per-unit array.
    """
    return (ge * E_E + gl * E_L + gi * E_I) / (ge + gl + gi)


def activation(vm: np.ndarray, theta: float = THETA, gain: float = GAIN) -> np.ndarray:
    """
    Saturating threshold function ('noisy x-over-x-plus-1' style):
    zero beow threshold, rises and saturates toward 1 above it.
    """
    x = gain * np.maximum(vm - theta, 0.0)
    return x / (x + 1.0)


def inhib_needed_for_threshold(ge: np.ndarray, gl: float = G_L, theta: float = THETA) -> np.ndarray:
    """
    For each unit, solve for the inhibitory conductance that would put THAT 
    UNIT exactly at firing threshold, given its own excitatory drive.
    Derived by setting vm_equilibrium(ge, gi) == theta and solving for gi.
    """
    return (ge * (E_E - theta) + gl * (E_L - theta)) / (theta - E_I)


def kwta(ge: np.ndarray, k: int) -> float:
    """
    Pick one gloabl inhibitory conductance for the whole layer such that 
    (approximately) the top-k units by ge end up above threshold. 
    """
    gi_needed = inhib_needed_for_threshold(ge)
    ranked = np.sort(gi_needed)[::-1] # descending: most-easily-active first
    if k >= len(ge):
        return ranked[-1] * 0.9 # let everyone through
    g_k = ranked[k - 1] # inhibition that would just keep unit k active
    g_k1 = ranked[k] # inhibition that woud just silence unit k+1 
    return 0.5 * (g_k + g_k1) # halfway between the two so that k units are active and the rest aren't


def layer_activation(ge: np.ndarray, k: int) -> np.ndarray:
    """
    Full pipeline: excitatory input in, sparse activation pattern out.
    """
    gi = kwta(ge, k)
    vm = vm_equilibrium(ge, gi)
    return activation(vm)


if __name__ == "__main__":
    # A toy layer of 10 units receiving varied excitatory drive (imagine this
    # came from a weighted sum of sending-unit activations).
    ge = np.array([0.9, 0.3, 0.7, 0.1, 0.85, 0.2, 0.5, 0.05, 0.6, 0.4])

    for k in [1, 8]:
        act = layer_activation(ge, k)
        print(f"k={k}:")
        for i, (g, a) in enumerate(zip(ge, act)):
            marker = " <- active" if a > 0.01 else ""
            print(f" unit {i}: ge={g:.2f} act={a:.3f}{marker}")
        print(f" number of active units: {(act > 0.01).sum()}\n")

