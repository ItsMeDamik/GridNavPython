"""
FFFB version: feedforward + feedback inhibition.
Replaces the sort-based kWTA with leabra's current mechanism: a single
layer-wide inhibitiory conductance (gi) built frpm two components --

    FF (feedforward): proportional to how much excitatory drive is arriving into
    the layer right now (avg/max of ge across units)
    FB (feedback): proportional to how active the layer already is (avg 
    activation), integrated over cycles with a leaky integrator to avoid oscillation.

Because FB depends on activation, and activation depends on gi (which depends on
FB) -- this is circular, and has to be *settled* over several cycles rather than
computed in one shot.
"""
import numpy as np
from neuron import activation


def fffb_cycle(ge: np.ndarray, fbi_prev: float, prev_act: np.ndarray,
              gi_gain: float = 1.8, ff: float = 1.0, fb: float = 1.0,
              ff0: float = 0.1, fb_tau: float = 8.0):
    """One settling cycle's worth of feedforward+feedback inhibition."""
    avg_ge = ge.mean()
    ffi = ff * max(avg_ge - ff0, 0.0)
    avg_act = prev_act.mean() if prev_act is not None else 0.0
    fbi_new = fbi_prev + (1.0 / fb_tau) * (fb * avg_act - fbi_prev)
    gi = gi_gain * (ffi + fbi_new)
    return activation(ge, gi), gi, fbi_new


if __name__ == "__main__":
    ge = np.array([0.9, 0.3, 0.7, 0.1, 0.85, 0.2, 0.5, 0.05, 0.6, 0.4])
    fbi_prev = 0.0
    prev_act = np.zeros_like(ge)

    print("Single-cycle stepping with persistent FFB state:")
    for cyc in range(10):
        act, gi, fbi_prev = fffb_cycle(ge, fbi_prev, prev_act)
        prev_act = act
        n_active = (act > 0.01).sum()
        print(f" cycle {cyc:2d}: gi={gi:.3f} fbi={fbi_prev:.3f} active_units={n_active}")

    print()
    print("Final activations:")
    for i, (g, a) in enumerate(zip(ge, prev_act)):
        marker = " <- active" if a > 0.01 else ""
        print(f" unit {i}: ge={g:.2f} act={a:.3f}{marker}")
    print(f"\nTotal active units: {(prev_act > 0.01).sum()} out of {len(ge)}")

"""
the main takeaway is that we have ge and we need to compute gi. 
gi is a sum of (ffi and fbi). 
feedforward inhibition is proportional to the excitatory 
drive coming into the layer -> ge (not sure why ff_netin is not avg_ge, 
it is avg_ge + smth)
fbi is proportinal to how active the layer already rn, so 
it depends on on average activations of the layer (proportionality
is not instantaneous, it is a leaky integrator).
settling happens because initially gi is ffi only as there 
are no activations yet => makes some units active => act 
increase => increase the fbi => gi increase => vm decrease 
=> act decrease => until they settle into an eq. 
"""