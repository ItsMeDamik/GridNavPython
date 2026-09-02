"""
FFFB version: feedforward + feedback inhibition.
Replaces the sort-based kWTA with leabra's actual current mechanism: a single
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
from neuron import E_E, E_L, E_I, G_L, THETA, GAIN, vm_equilibrium, activation


def fffb_settle(
        ge: np.ndarray,
        gi_gain: float = 1.8,
        ff: float = 1.0,
        fb: float = 1.0,
        ff0: float = 0.1,
        fb_tau: float = 1.4,
        max_vs_avg: float = 0.0,
        n_cycles: int = 30,
):
    """
    Run the FFFB feedback loop to a settled (approximate) 
    equilibrium. Returns (final_gi, final_act, history) where 
    history is a list of (gi, act) pairs, one per cycle,
    so you can watch it converge.
    """
    fb_dt = 1.0 / fb_tau
    fbi = 0.0
    act = np.zeros_like(ge) # start fully silent, nothing has fired yet
    history = []

    avg_ge = ge.mean()
    max_ge = ge.max()
    ff_netin = avg_ge + max_vs_avg * (max_ge - avg_ge)
    ffi = ff * max(ff_netin - ff0, 0.0) # FF only depends on ge -- constant across cycles

    for _ in range(n_cycles):
        avg_act = act.mean()
        fbi = fbi + fb_dt * (fb * avg_act - fbi) # leaky integrator toward fb*avg_act
        gi = gi_gain * (ffi + fbi)
        vm = vm_equilibrium(ge, gi)
        act = activation(vm)
        history.append((gi, act.copy()))

    return gi, act, history


if __name__ == "__main__":
    ge = np.array([0.9, 0.3, 0.7, 0.1, 0.85, 0.2, 0.5, 0.05, 0.6, 0.4])

    final_gi, final_act, history = fffb_settle(ge)

    print("Cycle-by-cycle settling (gi, then how many units are active):")
    for cyc, (gi, act) in enumerate(history):
        n_active = (act > 0.01).sum()
        if cyc < 5 or cyc % 5 == 0 or cyc == len(history) - 1:
            print(f" cycle {cyc:2d}: gi={gi:.3f} active_units={n_active}")

    print()
    print("Final activations:")
    for i, (g, a) in enumerate(zip(ge, final_act)):
        marker = " <- active" if a > 0.01 else ""
        print(f" unit {i}: ge={g:.2f} act={a:.3f}{marker}")
    print(f"\nTotal active units: {(final_act > 0.01).sum()} out of {len(ge)}")

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