"""
The learning rule.
Implements CHL (Contrastive Hebbian Learning). XCAL will be implemented later; it learns from continuous short/medium/long-term
running averages of activaton sampled across the whole ettling trajectory.
CHL only looks at the two settled endpoints (minus, plus phase), so it is simpler to implement.

dwt[i,j] = lrate * (send_plus[j] * recv_plus[i] - send_minus[j] * recv_minus[i])

Strengthen a connection if sender and receiver were more co-active when the correct outcome was known (plus)
than when the network was only guessing (minus); weaken it if in reverse.
"""

import numpy as np

def chl_dwt(send_act_minus: np.ndarray, recv_act_minus: np.ndarray, send_act_plus: np.ndarray, recv_act_plus: np.ndarray, \
            lrate: float) -> np.ndarray:
    """
    CHL weight change for one projection: both sender and receiver may differ between plus and minus phases.
    Returns a (recv_size, send_size) array, same shape as Projection.wts.
    """
    plus_term = np.outer(recv_act_plus.flatten(), send_act_plus.flatten())
    minus_term = np.outer(recv_act_minus.flatten(), send_act_minus.flatten())
    return lrate * (plus_term - minus_term)