"""Model B: probability the project stalls before power-on.

    P_stall = 1 - (1 - P_local)(1 - P_state)(1 - P_commercial)

P_local will come from a hazard model on jurisdiction-month panels (features: sentiment_features(), moratorium stage,
neighbours' actions). Until it is trained, the layer starts from the prior in config/scoring.json and moves with the
features that are answered on a logit scale; missing features leave the prior unchanged (they are not read as calm)."""
from __future__ import annotations

import math

from .. import config as C

METHOD = 'A starting chance for each layer (local, state, commercial), moved up or down by local opposition, moratorium stage and discussion trend where those are known'
SHIFT = {'share_oppose': (1.5, 0.3), 'moratorium_stage': (0.3, 0.0), 'volume_trend': (0.15, 0.0)}   # feature: (logit per unit, neutral level); expert guesses
CLIP = {'volume_trend': (-1.0, 2.0)}


def model_b_stall(p_local: float, p_state: float, p_commercial: float) -> float:
    return 1 - (1 - p_local) * (1 - p_state) * (1 - p_commercial)


def _logit(p):
    return math.log(p / (1 - p))


def p_local(features: dict) -> dict:
    prior = C.scoring()['stall_priors']['p_local']
    z, used = _logit(prior), []
    for k, (beta, neutral) in SHIFT.items():
        v = features.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            lo, hi = CLIP.get(k, (float('-inf'), float('inf')))
            z += beta * (min(hi, max(lo, float(v))) - neutral)
            used.append(k)
    return {'p': 1 / (1 + math.exp(-z)), 'prior': prior, 'used': used}


def stall(features: dict, commercial_status: str | None = None, p_state: float | None = None) -> dict:
    pri = C.scoring()['stall_priors']
    loc = p_local(features)
    st = pri['p_state'] if p_state is None else p_state
    com = pri['p_commercial_by_status'].get(commercial_status or 'Unknown', pri['p_commercial'])
    return {'p_stall': model_b_stall(loc['p'], st, com), 'p_local': loc['p'], 'p_state': st, 'p_commercial': com,
            'local_inputs': loc['used'], 'method': METHOD}
