"""A3 gate task 5: which astro parameters move group-scale f_gas (WP-A3).

Correlates the per-design-point group-scale (logM500 13.0-13.8 Msun/h)
median cylindrical f_gas at z~0 with each of the 30 SB35 astro parameters
(Spearman over the 256 Sobol points), cross-checked against the twobound
1P upper-minus-lower deltas. Direction-finding evidence for the decision
memo, NOT an inference: marginal rank correlations over a 30-dim design
ignore interactions (notably the SN-wind <-> BH-growth interplay that is
the physical mechanism), and the observable is single-bin/cylindrical/z=0.

Run: python analysis/paper3a/scripts/gate_param_directions.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

design = json.load(open('/mnt/ceph/users/mlee1/bind_sb35/design/design.json'))
astro_idx, astro_names = design['astro_param_indices'], design['astro_param_names']
root = Path('/mnt/ceph/users/mlee1/paper3/A/wp3_gate/operator_tables')  # fgas identical in v2


def group_fgas(path):
    d = np.load(path)
    logm500 = np.log10(d['m500_msunh'])
    sel = (logm500 >= 13.0) & (logm500 < 13.8)
    return np.median(d['fgas_cyl_r500'][sel]), d['params']


fg, th = zip(*(group_fgas(root / f'sb35_run_{i:04d}_snap096.npz') for i in range(256)))
fg, th = np.array(fg), np.array(th)
rho = np.array([stats.spearmanr(th[:, j], fg).statistic for j in astro_idx])

tb = json.load(open('/mnt/home/mlee1/ceph/bind_portable_twobound/design/design.json'))
deltas = {}
for k in range(0, 60, 2):
    lo, up = tb['runs'][k], tb['runs'][k + 1]
    assert lo['param'] == up['param'] and (lo['bound'], up['bound']) == ('lower', 'upper')
    f_lo, _ = group_fgas(root / f'twobound_run_{k:04d}_snap096.npz')
    f_up, _ = group_fgas(root / f'twobound_run_{k + 1:04d}_snap096.npz')
    deltas[lo['param']] = f_up - f_lo

print(f"{'param':<34}{'Sobol rho':>10}{'1P delta (up-lo)':>18}")
for j in np.argsort(-np.abs(rho)):
    n = astro_names[j]
    print(f"{n:<34}{rho[j]:>+10.3f}{deltas.get(n, float('nan')):>+18.4f}")
