import sys
sys.path.insert(0, 'src')
from pathlib import Path
from dataclasses import replace
import numpy as np
from analysis.evaluate import _load_assessment_data, _return_curves, _oracle_config, _fit_stop_controllers, _regret_at, fit_singlehalt_stop

packed_root = Path('/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k/mc_packed')
cache = '/scratch/gpfs/GRIFFITHS/hl4291/lmcos/ysagiv_xaba20k/materialized/validation_cache.pt'
episodes, z_by_ep, fit_idx, ev_idx = _load_assessment_data(packed_root, cache, 32, 15000, 0, load_action_gaps=True)
base_cfg = _oracle_config(packed_root)

for lam in [0.0015, 0.003, 0.01, 0.03]:
    config = replace(base_cfg, time_mode='linear', time_lambda=lam, maintenance_scale=0.0, maintenance_exponent=1.0)
    curves = _return_curves(episodes, config)
    fit_curves = [curves[i] for i in fit_idx]
    ev = [curves[i] for i in ev_idx]
    kf = fit_singlehalt_stop(fit_curves)
    kmax = max(len(c) for c in ev)
    meaningful = 1 < kf < kmax - 1
    ctrl = _fit_stop_controllers(episodes, z_by_ep, fit_idx, ev_idx, fit_curves, 32, 0)
    zt_stops = np.array(ctrl['zt']); ag_stops = np.array(ctrl['ag'])
    zt_regret = _regret_at(ev, ctrl['zt']).mean(); ag_regret = _regret_at(ev, ctrl['ag']).mean()
    print(f'lambda={lam}: kf={kf} ceiling={kmax} meaningful={meaningful}', flush=True)
    print(f'  z_t stops: mean={zt_stops.mean():.2f} median={np.median(zt_stops):.1f} frac_at_0={np.mean(zt_stops<=0):.3f} frac_at_ceiling={np.mean(zt_stops>=kmax-1):.3f}  regret={zt_regret:.4f}', flush=True)
    print(f'  ag stops:  mean={ag_stops.mean():.2f} median={np.median(ag_stops):.1f} frac_at_0={np.mean(ag_stops<=0):.3f} frac_at_ceiling={np.mean(ag_stops>=kmax-1):.3f}  regret={ag_regret:.4f}', flush=True)
print('DONE', flush=True)
