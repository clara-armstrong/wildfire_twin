# Wildfire Digital Twin

A digital twin for wildfire spread and response. It simulates a fire, observes
it through noisy coarse sensor readings, infers where the fire started and which
way the wind is blowing, and uses that inference to pick water drop locations.
Built on a wildfire-risk senior thesis
(https://github.com/pkrish11/Wildfire-Risk-Management) and its public reference
implementation.

## Pipeline

1. **Simulate** ([`solver.py`](wildfire_twin/solver.py)): a
   reaction-diffusion-advection PDE (thesis Eq. 3.3), integrated with RK4 on a
   304x304 grid over contiguous forest/shrub/grass fuel regions and two
   protected asset zones.
2. **Observe** ([`observation.py`](wildfire_twin/observation.py)): the fire is
   never seen directly, only through a coarse 16x16 noisy sensor reading. With
   synthetic data, that noise is what keeps the inverse problem from being
   trivial.
3. **Infer** ([`domain.py`](wildfire_twin/domain.py),
   [`inverse.py`](wildfire_twin/inverse.py)): the unknowns (ignition location,
   wind) are discretized into 588 hypotheses. The posterior over them is
   computed by enumeration rather than sampling, and sharpens as observations
   arrive.
4. **Decide** ([`priority.py`](wildfire_twin/priority.py),
   [`policy.py`](wildfire_twin/policy.py),
   [`actions.py`](wildfire_twin/actions.py)): a spatial priority index (thesis
   Eqs. 5.4-5.8) picks where to drop water, cut a firebreak, backburn, or
   evacuate, using the inferred risk map instead of a fixed assumption. Feeding
   it the posterior does not currently change the chosen cell while the fire is
   far from the assets, since siting is temperature-dominated in that regime.
5. **Render** ([`render.py`](wildfire_twin/render.py)): a 2x3 panel figure
   showing the real fire alongside the posterior and static prior risk maps.
6. **Validate** ([`validation.py`](wildfire_twin/validation.py)):
   identical-twin (OSSE) experiments that score posterior accuracy and tune the
   likelihood temperature until the 90% credible interval on asset damage
   covers truth about 90% of the time.

`physical_asset.py` wraps steps 1-2 and `digital_asset.py` wraps steps 3-4
behind the [`pgmtwin`](https://github.com/pgmtwin/pgmtwin) digital-twin
framework's interface.

## Running it

```bash
# pgmtwin is the digital-twin framework this builds on. It isn't on PyPI and
# isn't tracked in this repo, so clone and install it alongside:
git clone https://github.com/pgmtwin/pgmtwin.git
pip install -e ./pgmtwin

pip install -e .
PYTHONPATH=. pytest wildfire_twin/wildfiretwintest.py -q
```

```python
from wildfire_twin.config import WildfireConfig, ObservationConfig, LatentGridConfig
from wildfire_twin.domain import build_latent_domain
from wildfire_twin.inverse import build_forward_cache
from wildfire_twin.physical_asset import WildfirePhysicalAsset
from wildfire_twin.digital_asset import WildfireDigitalAsset

cfg = WildfireConfig()
obs_cfg = ObservationConfig()
domain = build_latent_domain(cfg, LatentGridConfig())
cache = build_forward_cache(cfg, domain, obs_cfg, progress=False)

phys = WildfirePhysicalAsset(cfg, obs_config=obs_cfg)
digi = WildfireDigitalAsset(cfg, domain, cache, obs_cfg)

n_steps = obs_cfg.n_observations * obs_cfg.observe_every
for step in range(1, n_steps + 1):
    phys.update(action=None)
    if step % obs_cfg.observe_every == 0:
        digi.get_assimilation(phys.get_observations())
```

## Examples

```bash
python examples/00_test_render.py
```

This runs the solver ([`solver.py`](wildfire_twin/solver.py)) directly, with no
inference and no forward cache, so it starts right away without building
anything first. It draws the fire itself rather than the twin's belief about it.

It runs on a 128x128 grid for speed, where `q=1.0` would leave the combustion
front narrower than one cell, so it raises `q` to 10.0 instead. The package
default resolves the front by refining the grid rather than by changing `q`.
See the note on `grid_size` in [`config.py`](wildfire_twin/config.py).

```bash
python examples/01_domain_and_evolution.py
```

Two survey figures from [`render.py`](wildfire_twin/render.py): the domain
before ignition (fuel types, ignition threshold, wind) and the fire at four
timesteps. Solver-only on the same 128x128 / `q=10.0` pairing, so the evolution
figure's top row is ground-truth burn status; with a digital asset it becomes
the posterior risk map, as sketched at the bottom of the script.

## Status

62 tests pass. Observations can only come from the simulator itself; there is no
real-sensor ingestion path yet. The decision policy is hand-tuned rules rather
than learned, and some smaller fidelity gaps against the thesis are still open
(see the docstrings and TODOs, for instance `config.py`'s note on `q`).
