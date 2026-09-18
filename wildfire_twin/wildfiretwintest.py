"""Tests for the wildfire digital twin.

Run with:  PYTHONPATH=. pytest wildfire_twin/wildfiretwintest.py -q

"""
import numpy as np
import pytest

from wildfire_twin.config import (
    FUEL_TYPES,
    STATE_FIELDS,
    LatentGridConfig,
    ObservationConfig,
    WildfireConfig,
)
from wildfire_twin.digital_asset import WildfireDigitalAsset
from wildfire_twin.domain import VAR_NAMES, build_latent_domain
from wildfire_twin.inverse import (
    ForwardCache,
    InverseSolver,
    _with_progress,
    build_forward_cache
)
from wildfire_twin.observation import (
    coarsen,
    observation_dim,
    observe,
    pack_state,
    unpack_state,
)
from wildfire_twin.physical_asset import WildfirePhysicalAsset
from wildfire_twin.actions import DoNothingAction, WaterDropAction
from wildfire_twin.priority import optimal_drop, priority_index
from wildfire_twin.solver import (
    BURNED,
    BURNING,
    UNBURNED,
    Latent,
    Wildfire,
    build_fuel_map,
)


# Fixtures: small and fast

@pytest.fixture(scope="module")
def cfg():
    return WildfireConfig(grid_size=32, dt=0.05, q=10.0) # "Uses" q


@pytest.fixture(scope="module")
def obs_cfg():
    return ObservationConfig(coarse_size=8, observe_every=20, n_observations=3, noise_std=0.05)


@pytest.fixture(scope="module")
def domain(cfg):
    return build_latent_domain(
        cfg, LatentGridConfig(n_x0=3, n_y0=3, wind_speeds=(1.0,), wind_directions_deg=(0.0, 45.0))
    )


@pytest.fixture(scope="module")
def cache(cfg, domain, obs_cfg):
    return build_forward_cache(cfg, domain, obs_cfg, risk_steps=[60], progress=False)


# Tests for the solver

class TestSolver:
    def test_fuel_map_matches_fuel_types(self, cfg):
        fuel_type, beta, u_pc, asset = build_fuel_map(cfg)
        assert fuel_type.shape == beta.shape == u_pc.shape == (cfg.grid_size,) * 2
        for k, props in FUEL_TYPES.items():
            mask = fuel_type == k
            if mask.any():
                assert np.allclose(beta[mask], props.fuel_fraction)
                assert np.allclose(u_pc[mask], props.ignition_threshold)

    def test_fuel_regions_are_contiguous(self, cfg):
        fuel_type, _, _, _ = build_fuel_map(cfg)
        # A random mosaic would disagree with its neighbour ~3/4 of the time.
        same = (fuel_type[1:, :] == fuel_type[:-1, :]).mean()
        assert same > 0.9

    def test_assets_are_nonempty_and_bounded(self, cfg):
        _, _, _, asset = build_fuel_map(cfg)
        assert asset.max() == 1.0 and asset.min() == 0.0
        assert 0 < asset.mean() < 0.5

    def test_ignition_seeds_a_fire(self, cfg):
        fire = Wildfire(cfg, Latent(-35, -35, 1.0, 45.0))
        assert (fire.burn_status == BURNING).sum() > 0

    def test_state_stays_bounded(self, cfg):
        fire = Wildfire(cfg, Latent(-35, -35, 1.5, 45.0))
        for _ in range(200):
            fire.step()
            assert np.isfinite(fire.u).all()
            assert fire.u.min() >= 0.0 and fire.u.max() <= 8.0
            assert fire.beta.min() >= 0.0 and fire.beta.max() <= 1.0
        assert set(np.unique(fire.burn_status)) <= {UNBURNED, BURNING, BURNED}

    def test_burn_status_is_monotone(self, cfg):
        # Cells may only advance 0 -> 1 -> 2 (unburned to burning to burned), never regressing.
        fire = Wildfire(cfg, Latent(-35, -35, 1.0, 45.0))
        prev = fire.burn_status.copy()
        for _ in range(100):
            fire.step()
            assert (fire.burn_status >= prev).all()
            prev = fire.burn_status.copy()

    def test_fuel_is_non_increasing(self, cfg):
        fire = Wildfire(cfg, Latent(-35, -35, 1.0, 45.0))
        prev = fire.beta.copy()
        for _ in range(100):
            fire.step()
            assert (fire.beta <= prev + 1e-6).all()
            prev = fire.beta.copy()

    def test_deterministic(self, cfg):
        a = Wildfire(cfg, Latent(-35, -35, 1.0, 45.0))
        b = Wildfire(cfg, Latent(-35, -35, 1.0, 45.0))
        for _ in range(50):
            a.step()
            b.step()
        assert np.array_equal(a.u, b.u)

    def test_wind_direction_steers_the_fire(self, cfg):
        # Sanity: the fire spreads downwind, not upwind.
        east = Wildfire(cfg, Latent(-35, -35, 1.5, 0.0))
        north = Wildfire(cfg, Latent(-35, -35, 1.5, 90.0))
        for _ in range(300):
            east.step()
            north.step()
        ii, jj = np.indices(east.u.shape)
        e_centre = ii[east.burn_status > 0].mean()
        n_centre = jj[north.burn_status > 0].mean()
        e_ref = ii[north.burn_status > 0].mean()
        n_ref = jj[east.burn_status > 0].mean()
        assert e_centre > e_ref
        assert n_centre > n_ref

    def test_front_width_criterion(self):
        # q=1 is both the class default and thesis Table 5.1's literal value.
        # At the thesis's 128 cells it leaves the front narrower than one
        # cell; the shipped grid_size refines until delta_over_dx clears 1,
        # which is the property that has to hold.
        assert WildfireConfig(grid_size=128).q == 1.0
        assert WildfireConfig(grid_size=128).resolution_report()["delta_over_dx"] < 1.0
        assert WildfireConfig().resolution_report()["delta_over_dx"] > 1.0

    def test_cfl_is_satisfied(self, cfg):
        r = cfg.cfl_report()
        assert r["diffusion_ratio"] < 1.0 and r["advection_ratio"] < 1.0
        # The shipped defaults too: the diffusion limit falls as dx^2, so
        # refining grid_size without lowering dt would break them.
        d = WildfireConfig().cfl_report()
        assert d["diffusion_ratio"] < 1.0 and d["advection_ratio"] < 1.0

    def test_boundary_values_are_pinned(self, cfg):
        # apply_boundary_conditions zeroes du/dt and dbeta/dt on all four edges, so
        # edge cells must hold their initial values exactly however hard the
        # fire pushes against them. Ignites in the corner so it does push.
        fire = Wildfire(cfg, Latent(-100.0, -100.0, 2.0, 225.0))
        u0, beta0 = fire.u.copy(), fire.beta.copy()
        for _ in range(50):
            fire.step()
        for edge in (np.s_[0, :], np.s_[-1, :], np.s_[:, 0], np.s_[:, -1]):
            assert np.array_equal(fire.u[edge], u0[edge])
            assert np.array_equal(fire.beta[edge], beta0[edge])

    def test_interior_still_evolves_under_boundary_conditions(self, cfg):
        # Guards the obvious over-correction: pinning the edges must not
        # freeze the whole field.
        fire = Wildfire(cfg, Latent(-35, -35, 1.0, 45.0))
        u0 = fire.u.copy()
        for _ in range(20):
            fire.step()
        assert not np.allclose(fire.u[1:-1, 1:-1], u0[1:-1, 1:-1])


# Tests for the observation operator

class TestObservation:
    def test_coarsen_preserves_mean(self):
        f = np.random.default_rng(0).random((32, 32))
        assert coarsen(f, 8).shape == (8, 8)
        assert np.isclose(coarsen(f, 8).mean(), f.mean())

    def test_coarsen_rejects_indivisible(self):
        with pytest.raises(ValueError, match="divisible"):
            coarsen(np.zeros((32, 32)), 7)

    def test_pack_unpack_roundtrip(self, cfg):
        # Guards against pack/unpack disagreeing on field order.
        fire = Wildfire(cfg, Latent(-35, -35, 1.0, 45.0))
        for _ in range(10):
            fire.step()
        state = fire.state_dict()
        back = unpack_state(pack_state(state, cfg.grid_size), cfg.grid_size)
        for k, v in state.items():
            assert np.allclose(np.asarray(v, dtype=float), np.asarray(back[k], dtype=float))

    def test_pack_rejects_wrong_shape(self, cfg):
        bad = {k: np.zeros((4, 4)) for k in ("temperature", "fuel_fraction",
                                             "ignition_threshold", "burn_status",
                                             "asset_importance")}
        with pytest.raises(ValueError, match="expected"):
            pack_state(bad, cfg.grid_size)

    def test_pack_rejects_missing_field(self, cfg):
        with pytest.raises(KeyError, match="missing required field"):
            pack_state({"temperature": np.zeros((cfg.grid_size,) * 2)}, cfg.grid_size)

    def test_observation_dimension(self, cfg, obs_cfg):
        fire = Wildfire(cfg, Latent(-35, -35, 1.0, 45.0))
        y = observe(fire.state_dict(), obs_cfg, noiseless=True)
        assert y.shape == (observation_dim(obs_cfg),)

    def test_observation_hides_latent_fields(self, cfg, obs_cfg):
        # The whole point: an observation must not contain what we infer.
        assert "ignition_threshold" not in obs_cfg.channels
        assert "asset_importance" not in obs_cfg.channels
        assert "fuel_fraction" not in obs_cfg.channels

    def test_noise_is_applied(self, cfg, obs_cfg):
        fire = Wildfire(cfg, Latent(-35, -35, 1.0, 45.0))
        rng = np.random.default_rng(0)
        clean = observe(fire.state_dict(), obs_cfg, noiseless=True)
        noisy = observe(fire.state_dict(), obs_cfg, rng=rng)
        assert not np.allclose(clean, noisy)
        assert abs(float(np.std(noisy - clean)) - obs_cfg.noise_std) < 0.02


# Tests for latent domain

class TestDomain:
    def test_size_and_shape(self, domain):
        assert len(domain) == int(np.prod(domain.shape)) == domain.table.shape[0]

    def test_index_value_roundtrip(self, domain):
        for i in range(len(domain)):
            assert domain.values2index(domain.index2values(i)) == i

    def test_wind_direction_wraps_circularly(self, cfg):
        d = build_latent_domain(cfg, LatentGridConfig(
            n_x0=1, n_y0=1, wind_speeds=(1.0,), wind_directions_deg=(0.0, 90.0, 350.0)))
        i = d.values2index([d.var2values["x0"][0], d.var2values["y0"][0], 1.0, 352.0])
        assert d.index2values(i)[3] == 350.0
        j = d.values2index([d.var2values["x0"][0], d.var2values["y0"][0], 1.0, 5.0])
        assert d.index2values(j)[3] == 0.0

    def test_values2index_rejects_bad_length(self, domain):
        with pytest.raises(ValueError, match="length"):
            domain.values2index([1.0, 2.0])

    def test_prior_is_normalised(self, domain):
        assert np.isclose(domain.prior().sum(), 1.0)

    def test_marginals_normalise(self, domain):
        w = np.random.default_rng(0).random(len(domain))
        w /= w.sum()
        for name in VAR_NAMES:
            assert np.isclose(domain.marginal(w, name).sum(), 1.0)

    def test_marginal_rejects_unknown_variable(self, domain):
        with pytest.raises(KeyError):
            domain.marginal(domain.prior(), "not_a_variable")

    def test_ignition_range_follows_eq_5_3(self, domain, cfg):
        x = domain.var2values["x0"]
        assert np.isclose(x.min(), cfg.ignition_center_x - cfg.ignition_sigma)
        assert np.isclose(x.max(), cfg.ignition_center_x + cfg.ignition_sigma)


# Tests for Inverse solver

class TestInverse:
    def test_cache_shapes(self, cache, domain, obs_cfg):
        assert cache.obs.shape == (len(domain), obs_cfg.n_observations,
                                   observation_dim(obs_cfg))
        assert cache.burn.shape[0] == len(domain)

    def test_cache_roundtrip(self, cache, tmp_path):
        # save/load enumerate dataclass fields rather than naming them, so
        # every field has to survive -- including the two that load() has to
        # cast back to int, which np.load returns as 0-d arrays.
        p = tmp_path / "c.npz"
        cache.save(p)
        back = ForwardCache.load(p)
        for f in ("obs", "burn", "asset_burnt", "burnt_fraction", "obs_steps", "risk_steps"):
            assert np.array_equal(getattr(cache, f), getattr(back, f)), f
        assert back.grid_size == cache.grid_size and back.coarse_size == cache.coarse_size
        assert isinstance(back.grid_size, int) and isinstance(back.coarse_size, int)

    def test_parallel_cache_matches_serial(self, cfg, domain, obs_cfg, cache):
        # imap_unordered returns rows out of order, so the scatter back into
        # the arrays has to go by index, not by arrival.
        par = build_forward_cache(cfg, domain, obs_cfg, risk_steps=[60],
                                  n_workers=2, progress=False)
        for f in ("obs", "burn", "asset_burnt", "burnt_fraction"):
            assert np.array_equal(getattr(par, f), getattr(cache, f)), f

    def test_progress_wrapper_is_transparent(self, capsys):
        assert list(_with_progress(iter(range(120)), 120, False)) == list(range(120))
        assert capsys.readouterr().out == ""
        assert list(_with_progress(iter(range(120)), 120, True)) == list(range(120))
        assert capsys.readouterr().out.count("forward cache") == 2

    def test_posterior_is_a_distribution(self, domain, cache, obs_cfg):
        s = InverseSolver(domain, cache, obs_cfg)
        s.update(cache.obs[0, 0], 0)
        p = s.posterior
        assert np.isclose(p.sum(), 1.0) and (p >= 0).all()

    def test_prior_before_any_data(self, domain, cache, obs_cfg):
        s = InverseSolver(domain, cache, obs_cfg)
        assert np.allclose(s.posterior, domain.prior())

    def test_recovers_its_own_forward_model(self, domain, cache, obs_cfg):
        # Noiseless self-consistency: the MAP must be the generating latent.
        s = InverseSolver(domain, cache, obs_cfg, model_error_std=1e-6, temperature=1.0)
        for i in range(len(domain)):
            s.assimilate_sequence([cache.obs[i, k] for k in range(cache.obs.shape[1])])
            assert int(np.argmax(s.posterior)) == i

    def test_ess_falls_as_data_arrives(self, domain, cache, obs_cfg):
        s = InverseSolver(domain, cache, obs_cfg, model_error_std=1e-3, temperature=1.0)
        before = s.effective_sample_size
        s.assimilate_sequence([cache.obs[0, k] for k in range(cache.obs.shape[1])])
        assert s.effective_sample_size < before

    def test_rejects_too_many_observations(self, domain, cache, obs_cfg):
        s = InverseSolver(domain, cache, obs_cfg)
        with pytest.raises(IndexError):
            s.update(cache.obs[0, 0], cache.obs.shape[1])

    def test_discrepancy_shape_validated(self, domain, cache, obs_cfg):
        with pytest.raises(ValueError, match="discrepancy"):
            InverseSolver(domain, cache, obs_cfg, discrepancy=np.zeros((2, 3)))

    def test_discrepancy_shifts_the_posterior(self, domain, cache, obs_cfg):
        base = InverseSolver(domain, cache, obs_cfg, model_error_std=1e-3, temperature=1.0)
        base.update(cache.obs[0, 0], 0)
        delta = np.full(cache.obs.shape[1:], 0.2, dtype=np.float32)
        bias = InverseSolver(domain, cache, obs_cfg, model_error_std=1e-3,
                             temperature=1.0, discrepancy=delta)
        bias.update(cache.obs[0, 0], 0)
        assert not np.allclose(base.posterior, bias.posterior)

    def test_risk_map_is_a_probability(self, domain, cache, obs_cfg):
        # Insists risk map follows probability axioms
        s = InverseSolver(domain, cache, obs_cfg)
        R = s.risk_map(int(cache.risk_steps[0]))
        assert R.shape == (cache.coarse_size,) * 2
        assert R.min() >= 0.0 and R.max() <= 1.0

    def test_prior_risk_map_equals_uniform_average(self, domain, cache, obs_cfg):
        s = InverseSolver(domain, cache, obs_cfg)
        step = int(cache.risk_steps[0])
        k = int(np.where(cache.risk_steps == step)[0][0])
        assert np.allclose(s.prior_risk_map(step), cache.burn[:, k].mean(axis=0))

    def test_unknown_risk_step_raises(self, domain, cache, obs_cfg):
        s = InverseSolver(domain, cache, obs_cfg)
        with pytest.raises(KeyError, match="not a cached risk step"):
            s.risk_map(999999)

    def test_credible_interval_brackets_the_mean(self, domain, cache, obs_cfg):
        s = InverseSolver(domain, cache, obs_cfg)
        step = int(cache.risk_steps[0])
        lo, hi = s.credible_interval(step)
        mean, _ = s.expected_asset_damage(step)
        assert lo <= mean <= hi


# Tests for priority index

class TestPriority:
    def test_shapes(self):
        rng = np.random.default_rng(0)
        risk = rng.random((8, 8))
        temp = rng.random((8, 8))
        asset = np.zeros((8, 8))
        asset[6, 1] = asset[1, 6] = 1.0
        out = priority_index(risk, temp, asset, (1.0, 1.0))
        assert out["priority"].shape == (8, 8)
        assert out["gradient"].shape == (8, 8, 2)

    def test_no_assets_gives_zero_priority(self):
        rng = np.random.default_rng(0)
        out = priority_index(rng.random((8, 8)), rng.random((8, 8)),
                             np.zeros((8, 8)), (1.0, 0.0))
        assert np.allclose(out["priority"], 0.0)

    def test_optimal_drop_is_in_bounds(self):
        rng = np.random.default_rng(1)
        asset = np.zeros((8, 8))
        asset[6, 1] = 1.0
        d = optimal_drop(rng.random((8, 8)), rng.random((8, 8)), asset, (1.0, 1.0))
        i, j = d["location"]
        assert 0 <= i < 8 and 0 <= j < 8
        assert 0.0 <= d["angle_deg"] < 360.0

    def test_priority_is_finite(self):
        rng = np.random.default_rng(2)
        asset = np.zeros((8, 8))
        asset[4, 4] = 1.0   # an asset cell sits at its own location: d = 0
        out = priority_index(rng.random((8, 8)), rng.random((8, 8)), asset, (1.0, 0.0))
        assert np.isfinite(out["priority"]).all()


# Tests for digital asset

class TestDigitalAsset:
    def _asset(self, cfg, domain, cache, obs_cfg):
        return WildfireDigitalAsset(cfg, domain, cache, obs_cfg)

    def test_assimilation_returns_latent_estimate(self, cfg, domain, cache, obs_cfg):
        da = self._asset(cfg, domain, cache, obs_cfg)
        est = da.get_assimilation(cache.obs[0, 0])
        assert est.shape == (4,)
        assert np.isfinite(est).all()

    def test_distribution_is_over_the_latent_domain(self, cfg, domain, cache, obs_cfg):
        da = self._asset(cfg, domain, cache, obs_cfg)
        d = da.get_assimilation_distribution(cache.obs[0, 0])
        assert d.shape == (len(domain),)
        assert np.isclose(d.sum(), 1.0)

    def test_bad_observation_raises_rather_than_defaulting(self, cfg, domain, cache, obs_cfg):
        # A wrong-length observation must fail loudly, not default silently.
        da = self._asset(cfg, domain, cache, obs_cfg)
        with pytest.raises(ValueError, match="expected"):
            da.get_assimilation(np.zeros(7))

    def test_history_is_recorded(self, cfg, domain, cache, obs_cfg):
        da = self._asset(cfg, domain, cache, obs_cfg)
        da.get_assimilation(cache.obs[0, :2])
        assert len(da.observation_history) == 2 == len(da.posterior_history)

    def test_reset_restores_the_prior(self, cfg, domain, cache, obs_cfg):
        da = self._asset(cfg, domain, cache, obs_cfg)
        da.get_assimilation(cache.obs[0, 0])
        da.reset()
        assert np.allclose(da.posterior, domain.prior())
        assert da.observation_history == []

    def test_qois_report_uncertainty(self, cfg, domain, cache, obs_cfg): # USES Q
        da = self._asset(cfg, domain, cache, obs_cfg)
        q = da.get_qois(int(cache.risk_steps[0]))
        for key in ("asset_damage_mean", "asset_damage_std",
                    "asset_damage_lo90", "asset_damage_hi90"):
            assert key in q
        assert q["asset_damage_lo90"] <= q["asset_damage_mean"] <= q["asset_damage_hi90"]

    def test_recommend_drop_runs_both_ways(self, cfg, domain, cache, obs_cfg):
        da = self._asset(cfg, domain, cache, obs_cfg)
        da.get_assimilation(cache.obs[0, 0])
        C = obs_cfg.coarse_size
        _, _, _, asset_fine = build_fuel_map(cfg)
        asset = coarsen(asset_fine, C)
        step = int(cache.risk_steps[0])
        for use_posterior in (True, False):
            d = da.recommend_drop(step, np.zeros((C, C)), asset, use_posterior=use_posterior)
            assert 0 <= d["location"][0] < C


# Tests for physical asset

class TestPhysicalAsset:
    def test_wind_is_a_single_scalar(self, cfg, obs_cfg):
        # Wind stays a single scalar pair: domain.LatentDomain's
        # uniform-wind theta cannot represent a per-cell field.
        phys = WildfirePhysicalAsset(cfg, obs_config=obs_cfg, rng=np.random.default_rng(0))
        assert isinstance(phys.wind_speed, float)
        assert isinstance(phys.wind_direction, float)
        assert 0.0 <= phys.wind_speed <= 3.0
        assert 0.0 <= phys.wind_direction < 360.0

    def test_fuel_regions_are_contiguous(self, cfg, obs_cfg):
        # Fuel regions are contiguous blocks, not an i.i.d. per-cell mosaic
        # (see solver.py's equivalent test for the same criterion).
        phys = WildfirePhysicalAsset(cfg, obs_config=obs_cfg, rng=np.random.default_rng(0))
        same = (phys.fuel_type[1:, :] == phys.fuel_type[:-1, :]).mean()
        assert same > 0.9

    def test_state_dict_has_the_expected_fields(self, cfg, obs_cfg):
        phys = WildfirePhysicalAsset(cfg, obs_config=obs_cfg, rng=np.random.default_rng(0))
        for name in STATE_FIELDS:
            assert phys.state_dict[name].shape == (cfg.grid_size, cfg.grid_size)

    def test_update_runs_without_crashing(self, cfg, obs_cfg):
        phys = WildfirePhysicalAsset(cfg, obs_config=obs_cfg, rng=np.random.default_rng(0))
        action = DoNothingAction(None)
        for _ in range(50):
            phys.update(action)
            assert np.isfinite(phys.state_dict["temperature"]).all()
            assert set(np.unique(phys.state_dict["burn_status"])) <= {0, 1, 2}

    def test_get_observations_shape(self, cfg, obs_cfg):
        phys = WildfirePhysicalAsset(cfg, obs_config=obs_cfg, rng=np.random.default_rng(0))
        obs = phys.get_observations(n_observations=2)
        assert obs.shape == (2, observation_dim(obs_cfg))

    def test_matches_solver_when_the_fire_reaches_the_boundary(self):
        # The "true" fire and the twin's forward model must apply the same
        # boundary condition. If only one of them zeroes the edge ring, the
        # two agree to ~1e-6 while the fire stays interior and then diverge
        # by the full temperature range (~7.4 of a 0-8 scale) once it
        # touches an edge. Ignites in the corner, with the sampling sigmas
        # pinned, to reach the edge fast.
        cfg = WildfireConfig(
            grid_size=64, q=10.0, dt=0.02,
            ignition_center_x=-85.0, ignition_center_y=-85.0,
            ignition_sigma=0.0, wind_speed_sigma=0.0,
            wind_direction_sigma=0.0, wind_direction_deg=225.0,
        )
        phys = WildfirePhysicalAsset(cfg, rng=np.random.default_rng(0))
        fire = Wildfire(cfg, Latent(
            x0=-85.0, y0=-85.0,
            wind_speed=phys.wind_speed, wind_direction_deg=phys.wind_direction,
        ))
        # Same initial state, so any drift is the dynamics, not the setup.
        fire.u = phys.state_dict["temperature"].copy()
        fire.beta = phys.state_dict["fuel_fraction"].copy()
        fire.burn_status = phys.state_dict["burn_status"].copy()
        fire.u_pc = phys.state_dict["ignition_threshold"].copy()

        for _ in range(250):
            fire.step()
            phys.update(DoNothingAction(None))

        assert np.abs(fire.u - phys.state_dict["temperature"]).max() < 1e-4
        assert np.abs(fire.beta - phys.state_dict["fuel_fraction"]).max() < 1e-4
        # The fire really did reach the edge -- otherwise this proves nothing.
        # Checked on the first interior row: the edge ring itself is held at
        # its initial value, which here stays below the ignition threshold,
        # so burn_status never flips there by construction.
        assert phys.state_dict["burn_status"][1, :].any()

    def test_water_drop_action_cools_its_target(self, cfg, obs_cfg):
        phys = WildfirePhysicalAsset(cfg, obs_config=obs_cfg, rng=np.random.default_rng(0))
        for _ in range(20):
            phys.update(DoNothingAction(None))
        i, j = np.unravel_index(
            np.argmax(phys.state_dict["temperature"]), phys.state_dict["temperature"].shape
        )
        before = phys.state_dict["temperature"][i, j]
        phys.update(WaterDropAction((int(i), int(j)), None, intensity=1.0, radius=2))
        assert phys.state_dict["temperature"][i, j] <= before