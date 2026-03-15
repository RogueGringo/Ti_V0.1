"""End-to-end integration test for the full ATFT pipeline."""
import numpy as np
import pytest

from atft.analysis.evolution_curves import EvolutionCurveComputer
from atft.analysis.statistical_tests import StatisticalValidator
from atft.analysis.waypoint_extractor import WaypointExtractor
from atft.core.types import CurveType
from atft.feature_maps.identity import IdentityMap
from atft.feature_maps.spectral_unfolding import SpectralUnfolding
from atft.sources.gue import GUESource
from atft.sources.poisson import PoissonSource
from atft.topology.analytical_h0 import AnalyticalH0


N_POINTS = 50
ENSEMBLE_SIZE = 20
K_WAYPOINTS = 2


class TestFullPipeline:
    def test_pipeline_runs_without_errors(self):
        ph = AnalyticalH0()
        curves_computer = EvolutionCurveComputer(n_steps=100)
        wp_extractor = WaypointExtractor(k_waypoints=K_WAYPOINTS)
        validator = StatisticalValidator(confidence_level=0.99)

        gue_src = GUESource(seed=42)
        gue_batch = gue_src.generate_batch(N_POINTS, ENSEMBLE_SIZE)
        gue_unfolded = SpectralUnfolding(method="rank").transform_batch(gue_batch)
        gue_pds = ph.compute_batch(gue_unfolded)
        gue_curves = [curves_computer.compute(pd) for pd in gue_pds]
        gue_sigs = [wp_extractor.extract(pd, c) for pd, c in zip(gue_pds, gue_curves)]

        poisson_src = PoissonSource(seed=43)
        poisson_cloud = poisson_src.generate(N_POINTS)
        poisson_unfolded = IdentityMap().transform(poisson_cloud)
        poisson_pd = ph.compute(poisson_unfolded)
        poisson_curves = curves_computer.compute(poisson_pd)
        poisson_sig = wp_extractor.extract(poisson_pd, poisson_curves)

        validator.fit_ensemble(gue_sigs, gue_curves)

        gue_result = validator.validate(gue_sigs[0], gue_curves[0])
        assert gue_result.p_value > 0.01

        poisson_result = validator.validate(poisson_sig, poisson_curves)
        assert poisson_result.mahalanobis_distance > gue_result.mahalanobis_distance

    def test_all_signatures_same_dimension(self):
        ph = AnalyticalH0()
        curves_computer = EvolutionCurveComputer(n_steps=50)
        wp_extractor = WaypointExtractor(k_waypoints=K_WAYPOINTS)

        gue_src = GUESource(seed=42)
        batch = gue_src.generate_batch(N_POINTS, 10)
        unfolded = SpectralUnfolding(method="rank").transform_batch(batch)
        pds = ph.compute_batch(unfolded)
        curveset = [curves_computer.compute(pd) for pd in pds]
        sigs = [wp_extractor.extract(pd, c) for pd, c in zip(pds, curveset)]

        dims = [s.vector_dimension for s in sigs]
        assert len(set(dims)) == 1
        assert dims[0] == 2 * K_WAYPOINTS + 3

    def test_betti_curve_properties(self):
        gue_src = GUESource(seed=42)
        cloud = gue_src.generate(N_POINTS)
        unfolded = SpectralUnfolding(method="rank").transform(cloud)
        pd = AnalyticalH0().compute(unfolded)
        curves = EvolutionCurveComputer(n_steps=200).compute(pd)
        betti = curves.betti[0]

        assert betti.values[0] == N_POINTS
        assert betti.values[-1] == 1.0
        assert np.all(np.diff(betti.values) <= 0)
