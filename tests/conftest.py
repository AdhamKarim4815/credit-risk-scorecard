import numpy as np
import pytest

from credit_scorecard import generate, pipeline


@pytest.fixture(scope="session")
def small_df():
    """A small synthetic portfolio WITH the planted data quality problems."""
    return generate.generate_loans(n=6000, seed=11)


@pytest.fixture(scope="session")
def mini_run(tmp_path_factory):
    """One end-to-end run on a small sample, shared by the slower tests."""
    cfg = pipeline.Config(workdir=tmp_path_factory.mktemp("mini"), n=8000, seed=5)
    results = pipeline.run_all(cfg, log=lambda *_: None)
    return cfg, results


@pytest.fixture
def rng():
    return np.random.default_rng(123)
