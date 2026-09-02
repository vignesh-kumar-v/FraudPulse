import pandas as pd
import pytest

from fraudpulse.data.synthetic import make_events


@pytest.fixture(scope="session")
def events() -> pd.DataFrame:
    return make_events(n_cards=120, n_events=4_000, seed=11)
