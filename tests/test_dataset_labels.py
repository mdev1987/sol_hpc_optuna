import polars as pl

from dataset_builder import DatasetBuilder
from simulator import SimulatorConfig


def _frame():
    mint_a = ["A"] * 20
    mint_b = ["B"] * 20
    ts = list(range(0, 200, 10)) * 2
    price_a = [1.0 + 0.1 * i for i in range(20)]
    price_b = [1.0] * 20
    return pl.DataFrame(
        {
            "mint": mint_a + mint_b,
            "timestamp": ts,
            "slot": list(range(40)),
            "price": price_a + price_b,
            "price_change_5": [0.0] * 40,
        }
    )


def _labels(df, config):
    builder = DatasetBuilder(config)
    return builder._compute_labels(df)


def test_labels_contain_both_classes():
    config = SimulatorConfig(take_profit=0.5, ttl_seconds=100)
    labels = _labels(_frame(), config)
    assert 1 in labels and 0 in labels


def test_balanced_build_equal_counts(tmp_path):
    config = SimulatorConfig(take_profit=0.5, ttl_seconds=100)
    out = tmp_path / "dataset.parquet"
    written = DatasetBuilder(config).build(_frame(), output=out, balanced=True)
    df = pl.read_parquet(out)
    assert written == len(df)
    labels = df["label"].to_list()
    assert labels.count(1) == labels.count(0) > 0


def test_unbalanced_build_keeps_all_rows(tmp_path):
    config = SimulatorConfig(take_profit=0.5, ttl_seconds=100)
    out = tmp_path / "dataset.parquet"
    written = DatasetBuilder(config).build(_frame(), output=out, balanced=False)
    assert written == 40
