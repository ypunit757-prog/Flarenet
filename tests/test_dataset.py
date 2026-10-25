import numpy as np

from flarenet.core.schema import WindowSet
from flarenet.pipelines.features.normalise import fit_normaliser, normalise
from flarenet.pipelines.features.split import block_split
from flarenet.pipelines.training.dataset import WindowDataset


def _ws(n=600, stride=60):
    rng = np.random.default_rng(0)
    return WindowSet(
        X=rng.normal(size=(n, 12, 90)).astype(np.float32),
        y_now=rng.integers(0, 5, n).astype(np.int64),
        y_fc=rng.integers(0, 2, n).astype(np.float32),
        t_end=np.arange(n, dtype=float) * stride + 89,
        missing_frac=np.zeros(n, dtype=np.float32),
        channels=[f"c{i}" for i in range(12)],
    )


def test_block_split_is_disjoint_and_complete():
    ws = _ws()
    sp = block_split(ws, block_days=1, seed=3)
    n = [len(sp[k]) for k in ("train", "val", "test")]
    assert sum(n) == len(ws) and n[0] > n[1] > 0 and n[2] > 0
    sets = [set(sp[k].t_end.tolist()) for k in ("train", "val", "test")]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])


def test_block_split_keeps_blocks_contiguous():
    sp = block_split(_ws(), block_days=1, seed=0)
    t0 = _ws().t_end[0]  # blocks are counted from the first window, not from midnight
    block = lambda ws: set(np.floor((ws.t_end - t0) / 1440).astype(int).tolist())  # noqa: E731
    assert not (block(sp["train"]) & block(sp["test"]))  # no day shared across splits


def test_block_split_is_deterministic_and_seed_dependent():
    a, b, c = (
        block_split(_ws(), 1, seed=1),
        block_split(_ws(), 1, seed=1),
        block_split(_ws(), 1, seed=2),
    )
    assert np.array_equal(a["test"].t_end, b["test"].t_end)
    assert not np.array_equal(a["test"].t_end, c["test"].t_end)


def test_normaliser_zero_mean_unit_std():
    X = _ws().X * 5 + 3
    mu, sg = fit_normaliser(X)
    Z = normalise(X, mu, sg)
    assert abs(float(Z.mean())) < 1e-3 and abs(float(Z.std()) - 1) < 1e-2
    assert mu.shape == (1, 12, 1) and Z.dtype == np.float32


def test_window_dataset_items_and_augmentation_only_when_asked():
    ws = _ws(50)
    plain = WindowDataset(ws.X, ws.y_now, ws.y_fc)
    assert np.array_equal(plain[0][0].numpy(), plain[0][0].numpy())
    x, y_n, y_f = plain[3]
    assert x.shape == (12, 90) and y_n.dtype == np.int64 or str(y_n.dtype) == "torch.int64"
    aug = WindowDataset(ws.X, ws.y_now, ws.y_fc, plain.mu, plain.sg, augment=True)
    assert not np.array_equal(aug[0][0].numpy(), plain[0][0].numpy())
