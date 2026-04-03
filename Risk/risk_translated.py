"""
Translated from selected scripts in Py14aK/Py14aK/Risk.

Included translations:
- BootStrap                  -> logistic_bootstrap(...)
- Fair coins                 -> fair_coin_markov(...)
- Moving averages            -> compute_moving_averages(...)
- Moving Block Bootstrap     -> simple_block_bootstrap_autoreg(...), moving_block_bootstrap_autoreg(...)
- path_finding_with_hash     -> reconstruct_paths(...), compute_path_lengths(...)

Notes:
- This is a pragmatic Python translation, not a byte-for-byte port.
- SAS PROC-specific features (stepwise logistic selection, ODS output, hash object syntax)
  are replaced by Python library equivalents.
- Some original scripts referenced external SAS macro variables or datasets that were not
  included in the fetched files. Those are parameterized here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from statsmodels.tsa.ar_model import AutoReg


# ---------------------------------------------------------------------------
# BootStrap
# ---------------------------------------------------------------------------

def logistic_bootstrap(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    n_bootstrap: int = 1000,
    random_state: int = 4322426,
    solver: str = "liblinear",
) -> pd.DataFrame:
    """
    Python translation of the SAS BootStrap script.

    Original SAS behavior:
    - resample rows with replacement many times
    - fit logistic regression on each sample
    - collect coefficient estimates

    Parameters
    ----------
    df : DataFrame
        Input data.
    target_col : str
        Binary target column.
    feature_cols : sequence of str
        Predictor columns.
    n_bootstrap : int
        Number of bootstrap resamples.
    random_state : int
        Random seed.
    solver : str
        sklearn LogisticRegression solver.

    Returns
    -------
    DataFrame with one row per bootstrap sample and coefficient summaries.
    """
    rng = np.random.default_rng(random_state)
    X = df.loc[:, feature_cols].to_numpy()
    y = df[target_col].to_numpy()

    rows = []
    for sample_id in range(1, n_bootstrap + 1):
        idx = rng.integers(0, len(df), size=len(df))
        Xb = X[idx]
        yb = y[idx]

        model = LogisticRegression(
            penalty=None,
            solver=solver,
            max_iter=1000,
        )
        model.fit(Xb, yb)

        row = {"sample_id": sample_id, "intercept": float(model.intercept_[0])}
        row.update({name: float(coef) for name, coef in zip(feature_cols, model.coef_[0])})
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fair coins
# ---------------------------------------------------------------------------

def fair_coin_markov(
    transition_matrix: np.ndarray | None = None,
    initial_state: np.ndarray | None = None,
    num_tosses: int = 30,
) -> pd.DataFrame:
    """
    Python translation of the 'Fair coins' SAS/IML script.

    The default matrix tracks progress toward a target pattern with a
    4-state Markov chain.
    """
    if transition_matrix is None:
        transition_matrix = np.array(
            [
                [0.5, 0.5, 0.0, 0.0],   # Null
                [0.0, 0.5, 0.5, 0.0],   # H
                [0.5, 0.0, 0.0, 0.5],   # HT
                [0.0, 0.0, 0.0, 1.0],   # HTH
            ],
            dtype=float,
        )
    if initial_state is None:
        initial_state = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    states = [initial_state]
    current = initial_state.copy()
    for _ in range(num_tosses):
        current = current @ transition_matrix
        states.append(current.copy())

    out = pd.DataFrame(states, columns=[f"state_{i+1}" for i in range(transition_matrix.shape[0])])
    out.insert(0, "iteration", np.arange(len(out)))
    return out


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def compute_moving_averages(
    df: pd.DataFrame,
    id_col: str = "id",
    date_col: str = "date_n",
    value_col: str = "close",
    span: int = 5,
    ewma_alpha: float = 0.37,
    weighted_window: Sequence[int] | None = None,
) -> pd.DataFrame:
    """
    Python translation of the 'Moving averages' SAS script.

    Produces simple moving average (MA), weighted moving average (WMA),
    and exponentially weighted moving average (EWMA) by entity.
    """
    if weighted_window is None:
        weighted_window = [1, 2, 3, 5, 7, 9, 11, 13, 17, 19, 23, 29, 37, 43, 71, 137, 143, 419]

    df = df.sort_values([id_col, date_col]).copy()

    def _wma(series: pd.Series) -> pd.Series:
        w = np.array(weighted_window, dtype=float)
        w = w / w.sum()

        def _rolling_weighted(arr: np.ndarray) -> float:
            local_w = w[-len(arr):]
            local_w = local_w / local_w.sum()
            return float(np.dot(arr, local_w))

        return series.rolling(window=len(weighted_window), min_periods=1).apply(_rolling_weighted, raw=True)

    df["MA"] = df.groupby(id_col)[value_col].transform(lambda s: s.rolling(span, min_periods=1).mean())
    df["WMA"] = df.groupby(id_col)[value_col].transform(_wma)
    df["EWMA"] = df.groupby(id_col)[value_col].transform(lambda s: s.ewm(alpha=ewma_alpha, adjust=False).mean())
    return df


# ---------------------------------------------------------------------------
# Time-series block bootstrap
# ---------------------------------------------------------------------------

@dataclass
class AutoRegBootstrapResult:
    bootstrapped_series: pd.DataFrame
    parameter_estimates: pd.DataFrame


def _fit_autoreg_with_residuals(series: pd.Series, lags: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """Fit an AutoReg model and return fitted values and residuals."""
    model = AutoReg(series.to_numpy(), lags=lags, old_names=False).fit()
    pred = model.fittedvalues
    resid = model.resid

    # align lengths to the model output
    return np.asarray(pred, dtype=float), np.asarray(resid, dtype=float), model.params


def simple_block_bootstrap_autoreg(
    series: pd.Series,
    block_length: int = 12,
    n_bootstrap: int = 1000,
    lags: int = 12,
    random_state: int = 12345,
) -> AutoRegBootstrapResult:
    """
    Python translation of the simple block bootstrap portion of the SAS script.
    """
    rng = np.random.default_rng(random_state)
    pred, resid, _ = _fit_autoreg_with_residuals(series, lags=lags)

    n = len(pred)
    if n % block_length != 0:
        raise ValueError("Series length after AutoReg alignment must be divisible by block_length.")

    k = n // block_length
    P = pred.reshape(k, block_length)
    R = resid.reshape(k, block_length)

    boot_rows = []
    est_rows = []

    for sample_id in range(1, n_bootstrap + 1):
        idx = rng.integers(0, k, size=k)
        yboot = (P + R[idx]).reshape(-1)

        boot_rows.append(pd.DataFrame({
            "sample_id": sample_id,
            "time": np.arange(len(yboot)),
            "yboot": yboot,
        }))

        fit = AutoReg(yboot, lags=lags, old_names=False).fit()
        row = {"sample_id": sample_id}
        row.update({f"param_{i}": float(v) for i, v in enumerate(fit.params)})
        est_rows.append(row)

    return AutoRegBootstrapResult(
        bootstrapped_series=pd.concat(boot_rows, ignore_index=True),
        parameter_estimates=pd.DataFrame(est_rows),
    )


def moving_block_bootstrap_autoreg(
    series: pd.Series,
    block_length: int = 12,
    n_bootstrap: int = 1000,
    lags: int = 12,
    random_state: int = 12345,
) -> AutoRegBootstrapResult:
    """
    Python translation of the moving block bootstrap portion of the SAS script.
    """
    rng = np.random.default_rng(random_state)
    pred, resid, _ = _fit_autoreg_with_residuals(series, lags=lags)

    n = len(pred)
    if n % block_length != 0:
        raise ValueError("Series length after AutoReg alignment must be divisible by block_length.")

    k = n // block_length
    P = pred.reshape(k, block_length)

    J = n - block_length + 1
    R = np.vstack([resid[i:i + block_length] for i in range(J)])

    boot_rows = []
    est_rows = []

    for sample_id in range(1, n_bootstrap + 1):
        idx = rng.integers(0, J, size=k)
        yboot = (P + R[idx]).reshape(-1)

        boot_rows.append(pd.DataFrame({
            "sample_id": sample_id,
            "time": np.arange(len(yboot)),
            "yboot": yboot,
        }))

        fit = AutoReg(yboot, lags=lags, old_names=False).fit()
        row = {"sample_id": sample_id}
        row.update({f"param_{i}": float(v) for i, v in enumerate(fit.params)})
        est_rows.append(row)

    return AutoRegBootstrapResult(
        bootstrapped_series=pd.concat(boot_rows, ignore_index=True),
        parameter_estimates=pd.DataFrame(est_rows),
    )


# ---------------------------------------------------------------------------
# path_finding_with_hash
# ---------------------------------------------------------------------------

def reconstruct_paths(wanders: pd.DataFrame) -> pd.DataFrame:
    """
    Python translation of the SAS path reconstruction logic.

    Input columns:
    - id: observation id
    - holeId: watering hole id
    - pid: previous observation id (or NaN for path start)

    Returns a recovered path table with:
    - subjectId: arbitrary reconstructed path identifier
    - sequence: position in path
    - id, pid, holeId
    """
    required = {"id", "holeId", "pid"}
    missing = required - set(wanders.columns)
    if missing:
        raise ValueError(f"wanders is missing columns: {missing}")

    by_id = {
        int(row.id): {"id": int(row.id), "pid": (None if pd.isna(row.pid) else int(row.pid)), "holeId": int(row.holeId)}
        for row in wanders.itertuples(index=False)
    }

    next_map: Dict[int, int] = {}
    for row in by_id.values():
        if row["pid"] is not None:
            next_map[row["pid"]] = row["id"]

    visited = set()
    out = []
    subject_id = 0

    for obs_id in list(by_id.keys()):
        if obs_id in visited:
            continue

        # walk backward to start
        current = obs_id
        seen = set()
        while by_id[current]["pid"] is not None and by_id[current]["pid"] in by_id and current not in seen:
            seen.add(current)
            current = by_id[current]["pid"]

        # now walk forward using next_map
        subject_id += 1
        seq = 1
        node = current
        while node in by_id and node not in visited:
            row = by_id[node]
            out.append({
                "subjectId": subject_id,
                "sequence": seq,
                "id": row["id"],
                "pid": row["pid"],
                "holeId": row["holeId"],
            })
            visited.add(node)
            seq += 1
            if node not in next_map:
                break
            node = next_map[node]

    return pd.DataFrame(out).sort_values(["subjectId", "sequence"]).reset_index(drop=True)


def compute_path_lengths(
    paths: pd.DataFrame,
    nodes: pd.DataFrame,
    subject_col: str = "subjectId",
    hole_col: str = "holeId",
) -> pd.DataFrame:
    """
    Compute total path length from reconstructed paths and node coordinates.

    nodes columns:
    - id
    - x
    - y
    """
    required_paths = {subject_col, hole_col}
    required_nodes = {"id", "x", "y"}

    if required_paths - set(paths.columns):
        raise ValueError(f"paths missing columns: {required_paths - set(paths.columns)}")
    if required_nodes - set(nodes.columns):
        raise ValueError(f"nodes missing columns: {required_nodes - set(nodes.columns)}")

    merged = paths.merge(nodes.rename(columns={"id": hole_col}), on=hole_col, how="left")
    merged = merged.sort_values([subject_col, "sequence"]).copy()

    merged["x_prev"] = merged.groupby(subject_col)["x"].shift(1)
    merged["y_prev"] = merged.groupby(subject_col)["y"].shift(1)
    merged["step_length"] = np.sqrt((merged["x"] - merged["x_prev"]) ** 2 + (merged["y"] - merged["y_prev"]) ** 2)
    merged["step_length"] = merged["step_length"].fillna(0.0)

    report = merged.groupby(subject_col, as_index=False)["step_length"].sum().rename(columns={"step_length": "pathLength"})
    return report


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Fair coins example
    mc = fair_coin_markov(num_tosses=10)
    print(mc.head())

    # Example moving averages
    df_demo = pd.DataFrame({
        "id": [1]*10,
        "date_n": pd.date_range("2024-01-01", periods=10, freq="D"),
        "close": np.linspace(10, 20, 10) + np.random.default_rng(1).normal(scale=0.5, size=10),
    })
    print(compute_moving_averages(df_demo).head())
