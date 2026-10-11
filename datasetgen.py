import numpy as np
import pandas as pd

# ----------------------------- CONFIG ----------------------------------------
N_ROWS      = 12000            # total rows (split evenly across species)
SEED        = 42              # change for a different random draw
OUT_PATH    = "rf_training_data.csv"
FOCUS_FRAC  = 0.70            # share of rows sampled near a species' range
LABEL_NOISE = 0.0             # e.g. 0.02 flips 2% of labels (realism / robustness)
MIN_SCORE   = 0.05            # drop rows no species fits at all
DOWNLOAD    = True            # auto-download the CSV in Colab
# -----------------------------------------------------------------------------

# (lo_edge, lo_core, hi_core, hi_edge) per parameter; NH3 = free-NH3 limit.
# Core ranges come from the tolerance table; edges are the soft shoulders.
SPECIES = {
    "Nile tilapia": dict(
        temp=(18.5, 25.0, 30.0, 35.0),     # optimum cited 20-30, recommended 25-30
        ph=(6.50, 6.88, 8.69, 9.20),
        do=(3.50, 4.99, 12.0, 14.0),       # min ~4.99
        nh3=0.0023),
    "Bangus": dict(
        temp=(26.0, 29.0, 30.8, 34.0),
        ph=(6.00, 6.60, 7.40, 8.20),
        do=(1.50, 2.20, 12.0, 14.0),       # min 2.2
        nh3=0.0076),
    "African catfish": dict(
        temp=(24.0, 27.4, 28.8, 32.0),
        ph=(4.30, 7.00, 8.00, 9.20),       # optimal 7-8, survival 4.3-9.2
        do=(2.50, 3.72, 12.0, 14.0),       # farm minimum 3.72
        nh3=0.41),
}
NAMES = list(SPECIES)

# Broad "anywhere" sampling domain
T_RANGE, PH_RANGE, DO_RANGE = (15.0, 36.0), (4.5, 9.5), (1.0, 12.0)
NH3_RANGE = (0.0003, 1.0)


def trap(x, a, b, c, d):
    """Trapezoid membership: 0..1."""
    x = np.asarray(x, dtype=float)
    up = np.clip((x - a) / (b - a), 0, 1)
    down = np.clip((d - x) / (d - c), 0, 1)
    return np.minimum(up, down)


def nh3_score(x, limit):
    """1 at/below limit, log-decay to 0 at 5x the limit."""
    x = np.asarray(x, dtype=float)
    return np.clip(1 - np.log(np.maximum(x, 1e-12) / limit) / np.log(5), 0, 1)


def scores(temp, ph, do, nh3):
    """Return array (n, n_species) of suitability scores."""
    cols = []
    for sp in NAMES:
        p = SPECIES[sp]
        s = (trap(temp, *p["temp"]) * trap(ph, *p["ph"]) *
             trap(do, *p["do"]) * nh3_score(nh3, p["nh3"]))
        cols.append(s)
    return np.column_stack(cols)


def log_uniform(rng, lo, hi, n):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n))


def sample_batch(rng, n):
    """Mix of rows near a random species' range and rows from anywhere."""
    n_focus = int(n * FOCUS_FRAC)
    temp = rng.uniform(*T_RANGE, n)
    ph = rng.uniform(*PH_RANGE, n)
    do = rng.uniform(*DO_RANGE, n)
    nh3 = log_uniform(rng, *NH3_RANGE, n)

    pick = rng.integers(0, len(NAMES), n_focus)
    for i, k in enumerate(pick):
        p = SPECIES[NAMES[k]]
        temp[i] = rng.uniform(p["temp"][0], p["temp"][3])
        ph[i] = rng.uniform(p["ph"][0], p["ph"][3])
        do[i] = rng.uniform(p["do"][0], min(p["do"][3], 12.0))
        nh3[i] = log_uniform(rng, 0.0003, p["nh3"] * 4, 1)[0]
    return temp, ph, do, nh3


def generate(n_rows=N_ROWS, seed=SEED):
    rng = np.random.default_rng(seed)
    quota = n_rows // len(NAMES)
    counts = {sp: 0 for sp in NAMES}
    rows = []

    while min(counts.values()) < quota:
        t, p, d, a = sample_batch(rng, 4000)
        sc = scores(t, p, d, a)
        # tiny jitter breaks exact ties randomly
        best = np.argmax(sc + rng.uniform(0, 1e-9, sc.shape), axis=1)
        ok = sc[np.arange(len(sc)), best] >= MIN_SCORE
        for i in np.where(ok)[0]:
            sp = NAMES[best[i]]
            if counts[sp] < quota:
                counts[sp] += 1
                rows.append((t[i], p[i], d[i], a[i], sp))

    df = pd.DataFrame(rows, columns=["temp_c", "ph", "do_mgL", "ammonia_mgL", "species"])

    if LABEL_NOISE > 0:
        flip = rng.random(len(df)) < LABEL_NOISE
        df.loc[flip, "species"] = rng.choice(NAMES, flip.sum())

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df.round({"temp_c": 2, "ph": 2, "do_mgL": 2, "ammonia_mgL": 5})


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows -> {OUT_PATH}\n")
    print(df["species"].value_counts(), "\n")
    print(df.groupby("species")[["temp_c", "ph", "do_mgL", "ammonia_mgL"]]
            .agg(["min", "median", "max"]).round(3).T)
    print("\nFirst rows:")
    print(df.head())
    if DOWNLOAD:
        try:
            from google.colab import files  # type: ignore
            files.download(OUT_PATH)
        except ImportError:
            pass