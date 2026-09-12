import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fixed once for the whole study.  All experimental conditions for every seed use
# exactly the same held-out test set.  Experimental seeds change client partitions,
# injected noise, and SGD shuffling, not the test split.
TRAIN_TEST_SPLIT_SEED = 42


def prepare_global_data(seed: int = 42):
    """Load California Housing and create the frozen 80/20 train-test split.

    The StandardScaler is fitted ONLY on the training split and then applied to
    training and test data.  The `seed` argument is retained for backward
    compatibility but does not alter the frozen train/test split.
    """
    X, y = fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=TRAIN_TEST_SPLIT_SEED
    )
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def make_partitions(X_train, y_train, num_clients: int, iid: bool, seed: int):
    """Create deterministic IID or feature-skew non-IID partitions.

    IID: randomly permute training records using the experiment seed and split
    equally.  non-IID: sort by the standardized median-income feature and assign
    contiguous shards.  For a given seed/client-count/partition type, baseline,
    noise, and crowd runs therefore receive exactly the same records.
    """
    n = len(X_train)
    if iid:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
    else:
        idx = np.argsort(X_train[:, 0], kind="stable")
    return [np.asarray(a, dtype=int) for a in np.array_split(idx, num_clients)]
