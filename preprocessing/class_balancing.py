from __future__ import annotations

from typing import Literal

import numpy as np


BalanceMethod = Literal["none", "smote", "smote_tomek", "undersample", "oversample"]


def balance_dataset(X, y, method: BalanceMethod = "smote_tomek", random_state: int = 42):
	if method == "none":
		return X, y

	try:
		from imblearn.combine import SMOTETomek
		from imblearn.over_sampling import RandomOverSampler, SMOTE
		from imblearn.under_sampling import RandomUnderSampler
	except Exception as exc:  # pragma: no cover - dependency missing in some envs
		raise RuntimeError(
			"imbalanced-learn is required for balancing; install it or use method='none'."
		) from exc

	if method == "smote":
		sampler = SMOTE(random_state=random_state)
	elif method == "undersample":
		sampler = RandomUnderSampler(random_state=random_state)
	elif method == "oversample":
		sampler = RandomOverSampler(random_state=random_state)
	else:
		sampler = SMOTETomek(random_state=random_state)

	X_res, y_res = sampler.fit_resample(X, y)
	if isinstance(X_res, np.ndarray):
		return X_res, y_res
	return X_res.values, y_res
