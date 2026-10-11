"""Model: LightGBM (L1 objective on log-rate) x 5 seeds, recency-weighted, + linear daily-market term."""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from .features import FEATURES_FINAL

PARAMS = dict(objective="l1", n_estimators=1000, learning_rate=0.03, num_leaves=15, min_child_samples=60,
              reg_lambda=5.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=1)
HALF_LIFE_DAYS = 45
CLEAN_BAND = 0.3  # |log residual| below this = clean row when estimating the market slope


class FreightRateModel:
    def __init__(self, seeds=(0, 1, 2), half_life=HALF_LIFE_DAYS, features=FEATURES_FINAL, params=None, use_market=True):
        self.seeds, self.half_life, self.features, self.use_market = seeds, half_life, list(features), use_market
        self.params = {**PARAMS, **(params or {})}

    def fit(self, X: pd.DataFrame, rate: pd.Series, dates: pd.Series):
        y = np.log(np.asarray(rate, float))
        age = (dates.max() - dates).dt.days.values
        w = 0.5 ** (age / self.half_life) if self.half_life else None
        self.models_ = [lgb.LGBMRegressor(**self.params, random_state=s).fit(X[self.features], y, sample_weight=w) for s in self.seeds]
        self.slope_ = 0.0
        if self.use_market:
            r = y - self._base(X)
            ok = np.abs(r) < CLEAN_BAND
            self.slope_ = float(np.polyfit(X["dm_dev"].values[ok], r[ok], 1)[0])
        return self

    def _base(self, X):
        return np.mean([m.predict(X[self.features]) for m in self.models_], axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(self._base(X) + self.slope_ * X["dm_dev"].values)

    def importance(self):
        g = np.mean([m.booster_.feature_importance("gain") for m in self.models_], axis=0)
        return pd.Series(g / g.sum(), index=self.features).sort_values(ascending=False)
