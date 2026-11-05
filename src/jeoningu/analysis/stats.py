"""회귀 도구. 수업의 OLS 기울기 해석에 Newey-West(HAC) 표준오차 옵션만 더했습니다."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps


@dataclass
class OLSResult:
    coef: np.ndarray      # [절편, 기울기...]
    se: np.ndarray
    t: np.ndarray
    p: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n: int
    r2: float

    @property
    def alpha(self) -> float:
        return float(self.coef[0])

    def as_dict(self, prefix: str = "") -> dict:
        return {
            f"{prefix}alpha": self.alpha, f"{prefix}se_alpha": float(self.se[0]),
            f"{prefix}t_alpha": float(self.t[0]), f"{prefix}p_alpha": float(self.p[0]),
            f"{prefix}ci_low": float(self.ci_low[0]), f"{prefix}ci_high": float(self.ci_high[0]),
            f"{prefix}beta": float(self.coef[1]) if len(self.coef) > 1 else np.nan,
            f"{prefix}se_beta": float(self.se[1]) if len(self.se) > 1 else np.nan,
            f"{prefix}n": self.n, f"{prefix}r2": self.r2,
        }


def ols(y, x=None, hac_lags: int = 0) -> OLSResult:
    """y = a + b·x + e. x 가 None 이면 절편만 있는 모형(평균 검정).

    hac_lags > 0 이면 Newey-West(Bartlett 가중) 표준오차. 월이 건너뛰어진 표본에서는 근사입니다.
    관측치는 시간 순으로 정렬돼 있어야 합니다.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if x is None:
        X = np.ones((n, 1))
    else:
        x = np.asarray(x, dtype=float)
        X = np.column_stack([np.ones(n), x.reshape(n, -1)])
    k = X.shape[1]
    if n <= k:
        raise ValueError(f"관측치({n})가 모수({k})보다 많아야 합니다.")
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    if hac_lags <= 0:
        cov = (e @ e) / (n - k) * xtx_inv
    else:
        u = X * e[:, None]
        s = u.T @ u
        for lag in range(1, min(hac_lags, n - 1) + 1):
            w = 1.0 - lag / (hac_lags + 1.0)
            g = u[lag:].T @ u[:-lag]
            s += w * (g + g.T)
        cov = xtx_inv @ s @ xtx_inv * n / (n - k)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * sps.t.sf(np.abs(t), n - k)
    crit = sps.t.ppf(0.975, n - k)
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (e @ e) / ss_tot if ss_tot > 0 else np.nan
    return OLSResult(beta, se, t, p, beta - crit * se, beta + crit * se, n, float(r2))
