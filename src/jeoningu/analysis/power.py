"""검정력 어림 계산: '이 표본으로는 월 몇 %의 알파부터 구분할 수 있는가'.

매달 시가총액 상위 종목에서 k 개씩을 무작위로 P, N 으로 뽑았다고 가정하고, H−L 알파의 표준오차를
모의실험으로 구합니다. 실제 언급 종목 수와 구성은 다르므로 대략의 기준으로만 쓰세요.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .stats import ols


def alpha_detectability(panel: pd.DataFrame, ff: pd.DataFrame, months: list[int], k: int = 5,
                        top_n: int = 200, nsim: int = 300, hac_lags: int = 0, seed: int = 0) -> dict:
    """months 는 수익률을 보는 달(YYYYMM) 목록. 반환: 중앙값 표준오차와 구분 가능한 알파(5%, 80% 검정력)."""
    rng = np.random.default_rng(seed)
    p = panel[panel["ym"].isin(months)].dropna(subset=["ex"]).copy()
    p["rank"] = p.groupby("ym")["mcap"].rank(ascending=False, method="first")
    p = p[p["rank"] <= top_n]
    by_month = {ym: g["ex"].to_numpy() for ym, g in p.groupby("ym")}
    mk = ff.set_index("ym")["mkt_rf"]
    usable = [m for m in months if m in by_month and len(by_month[m]) >= 2 * k and m in mk.index]
    if len(usable) < 8:
        return {"k": k, "n_months": len(usable), "median_se": np.nan,
                "detectable_5pct": np.nan, "detectable_80pct_power": np.nan}
    x = mk.loc[usable].to_numpy()
    ses = []
    for _ in range(nsim):
        hl = []
        for m in usable:
            v = by_month[m]
            idx = rng.choice(len(v), size=2 * k, replace=False)
            hl.append(v[idx[:k]].mean() - v[idx[k:]].mean())
        ses.append(ols(np.array(hl), x, hac_lags=hac_lags).se[0])
    se = float(np.median(ses))
    return {"k": k, "n_months": len(usable), "median_se": se,
            "detectable_5pct": 2.0 * se, "detectable_80pct_power": 2.8 * se}
