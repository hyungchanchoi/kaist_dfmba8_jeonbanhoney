"""시장 수준 분석 (가설 H2): 월별 시장 톤이 높을수록 다음 달 코스피 초과수익률이 낮은가?

  r_m(t+1) = a + b · Tone(t) + e      (Tone 은 표준화해서 '톤 1 표준편차당' 효과로 읽음)

전반꿀이면 b < 0. 같은 달(t)과 직전 달(t−1)에도 돌려서 사전 추세와 동시 반응을 함께 봅니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..months import ym_add
from .stats import ols


def monthly_market_tone(tone: pd.DataFrame) -> pd.DataFrame:
    """대상이 시장인 행만 모아 월별 톤 표(ym, tone, n_sent, n_videos)를 만듭니다."""
    m = tone[tone["target_type"] == "market"][["ym", "tone", "n_sent", "n_videos"]]
    return m.sort_values("ym").reset_index(drop=True)


def market_regressions(market_tone: pd.DataFrame, ff: pd.DataFrame, offsets=(1, 0, -1),
                       hac_lags: int = 3, min_months: int = 12) -> pd.DataFrame:
    """offset k 는 '월 t 의 톤 → 월 t+k 의 시장 초과수익률'. k=1 이 H2 검정, 0 과 −1 은 점검용."""
    rows = []
    mt = market_tone.copy()
    sd = mt["tone"].std(ddof=0)
    mt["tone_z"] = (mt["tone"] - mt["tone"].mean()) / sd if sd and sd > 0 else np.nan
    for k in offsets:
        d = mt.copy()
        d["ym_ret"] = d["ym"].map(lambda v: ym_add(v, k))
        d = d.merge(ff[["ym", "mkt_rf"]].rename(columns={"ym": "ym_ret"}), on="ym_ret", how="inner")
        d = d.dropna(subset=["tone_z"]).sort_values("ym")
        row = {"month_offset": k, "n_months": len(d)}
        if len(d) >= max(min_months, 4):
            res = ols(d["mkt_rf"], d["tone_z"], hac_lags=hac_lags)
            row.update({"b_per_sd_tone": float(res.coef[1]), "se": float(res.se[1]),
                        "t": float(res.t[1]), "p": float(res.p[1]),
                        "ci_low": float(res.ci_low[1]), "ci_high": float(res.ci_high[1]),
                        "r2": res.r2})
        rows.append(row)
    return pd.DataFrame(rows)
