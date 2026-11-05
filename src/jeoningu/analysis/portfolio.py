"""종목 수준 분석 (수업 Case 1 엔진에서 정렬 변수만 '베타'에서 '톤'으로 바꾼 것).

  1) 월 t 의 톤으로 종목을 P(긍정) / N(부정) 그룹으로 나눕니다.
  2) 월 t+1 의 균등가중 초과수익률을 그룹별로 계산합니다(미래 정보를 쓰지 않음).
  3) H−L = P − N.  H−L 을 시장 초과수익률에 회귀해 알파를 봅니다(CAPM).

전반꿀이 맞다면 H−L 의 알파는 0 보다 작아야 합니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..months import ym_add
from .stats import ols


def assign_groups(tone: pd.DataFrame, method: str = "sign", threshold: float = 0.0) -> pd.DataFrame:
    """종목-월 톤 표(ym, code, tone)에 group('P'/'N') 을 붙입니다. 어느 쪽도 아니면 버립니다."""
    t = tone[["ym", "code", "tone"]].copy()
    if method == "sign":
        t["group"] = np.where(t["tone"] > threshold, "P", np.where(t["tone"] < -threshold, "N", ""))
    elif method == "median":
        med = t.groupby("ym")["tone"].transform("median")
        cnt = t.groupby("ym")["code"].transform("size")
        t["group"] = np.where(cnt < 2, "", np.where(t["tone"] > med, "P", np.where(t["tone"] < med, "N", "")))
    else:
        raise ValueError(f"알 수 없는 group_method: {method}")
    return t[t["group"] != ""].reset_index(drop=True)


def portfolio_returns(groups: pd.DataFrame, panel: pd.DataFrame, ff: pd.DataFrame,
                      horizon: int = 1, min_per_group: int = 1) -> pd.DataFrame:
    """그룹별 월 (t+horizon) 균등가중 초과수익률과 H−L 을 만듭니다.

    반환 열: ym(발언 월 t), ym_ret(수익률 월), r_P, r_N, n_P, n_N, hl, mkt_rf
    각 그룹에 수익률이 있는 종목이 min_per_group 미만인 달은 제외합니다.
    """
    g = groups[["ym", "code", "group"]].copy()
    g["ym_ret"] = g["ym"].map(lambda v: ym_add(v, horizon))
    p = panel[["code", "ym", "ex"]].rename(columns={"ym": "ym_ret"})
    m = g.merge(p, on=["code", "ym_ret"], how="inner").dropna(subset=["ex"])
    if m.empty:
        return pd.DataFrame(columns=["ym", "ym_ret", "r_P", "r_N", "n_P", "n_N", "hl", "mkt_rf"])
    agg = m.groupby(["ym", "ym_ret", "group"])["ex"].agg(["mean", "size"]).unstack("group")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index().rename(columns={"mean_P": "r_P", "mean_N": "r_N", "size_P": "n_P", "size_N": "n_N"})
    for c in ["r_P", "r_N", "n_P", "n_N"]:
        if c not in agg:
            agg[c] = np.nan
    agg = agg.dropna(subset=["r_P", "r_N"])
    agg = agg[(agg["n_P"] >= min_per_group) & (agg["n_N"] >= min_per_group)].copy()
    agg["hl"] = agg["r_P"] - agg["r_N"]
    agg = agg.merge(ff[["ym", "mkt_rf"]].rename(columns={"ym": "ym_ret"}), on="ym_ret", how="inner")
    return agg.sort_values("ym").reset_index(drop=True)[
        ["ym", "ym_ret", "r_P", "r_N", "n_P", "n_N", "hl", "mkt_rf"]]


def alpha_tests(pr: pd.DataFrame, hac_lags: int = 3) -> pd.DataFrame:
    """H−L, P, N 각각을 시장 초과수익률에 회귀한 알파 표."""
    rows = []
    for label, col in [("H-L (P-N)", "hl"), ("P (긍정)", "r_P"), ("N (부정)", "r_N")]:
        res = ols(pr[col], pr["mkt_rf"], hac_lags=hac_lags)
        row = {"portfolio": label, "mean_excess_return": float(pr[col].mean())}
        row.update(res.as_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def timing_check(groups: pd.DataFrame, panel: pd.DataFrame, ff: pd.DataFrame,
                 shifts=(-1, 0, 1), min_per_group: int = 1, hac_lags: int = 3) -> pd.DataFrame:
    """사전 추세와 동시 반응 점검. 같은 그룹을 발언 월 기준으로 t−1, t, t+1 에 놓고 H−L 평균을 봅니다.

    t−1 의 H−L 이 이미 크면(긍정 톤 종목이 직전에 올랐으면) 이후 결과를 발언 효과라고 읽기 어렵습니다.
    """
    rows = []
    for k in shifts:
        pr = portfolio_returns(groups, panel, ff, horizon=k, min_per_group=min_per_group)
        if len(pr) < 4:
            rows.append({"month_offset": k, "n_months": len(pr)})
            continue
        res = ols(pr["hl"], None, hac_lags=hac_lags)
        rows.append({"month_offset": k, "n_months": len(pr), "mean_hl": res.alpha,
                     "se": float(res.se[0]), "t": float(res.t[0]), "p": float(res.p[0])})
    return pd.DataFrame(rows)
