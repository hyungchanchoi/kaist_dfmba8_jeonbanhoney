"""분석 전체 실행: 점수가 붙은 언급 표 + 수익률 → H1(종목), H2(시장), 점검, 검정력, 판정."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..months import month_str_to_ym, ym_add, ym_range
from ..text.mentions import filter_sample
from ..text.tone import aggregate_tone
from .decision import Decision, decide
from .market import market_regressions, monthly_market_tone
from .portfolio import alpha_tests, assign_groups, portfolio_returns, timing_check
from .power import alpha_detectability


@dataclass
class AnalysisResults:
    summary: dict = field(default_factory=dict)
    tone: pd.DataFrame = field(default_factory=pd.DataFrame)
    groups: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio: pd.DataFrame = field(default_factory=pd.DataFrame)
    alpha: pd.DataFrame = field(default_factory=pd.DataFrame)
    timing: pd.DataFrame = field(default_factory=pd.DataFrame)
    market_tone: pd.DataFrame = field(default_factory=pd.DataFrame)
    market: pd.DataFrame = field(default_factory=pd.DataFrame)
    power: pd.DataFrame = field(default_factory=pd.DataFrame)
    decisions: dict[str, Decision] = field(default_factory=dict)


def run_analysis(scored: pd.DataFrame, panel: pd.DataFrame, ff: pd.DataFrame, *, sample_cfg: dict,
                 tone_method: str, mentions_cfg: dict, analysis_cfg: dict, with_power: bool = True
                 ) -> AnalysisResults:
    res = AnalysisResults()
    start_m, end_m = sample_cfg["start_month"], sample_cfg["end_month"]
    sample = filter_sample(scored, start_m, end_m, exclude_guest=sample_cfg.get("exclude_guest", True),
                           forward_only=False)
    tone = aggregate_tone(sample, tone_method, forward_only=mentions_cfg.get("forward_only", True))
    res.tone = tone

    stock_tone = tone[tone["target_type"] == "stock"].rename(columns={"target": "code"})
    groups = assign_groups(stock_tone, analysis_cfg.get("group_method", "sign"),
                           analysis_cfg.get("tone_threshold", 0.0))
    res.groups = groups
    hac = int(analysis_cfg.get("hac_lags", 3))
    min_g = int(analysis_cfg.get("min_per_group", 1))
    pr = portfolio_returns(groups, panel, ff, horizon=1, min_per_group=min_g)
    res.portfolio = pr

    if len(pr) >= 4:
        res.alpha = alpha_tests(pr, hac_lags=hac)
        res.timing = timing_check(groups, panel, ff, min_per_group=min_g, hac_lags=hac)

    mt = monthly_market_tone(tone)
    res.market_tone = mt
    if len(mt) >= 4:
        res.market = market_regressions(mt, ff, hac_lags=hac, min_months=min(12, max(len(mt) // 2, 4)))

    # ---- 표본 요약
    gv = sample.drop_duplicates("video_id")
    res.summary = {
        "분석 기간": f"{start_m} ~ {end_m}",
        "표본에 쓴 영상 수": int(len(gv)),
        "언급 문장 수(종목+시장)": int(len(sample)),
        "전망 문장 수": int(sample["is_forward_final"].sum()) if "is_forward_final" in sample else np.nan,
        "종목-월 톤 관측치": int(len(stock_tone)),
        "서로 다른 종목 수": int(stock_tone["target_name"].nunique()) if len(stock_tone) else 0,
        "P 그룹 종목-월": int((groups["group"] == "P").sum()) if len(groups) else 0,
        "N 그룹 종목-월": int((groups["group"] == "N").sum()) if len(groups) else 0,
        "H−L 분석 가능 월 수": int(len(pr)),
        "시장 톤 월 수": int(len(mt)),
        "톤 방법": tone_method,
    }

    # ---- 검정력 어림 계산(실제 그룹 크기 기준)
    if with_power and len(pr) >= 8 and len(panel):
        k = int(max(1, round(float(np.nanmedian(np.minimum(pr["n_P"], pr["n_N"]))))))
        months = ym_range(month_str_to_ym(start_m), month_str_to_ym(end_m))
        months = [ym_add(m, 1) for m in months]
        res.power = pd.DataFrame([alpha_detectability(panel, ff, months, k=k,
                                                      nsim=int(analysis_cfg.get("power_sims", 300)))])

    # ---- 판정
    meaningful = float(analysis_cfg.get("meaningful_alpha", 0.02))
    min_months = int(analysis_cfg.get("min_months", 24))
    if len(res.alpha):
        hl = res.alpha.iloc[0]
        pre_p = None
        if len(res.timing) and "p" in res.timing:
            row = res.timing[res.timing["month_offset"] == -1]
            if len(row) and pd.notna(row.iloc[0].get("p")):
                pre_p = float(row.iloc[0]["p"])
        res.decisions["H1 (종목 수준)"] = decide(hl["alpha"], hl["ci_low"], hl["ci_high"], hl["p_alpha"],
                                              int(hl["n"]), meaningful, min_months, pre_p)
    else:
        res.decisions["H1 (종목 수준)"] = Decision("판정 불가", "H−L 을 만들 수 있는 달이 부족함")
    if len(res.market) and "b_per_sd_tone" in res.market:
        row = res.market[res.market["month_offset"] == 1]
        if len(row) and pd.notna(row.iloc[0].get("b_per_sd_tone")):
            r = row.iloc[0]
            # 톤 1 표준편차당 효과의 '의미 있는 크기' 는 종목 수준과 같은 meaningful 을 씀
            res.decisions["H2 (시장 수준)"] = decide(r["b_per_sd_tone"], r["ci_low"], r["ci_high"], r["p"],
                                                  int(r["n_months"]), meaningful, min_months, None)
    if "H2 (시장 수준)" not in res.decisions:
        res.decisions["H2 (시장 수준)"] = Decision("판정 불가", "시장 언급이 있는 달이 부족함")
    return res
