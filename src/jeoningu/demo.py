"""합성 데이터로 파이프라인 전체를 점검하는 데모. 실제 데이터 없이도 처음부터 끝까지 돌아갑니다.

합성 세계: 종목마다 시장 베타를 가진 월 수익률을 만들고, 매달 몇 종목에 대해 '긍정/부정' 발언 문장을
넣은 가짜 스크립트를 만듭니다. planted_alpha 로 '긍정 톤 종목이 다음 달 이만큼 덜 오른다' 는 효과를
심을 수 있습니다(H−L 알파 = planted_alpha). 효과를 심었을 때 파이프라인이 그것을 찾아내는지, 심지 않았을 때
아무것도 찾지 않는지를 확인하는 용도입니다. 실제 결과가 아닙니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analysis.run import AnalysisResults, run_analysis
from .config import Config, load_config
from .months import month_str_to_ym, ym_add, ym_range, ym_to_month_str
from .pipeline import make_mentions, score
from .report import write_report

PREFIX = ["알파", "베타", "감마", "델타", "오메가", "세타", "시그마", "제타", "카파", "람다", "뮤", "파이"]
SUFFIX = ["전자", "화학", "바이오", "에너지", "건설", "제약"]


def make_synthetic(planted_alpha: float = 0.0, planted_market: float = 0.0, n_stocks: int = 60,
                   start_month: str = "2018-10", end_month: str = "2025-12", seed: int = 0,
                   mention_prob: float = 0.15):
    """(videos, universe, panel, ff, truth) 를 돌려줍니다. truth 는 심어 둔 월별 종목 톤 부호입니다."""
    rng = np.random.default_rng(seed)
    names = [p + s for p in PREFIX for s in SUFFIX][:n_stocks]
    s_ym, e_ym = month_str_to_ym(start_month), month_str_to_ym(end_month)
    months = ym_range(ym_add(s_ym, -1), ym_add(e_ym, 1))       # 앞뒤로 한 달씩 더
    sample_months = ym_range(s_ym, e_ym)

    # 월별 시장 톤과 종목 톤(심어 둘 정답)
    market_sign = {m: int(rng.choice([-1, 1])) for m in sample_months}
    stock_sign: dict[tuple[int, int], int] = {}
    for m in sample_months:
        for i in range(n_stocks):
            if rng.random() < mention_prob:
                stock_sign[(m, i)] = int(rng.choice([-1, 1]))

    # 팩터와 수익률
    ff = pd.DataFrame({"ym": months, "mkt_rf": rng.normal(0.006, 0.05, len(months)), "rf": 0.002})
    ff_idx = ff.set_index("ym")
    for m in sample_months:
        ff_idx.loc[ym_add(m, 1), "mkt_rf"] += planted_market * market_sign[m]
    ff = ff_idx.reset_index()
    betas = rng.uniform(0.6, 1.4, n_stocks)
    rows = []
    for m in months:
        mkt = ff_idx.loc[m, "mkt_rf"]
        prev = ym_add(m, -1)
        for i, name in enumerate(names):
            # 긍정(+1) 종목은 +α/2, 부정(−1) 종목은 −α/2 → H−L(P−N) 평균이 정확히 α
            effect = 0.5 * planted_alpha * stock_sign.get((prev, i), 0)
            ex = betas[i] * mkt + rng.normal(0, 0.08) + effect
            rows.append({"code": f"A{i:06d}", "name": name, "ym": m, "ex": ex, "size": float(n_stocks - i)})
    panel = pd.DataFrame(rows)
    universe = pd.DataFrame({"code": [f"A{i:06d}" for i in range(n_stocks)], "name": names, "aliases": ""})

    # 가짜 스크립트
    def stock_sentence(name: str, sign: int) -> str:
        return (f"{name}는 앞으로 크게 상승할 것입니다." if sign > 0 else f"{name}는 앞으로 크게 하락할 것입니다.")

    videos = []
    for m in sample_months:
        y, mo = divmod(m, 100)
        mentioned = [(i, stock_sign[(m, i)]) for i in range(n_stocks) if (m, i) in stock_sign]
        n_solo = 4
        chunks = [mentioned[j::n_solo] for j in range(n_solo)]
        for v in range(n_solo):
            sents = ["안녕하세요 오늘도 시청해 주셔서 감사합니다."]
            if v == 0:
                sents.append("증시는 앞으로 상승할 것으로 보입니다." if market_sign[m] > 0
                             else "증시는 앞으로 하락할 것으로 보입니다.")
            for i, sign in chunks[v]:
                sents.append(stock_sentence(names[i], sign))
            sents.append("오늘 영상은 여기까지입니다.")
            videos.append({"video_id": f"S{m}{v}", "title": f"{y}년 {mo}월 시장 이야기 {v}",
                           "published_at": f"{y}-{mo:02d}-15T03:00:00Z", "transcript": " ".join(sents)})
        # 게스트 영상: 분석에서 제외되어야 하는 문장(효과와 무관한 랜덤 톤)
        gi = int(rng.integers(0, n_stocks))
        videos.append({"video_id": f"G{m}", "title": f"{y}년 {mo}월 업종 분석(ft.김철수 박사)",
                       "published_at": f"{y}-{mo:02d}-20T03:00:00Z",
                       "transcript": stock_sentence(names[gi], int(rng.choice([-1, 1]))) + " 감사합니다."})
    videos = pd.DataFrame(videos)
    for c in ["channel_title", "description", "duration", "url"]:
        videos[c] = ""
    truth = pd.DataFrame([{"ym": m, "idx": i, "sign": s} for (m, i), s in stock_sign.items()])
    return videos, universe, panel, ff, truth


def run_demo(out_dir: str | Path = "demo_outputs", planted_alpha: float = 0.0, planted_market: float = 0.0,
             seed: int = 0, cfg: Config | None = None, write: bool = True) -> AnalysisResults:
    cfg = cfg or load_config()
    data = cfg.data
    videos, universe, panel, ff, _ = make_synthetic(planted_alpha, planted_market, seed=seed,
                                                    start_month=data["sample"]["start_month"],
                                                    end_month=data["sample"]["end_month"])
    mentions = make_mentions(videos, universe, data)
    scored = score(mentions, data, method="dictionary")
    res = run_analysis(scored, panel, ff, sample_cfg=data["sample"], tone_method="dictionary",
                       mentions_cfg=data["mentions"], analysis_cfg={**data["analysis"], "power_sims": 100})
    if write:
        write_report(res, out_dir, ff=ff, meaningful=data["analysis"]["meaningful_alpha"],
                     title=f"전반꿀 검증 결과 (합성 데모: planted_alpha={planted_alpha}, planted_market={planted_market})")
    return res
