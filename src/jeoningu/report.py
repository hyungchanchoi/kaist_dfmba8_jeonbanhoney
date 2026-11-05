"""결과 표(CSV), 그림(PNG), 요약 보고서(report.md)를 outputs/ 에 씁니다."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from .analysis.run import AnalysisResults  # noqa: E402


def _setup_font() -> None:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if name in installed:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    if df is None or df.empty:
        return "_(결과 없음)_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                if pd.isna(v):
                    cells.append("")
                elif float(v).is_integer() and ("n" in str(c) or "month" in str(c)):
                    cells.append(str(int(v)))          # 월 수, 관측치 수는 정수로
                else:
                    cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _fig_coverage(pr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(pr))
    ax.bar(x - 0.2, pr["n_P"], width=0.4, label="P(긍정) 종목 수", color="#E8604C")
    ax.bar(x + 0.2, pr["n_N"], width=0.4, label="N(부정) 종목 수", color="#12343B")
    step = max(1, len(pr) // 12)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(pr["ym"].astype(str).iloc[::step], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("종목 수")
    ax.set_title("월별 그룹 크기 (H-L 을 만들 수 있었던 달)")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_alpha(alpha: pd.DataFrame, meaningful: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(alpha))[::-1]
    a = alpha["alpha"].to_numpy() * 100
    lo = alpha["ci_low"].to_numpy() * 100
    hi = alpha["ci_high"].to_numpy() * 100
    ax.axvspan(-meaningful * 100, meaningful * 100, color="#D5E3E1", alpha=0.6, label=f"±{meaningful:.0%} (의미 있는 크기)")
    ax.axvline(0, color="#5C6F73", lw=1)
    ax.errorbar(a, y, xerr=[a - lo, hi - a], fmt="o", color="#E8604C", ecolor="#12343B", capsize=4)
    ax.set_yticks(y)
    ax.set_yticklabels(alpha["portfolio"].str.replace("−", "-", regex=False))   # 맑은 고딕에 U+2212 글리프가 없음
    ax.set_xlabel("월 알파 (%) 와 95% 신뢰구간")
    ax.set_title("CAPM 알파")
    ax.legend(loc="upper right", fontsize=8)   # 아래쪽 N 그룹 점을 가리지 않게
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_market(mt: pd.DataFrame, ff: pd.DataFrame, path: Path) -> None:
    from .months import ym_add

    d = mt.copy()
    d["ym_ret"] = d["ym"].map(lambda v: ym_add(v, 1))
    d = d.merge(ff[["ym", "mkt_rf"]].rename(columns={"ym": "ym_ret"}), on="ym_ret")
    if len(d) < 4:
        return
    z = (d["tone"] - d["tone"].mean()) / d["tone"].std(ddof=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(z, d["mkt_rf"] * 100, color="#12343B", s=18)
    b = np.polyfit(z, d["mkt_rf"] * 100, 1)
    xs = np.linspace(z.min(), z.max(), 20)
    ax.plot(xs, b[0] * xs + b[1], color="#E8604C")
    ax.set_xlabel("월 t 시장 톤 (표준화)")
    ax.set_ylabel("월 t+1 코스피 초과수익률 (%)")
    ax.set_title("시장 수준: 톤과 다음 달 수익률")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fig_cumulative(pr: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    cum = (1 + pr["hl"]).cumprod() - 1
    ax.plot(pr["ym"].astype(str), cum * 100, color="#E8604C")
    step = max(1, len(pr) // 12)
    ax.set_xticks(range(0, len(pr), step))
    ax.set_xticklabels(pr["ym"].astype(str).iloc[::step], rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#5C6F73", lw=1)
    ax.set_ylabel("누적 H-L (%)")
    ax.set_title("긍정-부정 포트폴리오의 누적 초과수익률(참고용, 거래비용 미반영)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_report(res: AnalysisResults, out_dir: str | Path, ff: pd.DataFrame | None = None,
                 meaningful: float = 0.02, title: str = "전반꿀 검증 결과") -> Path:
    out = Path(out_dir)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    _setup_font()

    tables = {
        "tone_monthly": res.tone, "groups": res.groups, "portfolio_returns": res.portfolio,
        "alpha_tests": res.alpha, "timing_check": res.timing, "market_tone": res.market_tone,
        "market_regressions": res.market, "power": res.power,
    }
    for name, df in tables.items():
        if df is not None and len(df):
            df.to_csv(out / "tables" / f"{name}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"가설": k, "판정": d.label, "근거": d.reason} for k, d in res.decisions.items()]).to_csv(
        out / "tables" / "decisions.csv", index=False, encoding="utf-8-sig")

    if len(res.portfolio) >= 4:
        _fig_coverage(res.portfolio, out / "figures" / "coverage.png")
        _fig_cumulative(res.portfolio, out / "figures" / "cumulative_hl.png")
    if len(res.alpha):
        _fig_alpha(res.alpha, meaningful, out / "figures" / "alpha_ci.png")
    if ff is not None and len(res.market_tone) >= 4:
        _fig_market(res.market_tone, ff, out / "figures" / "market_scatter.png")

    md = [f"# {title}", ""]
    md += ["## 판정", ""]
    for k, d in res.decisions.items():
        md.append(f"- **{k}: {d.label}** — {d.reason}")
    md += ["", "'기각'은 월 단위 분석에서만 말할 수 있습니다. 영상 직후 며칠의 반응은 이 데이터로 알 수 없습니다.", ""]
    md += ["## 표본", "", _md_table(pd.DataFrame(list(res.summary.items()), columns=["항목", "값"])), ""]
    md += ["## H1 종목 수준: 긍정−부정 포트폴리오의 CAPM 알파", "",
           "월 t 의 톤으로 나눈 그룹의 월 t+1 균등가중 초과수익률을 시장 초과수익률에 회귀한 결과입니다. "
           "전반꿀이면 H−L 의 알파가 0 보다 작아야 합니다.", "",
           _md_table(res.alpha[["portfolio", "mean_excess_return", "alpha", "se_alpha", "t_alpha", "p_alpha",
                                "ci_low", "ci_high", "beta", "n"]] if len(res.alpha) else res.alpha), ""]
    md += ["### 사전 추세와 동시 반응 점검", "",
           "month_offset −1 은 발언 전 달, 0 은 같은 달, 1 은 다음 달의 H−L 평균입니다. "
           "−1 이 유의하면 결과를 발언 효과로 읽기 어렵습니다.", "", _md_table(res.timing), ""]
    md += ["## H2 시장 수준: 월별 시장 톤과 코스피 초과수익률", "",
           "month_offset 1 이 H2 검정입니다(톤 1 표준편차당 다음 달 초과수익률 변화).", "",
           _md_table(res.market), ""]
    md += ["## 검정력 어림 계산", "",
           "이 표본으로 구분할 수 있는 월 알파의 크기입니다(무작위 종목 구성 가정, 대략의 기준).", "",
           _md_table(res.power), ""]
    figs = [("coverage.png", "월별 그룹 크기"), ("alpha_ci.png", "알파와 신뢰구간"),
            ("market_scatter.png", "시장 톤과 다음 달 수익률"), ("cumulative_hl.png", "누적 H−L")]
    md += ["## 그림", ""]
    for fn, cap in figs:
        if (out / "figures" / fn).exists():
            md.append(f"![{cap}](figures/{fn})")
    md += ["", "## 읽을 때 주의", "",
           "- 월 단위 분석이라 영상 직후 며칠의 반응은 볼 수 없습니다(해외 연구는 야간 급등과 이후 반전을 보고).",
           "- 한국어 톤 사전은 검증 전 파일럿 버전입니다. LLM 채점과 사람 라벨 일치도를 함께 확인하세요.",
           "- 같은 종목이 여러 달 이어서 언급되면 관측치가 독립이 아닙니다(HAC 표준오차로 일부만 보정).",
           "- 이 결과는 연구용이며 투자 조언이 아닙니다."]
    path = out / "report.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return path
