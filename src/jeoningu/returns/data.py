"""수익률 데이터: Yahoo Finance(yfinance)에서 일별 수정종가를 받아 월 수익률로 만듭니다.

흐름
  prices 명령 : config/universe.csv 의 종목 + 코스피 지수(^KS11) 일별 가격 → 월 수익률 CSV (네트워크 필요)
  analyze 단계: 저장된 CSV 만 읽음 (네트워크 불필요, 재현 가능)

알아둘 점
  - 종목 사전(universe.csv)에 없는 종목은 분석할 수 없습니다. 영상에서 자주 나오는 종목을 추가하세요.
  - Yahoo Finance 에는 상장폐지 종목이 거의 없어서 생존 편향이 있습니다.
  - 수정종가(auto_adjust)는 배당과 분할을 반영합니다. 코스피 지수(^KS11)는 배당이 빠진 가격 지수입니다.
  - 무위험수익률은 Yahoo 에 없습니다. rf_monthly.csv(ym, rf) 를 주거나 returns.rf_annual 상수를 씁니다.
    H-L(P-N) 수익률에서는 무위험수익률이 상쇄되므로 H1 알파에는 거의 영향이 없고, 시장 초과수익률에만 영향을 줍니다.
  - 시가총액은 구하지 않고 월 거래대금(종가 × 거래량 합)을 크기 대용(size)으로 씁니다. 검정력 계산에만 쓰입니다.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

FetchFn = Callable[[str, str, str], pd.DataFrame]  # (ticker, start, end) -> DataFrame[Close, Volume]


# ------------------------------------------------------------------ 종목 사전(universe)


def load_universe(path: str | Path) -> pd.DataFrame:
    """config/universe.csv 를 읽습니다. 열: code(Yahoo 티커), name, aliases(세미콜론으로 구분, 선택)."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    for col in ("code", "name"):
        if col not in df.columns:
            raise ValueError(f"universe 파일에 '{col}' 열이 없습니다: {path}")
    if "aliases" not in df.columns:
        df["aliases"] = ""
    for col in ("code", "name", "aliases"):
        df[col] = df[col].str.strip()
    df = df[(df["code"] != "") & (df["name"] != "")]
    return df.drop_duplicates(subset=["code"]).reset_index(drop=True)[["code", "name", "aliases"]]


# ------------------------------------------------------------------ 다운로드


def yahoo_fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
    """yfinance 로 한 종목의 일별 수정종가와 거래량을 받습니다(기본 구현)."""
    import yfinance as yf

    h = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=True)
    return h[["Close", "Volume"]] if not h.empty else h


def download_daily(tickers: list[str], start: str, end: str, fetch: FetchFn | None = None,
                   sleep_seconds: float = 0.3, retries: int = 2,
                   log: Callable[[str], None] = print) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """종목별 일별 가격을 받습니다. 반환: ({티커: 일별 표}, 실패한 티커 목록)."""
    fetch = fetch or yahoo_fetch
    out: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    for i, t in enumerate(tickers, start=1):
        data = None
        for attempt in range(retries + 1):
            try:
                data = fetch(t, start, end)
                break
            except Exception as exc:  # noqa: BLE001 - 네트워크·파싱 오류는 종류가 다양함
                if attempt == retries:
                    log(f"[{i}/{len(tickers)}] {t} 실패: {type(exc).__name__}: {str(exc)[:80]}")
                time.sleep(sleep_seconds * (attempt + 1))
        if data is None or len(data) == 0:
            failed.append(t)
        else:
            out[t] = data
        if sleep_seconds:
            time.sleep(sleep_seconds)
    return out, failed


# ------------------------------------------------------------------ 월 수익률


def to_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """일별(Close, Volume)을 월말 종가 기준 월 수익률로 바꿉니다. 반환 열: ym, ret, size.

    ret  = 이번 달 마지막 거래일 종가 / 지난 달 마지막 거래일 종가 − 1 (첫 달은 이전 값이 없어 제외)
    size = 그 달 거래대금(종가 × 거래량의 합). 시가총액 대용 지표
    """
    d = daily.copy()
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    d.index = idx
    per = d.index.to_period("M")
    close_m = d["Close"].groupby(per).last()
    size = (d["Close"] * d["Volume"]).groupby(per).sum()
    out = pd.DataFrame({"ret": close_m.pct_change(), "size": size})
    out.index = [p.year * 100 + p.month for p in out.index]
    out.index.name = "ym"
    return out.dropna(subset=["ret"]).reset_index()


def build_returns_table(daily_by_ticker: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """종목별 월 수익률을 긴 표로 합칩니다. 열: code, ym, ret, size."""
    parts = []
    for code, daily in daily_by_ticker.items():
        m = to_monthly(daily)
        m.insert(0, "code", code)
        parts.append(m)
    if not parts:
        return pd.DataFrame(columns=["code", "ym", "ret", "size"])
    return pd.concat(parts, ignore_index=True)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_returns(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


# ------------------------------------------------------------------ 무위험수익률, 시장, 패널


def make_rf(months: list[int], rf_annual: float = 0.0, rf_path: str | Path | None = None) -> pd.DataFrame:
    """월별 무위험수익률(소수). 파일(ym, rf)이 있으면 그것을, 없으면 연율 상수를 월로 바꿔 씁니다."""
    if rf_path and Path(rf_path).exists():
        rf = pd.read_csv(rf_path, encoding="utf-8-sig")[["ym", "rf"]]
        rf["ym"] = rf["ym"].astype(int)
        missing = sorted(set(months) - set(rf["ym"]))
        if missing:
            raise ValueError(f"rf 파일에 없는 달이 있습니다(예: {missing[:3]}).")
        return rf.sort_values("ym").reset_index(drop=True)
    monthly = (1.0 + rf_annual) ** (1.0 / 12.0) - 1.0
    return pd.DataFrame({"ym": sorted(set(months)), "rf": monthly})


def build_factors(market_monthly: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    """시장 월 수익률에서 무위험수익률을 빼서 mkt_rf 를 만듭니다. 반환 열: ym, mkt_rf, rf."""
    ff = market_monthly[["ym", "ret"]].merge(rf, on="ym", how="inner")
    ff["mkt_rf"] = ff["ret"] - ff["rf"]
    return ff[["ym", "mkt_rf", "rf"]].sort_values("ym").reset_index(drop=True)


def build_panel(returns: pd.DataFrame, universe: pd.DataFrame, rf: pd.DataFrame) -> pd.DataFrame:
    """종목-월 초과수익률 표. 반환 열: code, name, ym, ex(= ret − rf), size."""
    p = returns.merge(rf, on="ym", how="inner").merge(universe[["code", "name"]], on="code", how="left")
    p["ex"] = p["ret"] - p["rf"]
    return p[["code", "name", "ym", "ex", "size"]]


def load_panel(cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """설정에서 경로를 읽어 (panel, ff) 를 만듭니다. prices 명령으로 만든 CSV 가 있어야 합니다."""
    universe = load_universe(cfg.path("universe"))
    returns = load_returns(cfg.path("returns_monthly"))
    market = load_returns(cfg.path("market_monthly"))
    rcfg = cfg.section("returns")
    rf_path = cfg.path("rf_monthly") if cfg.has_path("rf_monthly") else None
    rf = make_rf(sorted(set(market["ym"])), float(rcfg.get("rf_annual", 0.0)), rf_path)
    return build_panel(returns, universe, rf), build_factors(market, rf)


def date_window(start_ym: int, end_ym: int) -> tuple[str, str]:
    """수익률 계산에 필요한 다운로드 기간. 시작 3개월 전부터 끝 2개월 뒤까지(t−1, t+1 수익률 확보)."""
    from ..months import ym_add

    def first_day(ym: int) -> str:
        y, m = divmod(ym, 100)
        return f"{y:04d}-{m:02d}-01"

    return first_day(ym_add(start_ym, -3)), first_day(ym_add(end_ym, 2))
