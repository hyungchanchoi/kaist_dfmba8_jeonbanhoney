"""수익률 데이터 로딩 (수업 Case 1: Korea 월별 주가 + Fama-French Korea 팩터).

데이터 파일은 수업에서 받은 것을 그대로 쓰며 저장소에는 올리지 않습니다(라이선스).
config/local.toml 의 paths.korea_stocks, paths.ff_korea 에 경로를 적어 주세요.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Korea_Stocks_2000_2025.csv 의 열 순서(한글 헤더는 인코딩 문제가 잦아 위치로 읽습니다)
STOCK_COLUMNS = ["code", "mkt", "name", "ksic", "ym", "date", "price", "ret_div", "ret", "volume", "shares_k"]


def _read_csv_any(path: str | Path, **kwargs) -> pd.DataFrame:
    last: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last = exc
    raise last  # type: ignore[misc]


def load_korea_stocks(path: str | Path) -> pd.DataFrame:
    """월별 종목 수익률. 반환 열: code, mkt, name, ym(int), price, ret_div, mcap(원)."""
    df = _read_csv_any(path)
    if df.shape[1] != len(STOCK_COLUMNS):
        raise ValueError(f"예상한 11개 열이 아닙니다: {df.shape[1]}개")
    df.columns = STOCK_COLUMNS
    df["ym"] = df["ym"].astype(int)
    df["mcap"] = df["price"] * df["shares_k"] * 1000
    return df[["code", "mkt", "name", "ym", "price", "ret_div", "mcap"]]


def load_ff_korea(path: str | Path) -> pd.DataFrame:
    """월별 팩터(소수 단위). 반환 열: ym, mkt_rf, smb, hml, rf. 연간 요약 행과 설명 행은 버립니다."""
    raw = _read_csv_any(path, header=None, dtype=str)
    raw = raw[raw[0].astype(str).str.fullmatch(r"\d{6}")].copy()
    ff = raw.iloc[:, :5].copy()
    ff.columns = ["ym", "mkt_rf", "smb", "hml", "rf"]
    ff = ff.astype({"ym": int, "mkt_rf": float, "smb": float, "hml": float, "rf": float})
    for c in ["mkt_rf", "smb", "hml", "rf"]:
        ff[c] = ff[c] / 100.0
    return ff.sort_values("ym").reset_index(drop=True)


def build_panel(stocks: pd.DataFrame, ff: pd.DataFrame) -> pd.DataFrame:
    """종목-월 초과수익률 표. ex = 수익률(배당 재투자) − 무위험수익률."""
    p = stocks.merge(ff[["ym", "rf"]], on="ym", how="inner")
    p["ex"] = p["ret_div"] - p["rf"]
    return p[["code", "name", "ym", "ex", "mcap"]]
