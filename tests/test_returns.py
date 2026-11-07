"""Yahoo Finance 수익률 모듈 테스트: 실제 네트워크 대신 가짜 가격을 씁니다."""
import numpy as np
import pandas as pd
import pytest

from jeoningu.returns import data as rd


def _daily(prices_by_date: dict[str, float], volume: float = 1000.0, tz: str | None = "Asia/Seoul"):
    idx = pd.to_datetime(list(prices_by_date))
    if tz:
        idx = idx.tz_localize(tz)
    return pd.DataFrame({"Close": list(prices_by_date.values()), "Volume": volume}, index=idx)


def test_monthly_return_uses_last_close_of_each_month():
    daily = _daily({
        "2024-01-02": 100.0, "2024-01-31": 110.0,          # 1월 마지막 거래일 종가 110
        "2024-02-01": 90.0, "2024-02-29": 121.0,           # 2월 마지막 거래일 종가 121 → +10%
        "2024-03-04": 100.0, "2024-03-29": 121.0,          # 3월 → 0%
    })
    m = rd.to_monthly(daily).set_index("ym")
    assert list(m.index) == [202402, 202403]                # 첫 달은 이전 종가가 없어 제외
    assert m.loc[202402, "ret"] == pytest.approx(0.10)
    assert m.loc[202403, "ret"] == pytest.approx(0.0)


def test_monthly_size_is_traded_value_and_tz_is_handled():
    daily = _daily({"2024-01-31": 10.0, "2024-02-01": 10.0, "2024-02-29": 20.0}, volume=100.0)
    m = rd.to_monthly(daily).set_index("ym")
    assert m.loc[202402, "size"] == pytest.approx(10.0 * 100 + 20.0 * 100)
    naive = rd.to_monthly(_daily({"2024-01-31": 10.0, "2024-02-29": 11.0}, tz=None))
    assert naive.loc[0, "ret"] == pytest.approx(0.10)


def test_download_collects_failures_and_retries():
    calls = {"A": 0}

    def fetch(t, start, end):
        if t == "A":
            calls["A"] += 1
            if calls["A"] < 2:
                raise RuntimeError("일시적 오류")             # 한 번 실패한 뒤 성공해야 함
            return _daily({"2024-01-31": 1.0, "2024-02-29": 2.0})
        if t == "EMPTY":
            return pd.DataFrame(columns=["Close", "Volume"])
        raise RuntimeError("영구 오류")

    out, failed = rd.download_daily(["A", "EMPTY", "BAD"], "2024-01-01", "2024-03-01", fetch=fetch,
                                    sleep_seconds=0, retries=2, log=lambda m: None)
    assert list(out) == ["A"] and calls["A"] == 2
    assert set(failed) == {"EMPTY", "BAD"}


def test_returns_table_and_panel_subtract_rf():
    d1 = _daily({"2024-01-31": 100.0, "2024-02-29": 110.0, "2024-03-29": 99.0})
    table = rd.build_returns_table({"005930.KS": d1})
    assert set(table.columns) == {"code", "ym", "ret", "size"} and len(table) == 2
    universe = pd.DataFrame({"code": ["005930.KS"], "name": ["삼성전자"], "aliases": [""]})
    rf = rd.make_rf([202402, 202403], rf_annual=0.12)
    monthly_rf = 1.12 ** (1 / 12) - 1
    panel = rd.build_panel(table, universe, rf)
    row = panel[panel["ym"] == 202402].iloc[0]
    assert row["name"] == "삼성전자" and row["ex"] == pytest.approx(0.10 - monthly_rf)


def test_factors_market_excess_return():
    market = pd.DataFrame({"ym": [202402, 202403], "ret": [0.05, -0.02]})
    rf = rd.make_rf([202402, 202403], rf_annual=0.0)
    ff = rd.build_factors(market, rf)
    assert ff["mkt_rf"].tolist() == pytest.approx([0.05, -0.02])
    assert set(ff.columns) == {"ym", "mkt_rf", "rf"}


def test_rf_file_overrides_constant_and_must_cover_months(tmp_path):
    p = tmp_path / "rf.csv"
    pd.DataFrame({"ym": [202402, 202403], "rf": [0.002, 0.003]}).to_csv(p, index=False)
    rf = rd.make_rf([202402, 202403], rf_annual=0.5, rf_path=p)
    assert rf["rf"].tolist() == [0.002, 0.003]
    with pytest.raises(ValueError, match="없는 달"):
        rd.make_rf([202402, 202404], rf_path=p)


def test_universe_loader_cleans_and_requires_columns(tmp_path):
    p = tmp_path / "u.csv"
    p.write_text("code,name,aliases\n005930.KS, 삼성전자 ,삼전\n005930.KS,중복,\n,이름없음,\n", encoding="utf-8")
    u = rd.load_universe(p)
    assert len(u) == 1 and u.loc[0, "name"] == "삼성전자"
    bad = tmp_path / "bad.csv"
    bad.write_text("ticker,name\nX,Y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="code"):
        rd.load_universe(bad)


def test_shipped_universe_file_is_valid():
    from jeoningu.config import REPO_ROOT

    u = rd.load_universe(REPO_ROOT / "config" / "universe.csv")
    assert len(u) >= 30 and u["code"].str.fullmatch(r"\d{6}\.(KS|KQ)").all()
    assert u["code"].is_unique and u["name"].is_unique


def test_date_window_has_margin():
    start, end = rd.date_window(201810, 202512)
    assert start == "2018-07-01" and end == "2026-02-01"
