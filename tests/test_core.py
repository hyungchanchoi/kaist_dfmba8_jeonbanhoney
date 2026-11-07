"""핵심 로직 단위 테스트. 외부 데이터나 API 키 없이 돌아갑니다."""
import json

import numpy as np
import pandas as pd
import pytest

from jeoningu.analysis.decision import decide
from jeoningu.analysis.portfolio import alpha_tests, assign_groups, portfolio_returns
from jeoningu.analysis.stats import ols
from jeoningu.months import month_str_to_ym, ym_add, ym_range, ym_to_month_str
from jeoningu.text.clean import (build_sentence_table, clean_transcript, is_guest_video,
                                 split_sentences, to_kst_month)
from jeoningu.text.mentions import build_stock_dictionary, compile_stock_regex, extract_mentions
from jeoningu.text.tone import (JsonlCache, LLMScorer, aggregate_tone, score_dictionary,
                                score_mentions)


# ---------------------------------------------------------------- months
def test_ym_add_crosses_year():
    assert ym_add(201812, 1) == 201901
    assert ym_add(201801, -1) == 201712
    assert ym_add(202501, 12) == 202601


def test_month_conversions_and_range():
    assert month_str_to_ym("2018-10") == 201810
    assert ym_to_month_str(201810) == "2018-10"
    assert ym_range(201811, 201902) == [201811, 201812, 201901, 201902]


# ---------------------------------------------------------------- clean
def test_clean_and_split():
    text = "[음악] 안녕하세요.\n오늘은 코스피 이야기입니다. 감사합니다!"
    cleaned = clean_transcript(text)
    assert "[음악]" not in cleaned and "\n" not in cleaned
    assert split_sentences(cleaned) == ["안녕하세요.", "오늘은 코스피 이야기입니다.", "감사합니다!"]


def test_guest_detection():
    assert is_guest_video("지금은 이런 바이오 기업을 사야합니다(ft.강하나 박사 1부)")
    assert not is_guest_video("검은월요일, 코스피는 얼마나 떨어질까?(ft.딥시크, 코어위브)")


def test_kst_month_boundary():
    # UTC 10월 31일 16:00 은 한국 시간으로 11월 1일 01:00
    assert to_kst_month("2026-10-31T16:00:00Z") == "2026-11"
    assert to_kst_month("2026-10-31T10:00:00Z") == "2026-10"


# ---------------------------------------------------------------- mentions
def _universe():
    return pd.DataFrame([
        {"code": "005930.KS", "name": "삼성전자", "aliases": "삼전"},
        {"code": "000660.KS", "name": "SK하이닉스", "aliases": ""},
        {"code": "A3", "name": "대상", "aliases": ""},
        {"code": "196170.KQ", "name": "알테오젠", "aliases": ""},
    ])


def test_dictionary_excludes_short_names_and_adds_alias():
    d = build_stock_dictionary(_universe(), min_name_len=3, aliases={"하이닉스": "SK하이닉스"})
    assert "대상" not in d                       # 두 글자 일반 단어와 겹치는 이름 제외
    assert d["삼전"][0] == "005930.KS"           # 별칭은 두 글자여도 허용(universe 의 aliases 열)
    assert d["하이닉스"][0] == "000660.KS"       # 설정의 별칭
    assert d["SK하이닉스"] == ("000660.KS", "SK하이닉스")


def test_alias_for_unknown_stock_is_ignored():
    d = build_stock_dictionary(_universe(), aliases={"테슬라": "Tesla"})
    assert "테슬라" not in d


def test_longest_name_wins_and_prefix_guard():
    d = build_stock_dictionary(_universe(), aliases={"하이닉스": "SK하이닉스"})
    rx = compile_stock_regex(d)
    assert rx.search("SK하이닉스가 오를 겁니다").group(0) == "SK하이닉스"
    assert rx.search("삼성전자는 좋습니다").group(0) == "삼성전자"
    assert rx.search("한국삼성전자") is None      # 앞에 한글이 붙으면 다른 단어의 일부


def test_extract_mentions_market_and_forward():
    d = build_stock_dictionary(_universe())
    sent = pd.DataFrame([
        {"video_id": "v1", "month_kst": "2018-10", "guest": False, "sent_idx": 0,
         "sentence": "삼성전자는 앞으로 오를 것입니다."},
        {"video_id": "v1", "month_kst": "2018-10", "guest": False, "sent_idx": 1,
         "sentence": "어제 코스피는 하락했습니다."},
    ])
    m = extract_mentions(sent, d, ["코스피", "증시"], "것입니다|전망")
    by = {(r.target_type, r.target): r.is_forward for r in m.itertuples()}
    assert by[("stock", "005930.KS")] == True  # noqa: E712
    assert by[("market", "KOSPI")] == False                            # noqa: E712 (설명 문장)


# ---------------------------------------------------------------- tone
def test_dictionary_scores_count_words():
    df = score_dictionary(pd.Series(["내일 크게 상승하고 호재가 있습니다", "하락과 폭락이 우려됩니다"]))
    assert df.loc[0, "pos"] >= 2 and df.loc[0, "neg"] == 0
    assert df.loc[1, "neg"] >= 3


def test_aggregate_tone_pools_counts_and_filters_forward():
    m = pd.DataFrame([
        {"video_id": "v1", "month_kst": "2018-10", "guest": False, "sent_idx": 0, "sentence": "a",
         "target_type": "stock", "target": "A1", "target_name": "삼성전자", "is_forward": True,
         "pos": 2, "neg": 0, "unc": 0, "weak": 0, "strong": 0, "n_words": 10, "is_forward_final": True},
        {"video_id": "v1", "month_kst": "2018-10", "guest": False, "sent_idx": 1, "sentence": "b",
         "target_type": "stock", "target": "A1", "target_name": "삼성전자", "is_forward": False,
         "pos": 0, "neg": 5, "unc": 0, "weak": 0, "strong": 0, "n_words": 10, "is_forward_final": False},
    ])
    t = aggregate_tone(m, "dictionary", forward_only=True)
    assert len(t) == 1 and t.loc[0, "tone"] == pytest.approx(0.2)     # 설명 문장(부정 5개)은 제외
    t_all = aggregate_tone(m, "dictionary", forward_only=False)
    assert t_all.loc[0, "tone"] == pytest.approx((2 - 5) / 20)


def test_llm_scorer_parses_caches_and_clamps(tmp_path):
    calls = []

    def fake_complete(system, user):
        calls.append(user)
        items = json.loads(user)
        out = [{"id": i["id"], "sentiment": 9 if "좋" in i["text"] else -1, "uncertainty": 0,
                "forward": True} for i in items]
        return "결과입니다:\n" + json.dumps(out, ensure_ascii=False)

    sc = LLMScorer(fake_complete, model_tag="fake", cache_path=tmp_path / "cache.jsonl", batch_size=2)
    r = sc.score(["아주 좋다", "나쁘다", "아주 좋다"])
    assert r["sentiment"].tolist() == [2, -1, 2]                   # 범위(−2~+2)로 자름
    n_calls = len(calls)
    sc2 = LLMScorer(fake_complete, model_tag="fake", cache_path=tmp_path / "cache.jsonl", batch_size=2)
    sc2.score(["아주 좋다", "나쁘다"])
    assert len(calls) == n_calls                                   # 캐시로 재호출 없음


def test_llm_scorer_survives_bad_output():
    sc = LLMScorer(lambda s, u: "죄송합니다, 모르겠습니다", model_tag="bad")
    r = sc.score(["문장"])
    assert r["sentiment"].isna().all()


def test_score_mentions_llm_marks_forward():
    m = pd.DataFrame([{"video_id": "v", "month_kst": "2018-10", "guest": False, "sent_idx": 0,
                       "sentence": "삼성전자는 오를 것입니다", "target_type": "stock", "target": "A1",
                       "target_name": "삼성전자", "is_forward": True}])
    sc = LLMScorer(lambda s, u: json.dumps([{"id": 0, "sentiment": 2, "uncertainty": 1, "forward": False}]),
                   model_tag="x")
    out = score_mentions(m, "llm", sc)
    assert out.loc[0, "llm_sentiment"] == 2 and not out.loc[0, "is_forward_final"]


# ---------------------------------------------------------------- stats
def test_ols_matches_known_line():
    x = np.arange(20, dtype=float)
    y = 2.0 + 0.5 * x
    res = ols(y + np.random.default_rng(0).normal(0, 1e-9, 20), x)
    assert res.alpha == pytest.approx(2.0, abs=1e-6) and res.coef[1] == pytest.approx(0.5, abs=1e-6)


def test_ols_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    x = rng.normal(size=80)
    y = 0.3 + 0.8 * x + rng.normal(size=80)
    mine = ols(y, x, hac_lags=0)
    ref = sm.OLS(y, sm.add_constant(x)).fit()
    np.testing.assert_allclose(mine.coef, ref.params, rtol=1e-8)
    np.testing.assert_allclose(mine.se, ref.bse, rtol=1e-8)
    mine_hac = ols(y, x, hac_lags=3)
    ref_hac = sm.OLS(y, sm.add_constant(x)).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": True})
    np.testing.assert_allclose(mine_hac.se, ref_hac.bse, rtol=1e-6)


# ---------------------------------------------------------------- portfolio
def _panel_and_ff(n_months=60, seed=0):
    rng = np.random.default_rng(seed)
    months = ym_range(201801, ym_add(201801, n_months - 1))
    ff = pd.DataFrame({"ym": months, "mkt_rf": rng.normal(0.005, 0.04, len(months))})
    rows = []
    for m in months:
        for i in range(10):
            rows.append({"code": f"C{i}", "name": f"종목{i}", "ym": m, "ex": rng.normal(0.0, 0.05),
                         "size": 1e12})
    return pd.DataFrame(rows), ff


def test_portfolio_uses_next_month_and_hl_sign():
    panel, ff = _panel_and_ff(6)
    # 2018-01 에 C0 는 P, C1 은 N. 수익률은 2018-02 것을 써야 함
    panel.loc[(panel.code == "C0") & (panel.ym == 201802), "ex"] = 0.10
    panel.loc[(panel.code == "C1") & (panel.ym == 201802), "ex"] = -0.10
    panel.loc[(panel.code == "C0") & (panel.ym == 201801), "ex"] = -0.50   # 같은 달 값은 쓰면 안 됨
    tone = pd.DataFrame({"ym": [201801, 201801], "code": ["C0", "C1"], "tone": [0.3, -0.3]})
    groups = assign_groups(tone, "sign", 0.0)
    pr = portfolio_returns(groups, panel, ff, horizon=1, min_per_group=1)
    assert len(pr) == 1
    assert pr.loc[0, "ym_ret"] == 201802
    assert pr.loc[0, "r_P"] == pytest.approx(0.10) and pr.loc[0, "hl"] == pytest.approx(0.20)


def test_min_per_group_drops_thin_months():
    panel, ff = _panel_and_ff(6)
    tone = pd.DataFrame({"ym": [201801] * 3, "code": ["C0", "C1", "C2"], "tone": [0.3, 0.2, -0.3]})
    groups = assign_groups(tone)
    assert len(portfolio_returns(groups, panel, ff, min_per_group=2)) == 0
    assert len(portfolio_returns(groups, panel, ff, min_per_group=1)) == 1


def test_alpha_detects_planted_effect():
    rng = np.random.default_rng(3)
    months = ym_range(201801, ym_add(201801, 119))
    ff = pd.DataFrame({"ym": months, "mkt_rf": rng.normal(0.005, 0.04, len(months))})
    rows, tone_rows = [], []
    for k, m in enumerate(months[:-1]):
        for i in range(8):
            sign = 1 if i < 4 else -1
            tone_rows.append({"ym": m, "code": f"C{i}", "tone": 0.1 * sign})
    for m in months:
        mk = ff.set_index("ym").loc[m, "mkt_rf"]
        for i in range(8):
            sign = 1 if i < 4 else -1
            rows.append({"code": f"C{i}", "name": f"s{i}", "ym": m, "size": 1.0,
                         "ex": mk + rng.normal(0, 0.03) - 0.01 * sign})   # 긍정 종목이 월 1% 덜 오름
    pr = portfolio_returns(assign_groups(pd.DataFrame(tone_rows)), pd.DataFrame(rows), ff)
    res = alpha_tests(pr, hac_lags=3).iloc[0]
    assert res["alpha"] == pytest.approx(-0.02, abs=0.01) and res["p_alpha"] < 0.01


# ---------------------------------------------------------------- decision
def test_decision_rules():
    assert decide(-0.03, -0.05, -0.01, 0.01, 60).label == "지지"
    assert decide(-0.03, -0.05, -0.01, 0.01, 60, pretrend_p=0.01).label.startswith("지지 보류")
    assert decide(0.03, 0.01, 0.05, 0.01, 60).label == "반대 방향"
    assert decide(-0.002, -0.01, 0.006, 0.6, 60, meaningful=0.02).label.startswith("기각")
    assert decide(-0.01, -0.05, 0.03, 0.6, 60, meaningful=0.02).label == "판정 불가"
    assert decide(-0.03, -0.05, -0.01, 0.01, 10, min_months=24).label == "판정 불가"


def test_sentence_table_skips_failed_transcripts():
    v = pd.DataFrame([
        {"video_id": "a", "title": "t", "published_at": "2024-01-05T00:00:00Z", "transcript": "안녕하세요. 반갑습니다."},
        {"video_id": "b", "title": "t", "published_at": "2024-01-06T00:00:00Z", "transcript": float("nan")},
    ])
    st = build_sentence_table(v)
    assert set(st["video_id"]) == {"a"} and len(st) == 2
