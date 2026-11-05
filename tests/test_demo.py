"""합성 데이터 종단 점검: 심어 둔 효과를 찾아내고, 심지 않으면 찾지 않아야 합니다."""
import pytest

from jeoningu.config import load_config
from jeoningu.demo import make_synthetic, run_demo


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_pipeline_recovers_planted_alpha(cfg):
    res = run_demo(planted_alpha=-0.03, seed=1, cfg=cfg, write=False)
    hl = res.alpha.iloc[0]
    assert hl["alpha"] == pytest.approx(-0.03, abs=0.012)
    assert hl["p_alpha"] < 0.05
    assert res.decisions["H1 (종목 수준)"].label == "지지"


def test_pipeline_finds_nothing_without_effect(cfg):
    res = run_demo(planted_alpha=0.0, seed=2, cfg=cfg, write=False)
    hl = res.alpha.iloc[0]
    assert hl["p_alpha"] > 0.01          # 효과가 없으면 유의하지 않아야 함(시드 고정)
    assert res.decisions["H1 (종목 수준)"].label != "지지"


def test_market_effect_is_recovered(cfg):
    res = run_demo(planted_alpha=0.0, planted_market=-0.03, seed=3, cfg=cfg, write=False)
    row = res.market[res.market["month_offset"] == 1].iloc[0]
    assert row["b_per_sd_tone"] < 0 and row["p"] < 0.05


def test_guest_videos_are_excluded(cfg):
    res = run_demo(planted_alpha=0.0, seed=4, cfg=cfg, write=False)
    n_months = 87
    # 단독 영상 4편 × 87개월만 표본에 들어가야 함(게스트 영상 87편 제외)
    assert res.summary["표본에 쓴 영상 수"] <= 4 * n_months


def test_synthetic_data_covers_sample_with_margin(cfg):
    videos, stocks, ff, truth = make_synthetic(planted_alpha=-0.05, seed=5)
    assert len(videos) > 0
    assert stocks["ym"].min() < 201810 and stocks["ym"].max() > 202512   # 앞뒤 한 달씩 여유
