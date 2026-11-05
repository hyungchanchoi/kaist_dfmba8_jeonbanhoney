"""파이프라인 단계를 메모리 객체로 연결하는 함수들. CLI 와 데모, 테스트가 같은 코드를 씁니다."""
from __future__ import annotations

import pandas as pd

from .config import Config
from .months import month_str_to_ym
from .text.clean import build_sentence_table
from .text.mentions import build_stock_dictionary, extract_mentions
from .text.tone import LLMScorer, make_complete_fn, score_mentions


def make_mentions(videos: pd.DataFrame, stocks: pd.DataFrame, cfg: Config | dict) -> pd.DataFrame:
    """수집한 영상 표 → 문장 표 → 종목·시장 언급 표."""
    data = cfg.data if isinstance(cfg, Config) else cfg
    mcfg, scfg = data["mentions"], data["sample"]
    sentences = build_sentence_table(videos)
    dictionary = build_stock_dictionary(
        stocks, month_str_to_ym(scfg["start_month"]), month_str_to_ym(scfg["end_month"]),
        top_n=mcfg["top_n_stocks"], min_name_len=mcfg["min_name_len"], aliases=mcfg.get("aliases", {}))
    return extract_mentions(sentences, dictionary, mcfg["market_keywords"], mcfg["forward_pattern"])


def make_scorer(cfg: Config | dict) -> LLMScorer:
    data = cfg.data if isinstance(cfg, Config) else cfg
    t = data["tone"]
    complete = make_complete_fn(t["llm_provider"], t["llm_model"])
    cache = None
    if isinstance(cfg, Config) and cfg.has_path("llm_cache"):
        cache = cfg.path("llm_cache")
    return LLMScorer(complete, model_tag=f'{t["llm_provider"]}:{t["llm_model"]}', cache_path=cache,
                     batch_size=int(t.get("llm_batch_size", 20)))


def score(mentions: pd.DataFrame, cfg: Config | dict, scorer: LLMScorer | None = None,
          method: str | None = None) -> pd.DataFrame:
    data = cfg.data if isinstance(cfg, Config) else cfg
    method = method or data["tone"]["method"]
    if method == "llm" and scorer is None:
        scorer = make_scorer(cfg)
    return score_mentions(mentions, method, scorer)
