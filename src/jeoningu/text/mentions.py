"""종목·시장 언급 문장 추출 (수업 Case 3: 키워드가 든 문장을 regex 로 뽑아 표로 만들기).

종목 사전은 수업 Case 1 의 종목명(Korea_Stocks_2000_2025.csv)에서 만듭니다. 그래서 언급된 종목이
곧바로 주가 데이터와 연결됩니다. 두 글자 이름은 일반 단어와 겹쳐 오탐이 많아서 제외하고(min_name_len),
긴 이름부터 먼저 맞춥니다("SK하이닉스"가 "하이닉스"보다 우선).
"""
from __future__ import annotations

import re

import pandas as pd

from ..months import month_str_to_ym, ym_range

MARKET_TARGET = "KOSPI"
HANGUL_OR_ALNUM = "가-힣A-Za-z0-9"


def build_stock_dictionary(stocks: pd.DataFrame, start_ym: int, end_ym: int, top_n: int = 300,
                           min_name_len: int = 3, aliases: dict[str, str] | None = None
                           ) -> dict[str, tuple[str, str]]:
    """{표면형 이름: (종목코드, 대표 종목명)} 사전을 만듭니다.

    표본 기간의 어느 달이든 시가총액 상위 top_n 에 든 적이 있는 종목을 모두 포함합니다.
    (예전에 크게 언급되다 상장폐지된 종목이 빠지는 생존 편향을 줄이기 위함)
    """
    s = stocks[(stocks["ym"] >= start_ym) & (stocks["ym"] <= end_ym)].copy()
    s["rank"] = s.groupby("ym")["mcap"].rank(ascending=False, method="first")
    top = s[s["rank"] <= top_n].sort_values("ym")
    # 같은 이름이 여러 코드에 쓰였으면 가장 최근 코드를 씁니다.
    latest = top.drop_duplicates(subset=["name"], keep="last")
    out: dict[str, tuple[str, str]] = {}
    for r in latest.itertuples(index=False):
        name = str(r.name)
        if len(name) >= min_name_len:
            out[name] = (r.code, name)
    for alias, canonical in (aliases or {}).items():
        if canonical in out:
            out[alias] = out[canonical]
    return out


def compile_stock_regex(dictionary: dict[str, tuple[str, str]]) -> re.Pattern[str] | None:
    if not dictionary:
        return None
    names = sorted(dictionary, key=len, reverse=True)  # 긴 이름 우선
    body = "|".join(re.escape(n) for n in names)
    # 앞에 한글/영숫자가 붙어 있으면(다른 단어의 일부) 제외. 뒤에는 조사가 붙을 수 있어 제한하지 않음.
    return re.compile(rf"(?<![{HANGUL_OR_ALNUM}])(?:{body})")


def extract_mentions(sentences: pd.DataFrame, dictionary: dict[str, tuple[str, str]],
                     market_keywords: list[str], forward_pattern: str) -> pd.DataFrame:
    """문장 표에서 종목·시장 언급 행을 만듭니다.

    입력 열: video_id, month_kst, guest, sent_idx, sentence
    반환 열: video_id, month_kst, guest, sent_idx, sentence, target_type('stock'|'market'),
             target(종목코드 또는 'KOSPI'), target_name, is_forward
    한 문장에 종목이 여러 개면 종목마다 한 행을 만듭니다.
    """
    stock_re = compile_stock_regex(dictionary)
    market_re = re.compile("|".join(re.escape(k) for k in market_keywords)) if market_keywords else None
    fwd_re = re.compile(forward_pattern)

    rows = []
    for r in sentences.itertuples(index=False):
        sent = r.sentence
        is_fwd = bool(fwd_re.search(sent))
        base = (r.video_id, r.month_kst, r.guest, r.sent_idx, sent)
        if stock_re is not None:
            found = {m.group(0) for m in stock_re.finditer(sent)}
            seen_codes = set()
            for name in found:
                code, canon = dictionary[name]
                if code in seen_codes:
                    continue
                seen_codes.add(code)
                rows.append((*base, "stock", code, canon, is_fwd))
        if market_re is not None and market_re.search(sent):
            rows.append((*base, "market", MARKET_TARGET, "코스피(시장)", is_fwd))
    cols = ["video_id", "month_kst", "guest", "sent_idx", "sentence",
            "target_type", "target", "target_name", "is_forward"]
    return pd.DataFrame(rows, columns=cols)


def filter_sample(mentions: pd.DataFrame, start_month: str, end_month: str,
                  exclude_guest: bool = True, forward_only: bool = True) -> pd.DataFrame:
    """분석 표본 조건(기간, 게스트 제외, 전망 문장만)을 적용합니다."""
    start_ym, end_ym = month_str_to_ym(start_month), month_str_to_ym(end_month)
    ym = mentions["month_kst"].map(month_str_to_ym)
    m = mentions[(ym >= start_ym) & (ym <= end_ym)]
    if exclude_guest:
        m = m[~m["guest"]]
    if forward_only:
        m = m[m["is_forward"]]
    return m.reset_index(drop=True)


def months_in_sample(start_month: str, end_month: str) -> list[int]:
    return ym_range(month_str_to_ym(start_month), month_str_to_ym(end_month))
