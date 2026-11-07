"""종목·시장 언급 문장 추출 (수업 Case 3: 키워드가 든 문장을 regex 로 뽑아 표로 만들기).

종목 사전은 config/universe.csv(Yahoo 티커, 종목명, 별칭)에서 만듭니다. 그래서 언급된 종목이
곧바로 Yahoo Finance 수익률과 연결됩니다. 이 파일에 없는 종목은 분석에서 빠지므로 영상에 자주 나오는
종목을 PR 로 추가하세요. 두 글자 종목명은 일반 단어와 겹쳐 오탐이 많아서 제외하고(min_name_len),
긴 이름부터 먼저 맞춥니다("SK하이닉스"가 "하이닉스"보다 우선).
"""
from __future__ import annotations

import re

import pandas as pd

from ..months import month_str_to_ym, ym_range

MARKET_TARGET = "KOSPI"
HANGUL_OR_ALNUM = "가-힣A-Za-z0-9"


def build_stock_dictionary(universe: pd.DataFrame, min_name_len: int = 3,
                           aliases: dict[str, str] | None = None) -> dict[str, tuple[str, str]]:
    """{표면형 이름: (Yahoo 티커, 대표 종목명)} 사전을 만듭니다.

    universe 는 config/universe.csv 를 읽은 표(code, name, aliases)입니다.
    - 대표 종목명은 min_name_len 글자 이상일 때만 넣습니다(두 글자 이름은 일반 단어와 겹침).
    - 별칭(universe 의 aliases 열 + 설정의 [mentions.aliases])은 길이 제한 없이 넣습니다.
      별칭은 사람이 의도해서 적는 것이므로 오탐 위험을 감수합니다.
    """
    out: dict[str, tuple[str, str]] = {}
    by_name: dict[str, tuple[str, str]] = {}
    for r in universe.itertuples(index=False):
        entry = (r.code, r.name)
        by_name[r.name] = entry
        if len(r.name) >= min_name_len:
            out[r.name] = entry
        alias_col = getattr(r, "aliases", "") or ""
        for alias in [a.strip() for a in alias_col.split(";") if a.strip()]:
            out[alias] = entry
    for alias, canonical in (aliases or {}).items():
        if canonical in by_name:
            out[alias] = by_name[canonical]
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
