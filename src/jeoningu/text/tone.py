"""톤 수치화 (수업 week 9, Case 4, sentiment 실습).

세 가지 방법을 제공합니다.
  1) dictionary : 한국어 톤 사전으로 단어를 세는 방식. 점수 = 해당 단어 수 / 전체 단어 수 (Case 4)
  2) llm        : 프롬프트에 척도와 예시를 주고 문장별로 채점 (week 9 'AI 활용'). 주 방법.
  3) vader / finbert : 문장을 영어로 번역한 뒤 적용하는 교차 검증용 (sentiment 실습)

집계는 (월, 대상) 단위입니다. 사전 방식은 단어 수를 합쳐서 비율을 구하고(Case 4와 같은 정의),
LLM 방식은 문장 점수의 평균(−2~+2)을 씁니다. 두 방식 모두 부호가 '긍정/부정'을 뜻합니다.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..months import month_str_to_ym
from . import lexicon_ko as lex

CompleteFn = Callable[[str, str], str]  # (system, user) -> 모델 응답 텍스트
PROMPT_VERSION = "v1"

# ------------------------------------------------------------------ 1) 사전 방식


def _count(text: str, words: list[str]) -> int:
    return sum(text.count(w) for w in words)


def score_dictionary(sentences: pd.Series) -> pd.DataFrame:
    """문장별 단어 수와 사전 단어 수를 셉니다. (비율은 집계 단계에서 합산 후 계산)"""
    rows = []
    for t in sentences.astype(str):
        rows.append({
            "n_words": len(t.split()),
            "pos": _count(t, lex.POSITIVE),
            "neg": _count(t, lex.NEGATIVE),
            "unc": _count(t, lex.UNCERTAINTY),
            "weak": _count(t, lex.WEAK_MODAL),
            "strong": _count(t, lex.STRONG_MODAL),
        })
    return pd.DataFrame(rows, index=sentences.index)


# ------------------------------------------------------------------ 2) LLM 방식

SYSTEM_PROMPT = """당신은 한국 경제 유튜브 스크립트의 문장을 채점하는 연구 보조입니다.
각 문장에 대해 다음 세 값을 정하세요.

- sentiment: 문장이 말하는 대상(종목, 시장)에 대한 전망의 방향. 정수 -2~+2
  (-2 매우 부정, -1 부정, 0 중립, +1 긍정, +2 매우 긍정)
- uncertainty: 불확실성 표현의 정도. 정수 0~2 (0 없음, 1 보통, 2 높음)
- forward: 앞으로의 일에 대한 전망·예측·권유 문장이면 true, 이미 일어난 일을 설명하는 문장이면 false

규칙
- 이후 시장이 실제로 어떻게 됐는지는 모른다고 가정하세요. 문장의 표현만으로 판단합니다.
- 반드시 JSON 배열만 출력하세요. 다른 글은 쓰지 마세요.
- 입력의 id 를 그대로 돌려주세요.

예시
입력: [{"id":0,"text":"삼성전자는 앞으로 크게 오를 가능성이 높습니다."},{"id":1,"text":"어제 코스피는 3% 하락했습니다."},{"id":2,"text":"이 부분은 아직 잘 모르겠습니다."}]
출력: [{"id":0,"sentiment":2,"uncertainty":1,"forward":true},{"id":1,"sentiment":-1,"uncertainty":0,"forward":false},{"id":2,"sentiment":0,"uncertainty":2,"forward":false}]
"""

TRANSLATE_PROMPT = """Translate each Korean sentence into natural English, keeping its tone (positive/negative/uncertain) intact.
Return only a JSON array of objects like [{"id":0,"en":"..."}]. Keep the same ids."""


def make_complete_fn(provider: str, model: str, temperature: float | None = None,
                     max_tokens: int = 4000) -> CompleteFn:
    """공급자별 호출 함수를 만듭니다. SDK 는 쓸 때만 불러옵니다(pip install .[llm])."""
    if not model:
        raise ValueError("config 의 tone.llm_model 에 모델 ID 를 적어 주세요.")
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()

        def complete(system: str, user: str) -> str:
            kwargs = {"model": model, "max_tokens": max_tokens, "system": system,
                      "messages": [{"role": "user", "content": user}]}
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.messages.create(**kwargs)
            return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

        return complete
    if provider == "openai":
        from openai import OpenAI

        client = OpenAI()

        def complete(system: str, user: str) -> str:
            kwargs = {"model": model,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
            if temperature is not None:
                kwargs["temperature"] = temperature
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""

        return complete
    raise ValueError(f"지원하지 않는 공급자: {provider}")


def _extract_json_array(text: str):
    m = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not m:
        raise ValueError("응답에서 JSON 배열을 찾지 못했습니다.")
    return json.loads(m.group(0))


class JsonlCache:
    """한 줄에 하나씩 {key, value} 를 쌓는 단순 캐시. 같은 문장을 다시 채점하지 않기 위함."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.data: dict[str, dict] = {}
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["value"]

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value: dict) -> None:
        self.data[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def _key(kind: str, text: str, model_tag: str) -> str:
    return hashlib.sha1(f"{kind}|{PROMPT_VERSION}|{model_tag}|{text}".encode("utf-8")).hexdigest()


class LLMScorer:
    """문장을 배치로 묶어 LLM 에 채점시키고 결과를 캐시합니다."""

    def __init__(self, complete_fn: CompleteFn, model_tag: str = "model", cache_path=None,
                 batch_size: int = 20):
        self.complete = complete_fn
        self.model_tag = model_tag
        self.cache = JsonlCache(cache_path)
        self.batch_size = batch_size

    def _run_batch(self, texts: list[str]) -> list[dict] | None:
        user = json.dumps([{"id": i, "text": t} for i, t in enumerate(texts)], ensure_ascii=False)
        for _ in range(2):  # 형식 오류가 나면 한 번 더 시도
            try:
                arr = _extract_json_array(self.complete(SYSTEM_PROMPT, user))
                by_id = {int(a["id"]): a for a in arr}
                out = []
                for i in range(len(texts)):
                    a = by_id[i]
                    out.append({
                        "sentiment": int(np.clip(int(a["sentiment"]), -2, 2)),
                        "uncertainty": int(np.clip(int(a["uncertainty"]), 0, 2)),
                        "forward": bool(a["forward"]),
                    })
                return out
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        return None

    def score(self, sentences: list[str]) -> pd.DataFrame:
        """입력 순서대로 sentiment, uncertainty, forward 를 돌려줍니다(실패한 문장은 NaN)."""
        unique = list(dict.fromkeys(sentences))
        todo = [s for s in unique if self.cache.get(_key("score", s, self.model_tag)) is None]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            result = self._run_batch(batch)
            if result is None:
                continue
            for s, r in zip(batch, result):
                self.cache.put(_key("score", s, self.model_tag), r)
        rows = []
        for s in sentences:
            r = self.cache.get(_key("score", s, self.model_tag))
            rows.append(r or {"sentiment": np.nan, "uncertainty": np.nan, "forward": np.nan})
        return pd.DataFrame(rows)

    def translate(self, sentences: list[str]) -> list[str | None]:
        """한국어 문장을 영어로 번역합니다(VADER, FinBERT 교차 검증용)."""
        unique = list(dict.fromkeys(sentences))
        todo = [s for s in unique if self.cache.get(_key("tr", s, self.model_tag)) is None]
        for i in range(0, len(todo), self.batch_size):
            batch = todo[i:i + self.batch_size]
            user = json.dumps([{"id": j, "text": t} for j, t in enumerate(batch)], ensure_ascii=False)
            try:
                arr = _extract_json_array(self.complete(TRANSLATE_PROMPT, user))
                by_id = {int(a["id"]): str(a["en"]) for a in arr}
                for j, s in enumerate(batch):
                    if j in by_id:
                        self.cache.put(_key("tr", s, self.model_tag), {"en": by_id[j]})
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        out = []
        for s in sentences:
            r = self.cache.get(_key("tr", s, self.model_tag))
            out.append(r["en"] if r else None)
        return out


# ------------------------------------------------------------------ 3) 영어 도구 (번역본에 적용)


def vader_compound(english: list[str]) -> list[float]:
    """VADER compound 점수(−1~+1). nltk 와 vader_lexicon 이 필요합니다."""
    try:
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
    except (ImportError, LookupError) as exc:
        raise RuntimeError("pip install nltk 후 nltk.download('vader_lexicon') 를 실행하세요.") from exc
    return [sia.polarity_scores(t)["compound"] if t else np.nan for t in english]


def finbert_net(english: list[str]) -> list[float]:
    """FinBERT 의 positive − negative 확률. transformers 와 torch 가 필요합니다."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("pip install transformers torch 가 필요합니다.") from exc
    fb = pipeline("text-classification", model="ProsusAI/finbert", tokenizer="ProsusAI/finbert",
                  top_k=None, truncation=True)
    out = []
    for t in english:
        if not t:
            out.append(np.nan)
            continue
        scores = {d["label"]: d["score"] for d in fb(t)[0]}
        out.append(scores["positive"] - scores["negative"])
    return out


# ------------------------------------------------------------------ 언급 표 채점과 집계


def score_mentions(mentions: pd.DataFrame, method: str = "dictionary",
                   scorer: LLMScorer | None = None) -> pd.DataFrame:
    """언급 문장 표에 점수 열을 붙입니다. 'is_forward_final' 은 이후 필터에 쓰는 전망 여부입니다."""
    out = mentions.reset_index(drop=True).copy()
    if method == "dictionary":
        out = pd.concat([out, score_dictionary(out["sentence"])], axis=1)
        out["is_forward_final"] = out["is_forward"]
    elif method == "llm":
        if scorer is None:
            raise ValueError("method='llm' 이면 LLMScorer 가 필요합니다.")
        res = scorer.score(out["sentence"].tolist())
        out["llm_sentiment"] = res["sentiment"]
        out["llm_uncertainty"] = res["uncertainty"]
        out["is_forward_final"] = res["forward"].astype("boolean").fillna(False).astype(bool)
        out = pd.concat([out, score_dictionary(out["sentence"])], axis=1)  # 비교용으로 사전 점수도 보관
    else:
        raise ValueError(f"알 수 없는 method: {method}")
    return out


def aggregate_tone(scored: pd.DataFrame, method: str = "dictionary", forward_only: bool = True) -> pd.DataFrame:
    """(월, 대상) 단위의 톤을 만듭니다.

    반환 열: ym, target_type, target, target_name, tone, uncertainty, strong_modal, n_sent, n_videos
    """
    df = scored.copy()
    if forward_only:
        df = df[df["is_forward_final"]]
    if df.empty:
        return pd.DataFrame(columns=["ym", "target_type", "target", "target_name", "tone", "uncertainty",
                                     "strong_modal", "n_sent", "n_videos"])
    df["ym"] = df["month_kst"].map(month_str_to_ym)
    keys = ["ym", "target_type", "target", "target_name"]
    g = df.groupby(keys)
    out = g.agg(n_sent=("sentence", "size"), n_videos=("video_id", "nunique")).reset_index()
    if method == "dictionary":
        s = g[["pos", "neg", "unc", "strong", "n_words"]].sum().reset_index()
        w = s["n_words"].replace(0, np.nan)
        s["tone"] = (s["pos"] - s["neg"]) / w
        s["uncertainty"] = s["unc"] / w
        s["strong_modal"] = s["strong"] / w
        out = out.merge(s[keys + ["tone", "uncertainty", "strong_modal"]], on=keys)
    else:
        s = g.agg(tone=("llm_sentiment", "mean"), uncertainty=("llm_uncertainty", "mean")).reset_index()
        w = g[["strong", "n_words"]].sum().reset_index()
        w["strong_modal"] = w["strong"] / w["n_words"].replace(0, np.nan)
        out = out.merge(s, on=keys).merge(w[keys + ["strong_modal"]], on=keys)
    return out.dropna(subset=["tone"]).reset_index(drop=True)
