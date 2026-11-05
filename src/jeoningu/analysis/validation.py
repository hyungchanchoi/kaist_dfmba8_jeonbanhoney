"""텍스트 측정의 타당성 검증 (사람 라벨과의 일치도).

흐름: label-sample 로 약 100문장 표본을 뽑고 → 팀원 2명 이상이 같은 문장에 −2~+2 로 라벨링 →
      라벨러 간 일치도(Cohen's kappa)를 먼저 확인 → 모델 점수와 사람 라벨의 상관, 부호 일치율을 계산.
라벨러 간 일치도가 낮으면 모델을 탓하기 전에 척도 정의부터 고쳐야 합니다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_label_sample(scored: pd.DataFrame, n: int = 100, seed: int = 0,
                      out_path: str | Path | None = None) -> pd.DataFrame:
    """전망 문장 중에서 무작위로 n 개를 뽑아 라벨링용 표를 만듭니다."""
    pool = scored[scored["is_forward_final"]] if "is_forward_final" in scored else scored
    sample = pool.sample(n=min(n, len(pool)), random_state=seed)
    out = sample[["video_id", "sent_idx", "target_type", "target_name", "sentence"]].copy()
    out.insert(0, "sentence_id", out["video_id"] + "_" + out["sent_idx"].astype(str) + "_" + out["target_name"])
    out["label_a"] = ""  # -2..+2 (라벨러 A)
    out["label_b"] = ""  # -2..+2 (라벨러 B)
    out = out.reset_index(drop=True)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out


def cohens_kappa(a, b) -> float:
    a, b = pd.Series(a).astype(int), pd.Series(b).astype(int)
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def compare_to_human(scored: pd.DataFrame, labels: pd.DataFrame, model_col: str = "llm_sentiment") -> dict:
    """사람 라벨(label_a, label_b 의 평균)과 모델 점수의 관계를 요약합니다."""
    lab = labels.copy()
    for c in ("label_a", "label_b"):
        lab[c] = pd.to_numeric(lab[c], errors="coerce")
    lab = lab.dropna(subset=["label_a"])
    out: dict = {"n_labeled": int(len(lab))}
    both = lab.dropna(subset=["label_b"])
    if len(both) >= 10:
        out["inter_rater_kappa"] = cohens_kappa(both["label_a"], both["label_b"])
        out["inter_rater_exact"] = float((both["label_a"] == both["label_b"]).mean())
    lab["human"] = lab[["label_a", "label_b"]].mean(axis=1)
    s = scored.copy()
    s["sentence_id"] = s["video_id"] + "_" + s["sent_idx"].astype(str) + "_" + s["target_name"]
    merged = lab.merge(s[["sentence_id", model_col]], on="sentence_id", how="inner").dropna(subset=[model_col])
    out["n_matched"] = int(len(merged))
    if len(merged) >= 10:
        out["spearman"] = float(merged["human"].rank().corr(merged[model_col].rank()))
        nz = merged[(merged["human"] != 0) & (merged[model_col] != 0)]
        out["sign_agreement_nonzero"] = float((np.sign(nz["human"]) == np.sign(nz[model_col])).mean()) if len(nz) else np.nan
    return out
