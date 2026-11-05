"""스크립트 정리와 문장 분리 (수업: week 6 전처리, Case 3 regex 정보 추출)."""
from __future__ import annotations

import re

import pandas as pd

TAG_RE = re.compile(r"\[[^\]]*\]")                    # [음악], [박수] 같은 자막 태그
SENT_SPLIT_RE = re.compile(r"(?<=[.?!])\s+")           # 마침표/물음표/느낌표 뒤의 공백
GUEST_RE = re.compile(
    r"ft\.\s*[^)]*?(박사|센터장|기자|교수|대표|이사|연구원|애널리스트|위원|소장|팀장)"
)


def clean_transcript(text: str) -> str:
    """자막 태그를 지우고 줄바꿈과 공백을 한 칸 공백으로 합칩니다."""
    text = TAG_RE.sub(" ", str(text))
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """정리된 텍스트를 문장으로 나눕니다. 한 글자짜리 조각은 버립니다."""
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if len(s.strip()) > 1]


def is_guest_video(title: str) -> bool:
    """제목에 '(ft.OOO 박사)' 같은 게스트 표기가 있으면 True. 정규식 기반이라 수동 확인 권장."""
    return bool(GUEST_RE.search(str(title)))


def to_kst_month(published_at: str) -> str:
    """UTC 업로드 시각을 한국 시간 기준 'YYYY-MM' 으로 바꿉니다(월 경계 오차 방지)."""
    ts = pd.Timestamp(published_at)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("Asia/Seoul").strftime("%Y-%m")


def build_sentence_table(videos: pd.DataFrame) -> pd.DataFrame:
    """영상 표(수집 결과)를 문장 단위 표로 바꿉니다.

    반환 열: video_id, month_kst, guest, sent_idx, sentence
    자막 수집에 실패한 영상(빈 스크립트)은 제외됩니다.
    """
    rows = []
    for r in videos.itertuples(index=False):
        text = clean_transcript(getattr(r, "transcript", "") if isinstance(getattr(r, "transcript", ""), str) else "")
        if not text:
            continue
        month = to_kst_month(r.published_at)
        guest = is_guest_video(r.title)
        for i, sent in enumerate(split_sentences(text)):
            rows.append((r.video_id, month, guest, i, sent))
    return pd.DataFrame(rows, columns=["video_id", "month_kst", "guest", "sent_idx", "sentence"])
