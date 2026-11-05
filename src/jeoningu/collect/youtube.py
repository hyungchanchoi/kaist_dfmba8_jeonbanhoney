"""YouTube 채널의 영상 목록, 메타데이터, 자막을 수집합니다.

기존 노트북(youtube_channel_data.ipynb)의 수집 흐름을 그대로 옮기되, 전체 기간 수집에 맞게
아래를 더했습니다.
  - 기간 필터: 업로드 재생목록은 최신순이므로 start_date 보다 오래된 영상이 나오면 멈춥니다.
  - 이어받기: 이미 저장된 video_id 는 건너뜁니다.
  - 체크포인트: checkpoint_every 개마다 CSV 에 이어 씁니다(중간에 끊겨도 잃지 않음).
  - 차단 대응: 자막 요청 사이에 대기하고, IP 차단이 감지되면 안전하게 멈춥니다.

API 키는 환경변수 YOUTUBE_API_KEY 또는 --key-file 로 받습니다. 코드에 키를 쓰지 마세요.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Callable, Iterator

import pandas as pd
import requests

BASE_URL = "https://www.googleapis.com/youtube/v3"

COLUMNS = [
    "video_id", "title", "published_at", "channel_title", "description",
    "view_count", "like_count", "comment_count", "duration", "url",
    "transcript", "transcript_language", "transcript_generated",
    "transcript_status", "transcript_error",
]


class TranscriptBlocked(RuntimeError):
    """YouTube 가 자막 요청을 막았을 때. 잠시 뒤 같은 명령을 다시 실행하면 이어받습니다."""


def resolve_api_key(env_var: str = "YOUTUBE_API_KEY", key_file: str | Path | None = None) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key and key_file:
        key = Path(key_file).read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError(
            f"API 키가 없습니다. 환경변수 {env_var} 를 설정하거나 --key-file 로 키 파일 경로를 주세요."
        )
    return key


def _api_get(resource: str, key: str, session: requests.Session | None = None,
             retries: int = 3, **params) -> dict:
    http = session or requests
    last: Exception | None = None
    for attempt in range(retries):
        try:
            resp = http.get(f"{BASE_URL}/{resource}", params={**params, "key": key}, timeout=30)
            if resp.status_code >= 500:
                raise RuntimeError(f"서버 오류 {resp.status_code}")
            data = resp.json()
            if not resp.ok:
                message = data.get("error", {}).get("message", resp.text)
                raise RuntimeError(f"YouTube API 오류 ({resp.status_code}): {message}")
            return data
        except (requests.RequestException, RuntimeError) as exc:
            last = exc
            if isinstance(exc, RuntimeError) and "서버 오류" not in str(exc):
                raise  # 4xx(키 오류, 쿼터 초과 등)는 재시도해도 소용없음
            time.sleep(2 ** attempt)
    raise RuntimeError(f"YouTube API 요청 실패: {last}")


def get_channel(channel: str, key: str, session: requests.Session | None = None) -> dict:
    """채널 ID(UC...), @handle, 채널 URL 중 무엇이든 받아서 채널 정보를 돌려줍니다."""
    value = channel.strip()
    m = re.search(r"UC[A-Za-z0-9_-]{22}", value)
    if m:
        result = _api_get("channels", key, session, part="snippet,contentDetails", id=m.group(0))
    else:
        handle_match = re.search(r"@([^/?\s]+)", value)
        handle = handle_match.group(1) if handle_match else value.lstrip("@")
        result = _api_get("channels", key, session, part="snippet,contentDetails", forHandle=handle)
    items = result.get("items", [])
    if not items:
        raise ValueError("채널을 찾을 수 없습니다. UC... 채널 ID 또는 @handle 을 확인하세요.")
    return items[0]


def iter_upload_video_ids(playlist_id: str, key: str, start_date: str | None = None,
                          end_date: str | None = None, max_videos: int = 0,
                          session: requests.Session | None = None) -> Iterator[tuple[str, str]]:
    """업로드 재생목록을 최신순으로 훑으며 (video_id, published_at) 을 내보냅니다."""
    start = pd.Timestamp(start_date, tz="UTC") if start_date else None
    end = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) if end_date else None
    seen = 0
    page_token = None
    while True:
        params = {"part": "contentDetails", "playlistId": playlist_id, "maxResults": 50}
        if page_token:
            params["pageToken"] = page_token
        page = _api_get("playlistItems", key, session, **params)
        for item in page.get("items", []):
            cd = item.get("contentDetails", {})
            vid, published = cd.get("videoId"), cd.get("videoPublishedAt")
            if not vid or not published:
                continue
            ts = pd.Timestamp(published)
            if start is not None and ts < start:
                return  # 최신순이므로 이후는 모두 더 오래됨
            if end is not None and ts >= end:
                continue
            yield vid, published
            seen += 1
            if max_videos and seen >= max_videos:
                return
        page_token = page.get("nextPageToken")
        if not page_token:
            return


def get_video_metadata(ids: list[str], key: str, session: requests.Session | None = None) -> dict[str, dict]:
    videos: dict[str, dict] = {}
    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]
        result = _api_get("videos", key, session, part="snippet,statistics,contentDetails",
                          id=",".join(batch))
        for item in result.get("items", []):
            snip, stats = item.get("snippet", {}), item.get("statistics", {})
            videos[item["id"]] = {
                "video_id": item["id"],
                "title": snip.get("title"),
                "published_at": snip.get("publishedAt"),
                "channel_title": snip.get("channelTitle"),
                "description": snip.get("description"),
                "view_count": stats.get("viewCount"),
                "like_count": stats.get("likeCount"),
                "comment_count": stats.get("commentCount"),
                "duration": item.get("contentDetails", {}).get("duration"),
                "url": f"https://www.youtube.com/watch?v={item['id']}",
            }
    return videos


def fetch_transcript(api, video_id: str, languages: list[str], retries: int = 2,
                     sleep_seconds: float = 1.0) -> dict:
    """자막 하나를 가져옵니다. 자막이 없는 영상은 status 에 사유를 남기고 계속 진행합니다."""
    last_error = ""
    for attempt in range(retries + 1):
        try:
            fetched = api.fetch(video_id, languages=languages)
            return {
                "transcript": "\n".join(s.text for s in fetched),
                "transcript_language": fetched.language_code,
                "transcript_generated": fetched.is_generated,
                "transcript_status": "success",
                "transcript_error": "",
            }
        except Exception as exc:  # noqa: BLE001 - 라이브러리마다 예외 종류가 많음
            name = type(exc).__name__
            if "Blocked" in name:
                raise TranscriptBlocked(
                    f"{name}: YouTube 가 요청을 막았습니다. 잠시 후 같은 명령을 다시 실행하세요."
                ) from exc
            last_error = f"{name}: {exc}"
            if "NotFound" in name or "Disabled" in name or "Unavailable" in name:
                break  # 재시도해도 같은 결과
            time.sleep(sleep_seconds * (attempt + 1))
    return {
        "transcript": "", "transcript_language": "", "transcript_generated": None,
        "transcript_status": "unavailable_or_error", "transcript_error": last_error[:300],
    }


def collect_channel(channel_id: str, key: str, out_csv: str | Path, start_date: str | None,
                    end_date: str | None, max_videos: int = 0,
                    languages: list[str] | None = None, sleep_seconds: float = 1.0,
                    checkpoint_every: int = 20, max_retries: int = 2,
                    transcript_api=None, session: requests.Session | None = None,
                    log: Callable[[str], None] = print) -> pd.DataFrame:
    """채널의 기간 안 영상과 자막을 모두 수집해 CSV 로 저장합니다(이어받기 지원)."""
    languages = languages or ["ko", "en"]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    channel = get_channel(channel_id, key, session)
    playlist = channel["contentDetails"]["relatedPlaylists"]["uploads"]
    log(f"채널: {channel['snippet']['title']} ({channel['id']})")

    done: set[str] = set()
    if out_csv.exists():
        done = set(pd.read_csv(out_csv, usecols=["video_id"], encoding="utf-8-sig")["video_id"])
        log(f"이미 수집된 영상 {len(done)}개는 건너뜁니다.")

    ids = [v for v, _ in iter_upload_video_ids(playlist, key, start_date, end_date, max_videos, session)]
    todo = [v for v in ids if v not in done]
    log(f"기간 안 영상 {len(ids)}개 중 새로 수집할 영상 {len(todo)}개")

    if transcript_api is None:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_api = YouTubeTranscriptApi()

    n_ok = 0
    for start in range(0, len(todo), checkpoint_every):
        chunk = todo[start:start + checkpoint_every]
        meta = get_video_metadata(chunk, key, session)
        rows = []
        try:
            for i, vid in enumerate(chunk, start=start + 1):
                row = dict(meta.get(vid, {"video_id": vid}))
                row.update(fetch_transcript(transcript_api, vid, languages, max_retries, sleep_seconds))
                rows.append(row)
                n_ok += row["transcript_status"] == "success"
                log(f"[{i}/{len(todo)}] {row.get('title', vid)} — {row['transcript_status']}")
                time.sleep(sleep_seconds)
        finally:
            if rows:  # 차단으로 중단돼도 지금까지 받은 것은 저장
                chunk_df = pd.DataFrame(rows).reindex(columns=COLUMNS)
                chunk_df.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False,
                                encoding="utf-8-sig")
    log(f"저장 완료: {out_csv} (이번 실행 자막 성공 {n_ok}/{len(todo)})")
    return pd.read_csv(out_csv, encoding="utf-8-sig")
