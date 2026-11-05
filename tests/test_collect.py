"""수집 모듈 테스트: 실제 API 대신 가짜 응답을 씁니다(키와 네트워크 불필요)."""
from types import SimpleNamespace

import pandas as pd
import pytest

from jeoningu.collect import youtube as yt


class FakeSession:
    """YouTube Data API 응답을 흉내 냅니다."""

    def __init__(self, items):
        self.items = items          # [(video_id, published_at)] 최신순
        self.calls = []

    def get(self, url, params, timeout):
        resource = url.rsplit("/", 1)[-1]
        self.calls.append((resource, dict(params)))
        if resource == "channels":
            body = {"items": [{"id": "UC" + "a" * 22, "snippet": {"title": "테스트채널"},
                               "contentDetails": {"relatedPlaylists": {"uploads": "UUplaylist"}}}]}
        elif resource == "playlistItems":
            start = int(params.get("pageToken", 0) or 0)
            page = self.items[start:start + 50]
            body = {"items": [{"contentDetails": {"videoId": v, "videoPublishedAt": p}} for v, p in page]}
            if start + 50 < len(self.items):
                body["nextPageToken"] = str(start + 50)
        elif resource == "videos":
            ids = params["id"].split(",")
            body = {"items": [{"id": i, "snippet": {"title": f"제목 {i}", "publishedAt": dict(self.items)[i],
                                                    "channelTitle": "테스트채널", "description": "d"},
                               "statistics": {"viewCount": "10", "likeCount": "1", "commentCount": "0"},
                               "contentDetails": {"duration": "PT10M"}} for i in ids]}
        else:
            raise AssertionError(resource)
        return SimpleNamespace(status_code=200, ok=True, json=lambda: body, text="")


class FakeTranscriptApi:
    def __init__(self, fail=()):
        self.fail = set(fail)

    def fetch(self, vid, languages):
        if vid in self.fail:
            raise type("NoTranscriptFound", (Exception,), {})("없음")
        return _Fetched([SimpleNamespace(text="안녕하세요"), SimpleNamespace(text="코스피 이야기입니다")])


class _Fetched(list):
    language_code = "ko"
    is_generated = True


ITEMS = [("v5", "2025-06-01T00:00:00Z"), ("v4", "2025-01-10T00:00:00Z"), ("v3", "2024-05-01T00:00:00Z"),
         ("v2", "2018-10-01T00:00:00Z"), ("v1", "2018-08-01T00:00:00Z")]


def test_iter_stops_at_start_date_and_skips_after_end():
    s = FakeSession(ITEMS)
    got = list(yt.iter_upload_video_ids("UUplaylist", "k", "2018-09-01", "2025-03-31", session=s))
    assert [v for v, _ in got] == ["v4", "v3", "v2"]      # v5 는 기간 이후, v1 은 개설 이전


def test_iter_respects_max_videos():
    s = FakeSession(ITEMS)
    assert len(list(yt.iter_upload_video_ids("UUplaylist", "k", None, None, max_videos=2, session=s))) == 2


def test_collect_resumes_and_checkpoints(tmp_path):
    out = tmp_path / "videos.csv"
    s = FakeSession(ITEMS)
    api = FakeTranscriptApi(fail={"v3"})
    df = yt.collect_channel("UC" + "a" * 22, "k", out, "2018-09-01", "2025-03-31", transcript_api=api,
                            session=s, sleep_seconds=0, checkpoint_every=2, log=lambda m: None)
    assert list(df.columns) == yt.COLUMNS
    assert set(df["video_id"]) == {"v4", "v3", "v2"}
    status = dict(zip(df["video_id"], df["transcript_status"]))
    assert status["v3"] == "unavailable_or_error" and status["v4"] == "success"

    # 다시 실행하면 새 영상이 없으므로 아무것도 추가되지 않음(이어받기)
    df2 = yt.collect_channel("UC" + "a" * 22, "k", out, "2018-09-01", "2025-03-31", transcript_api=api,
                             session=s, sleep_seconds=0, checkpoint_every=2, log=lambda m: None)
    assert len(df2) == 3


def test_blocked_stops_but_keeps_progress(tmp_path):
    out = tmp_path / "videos.csv"

    class Blocking(FakeTranscriptApi):
        def fetch(self, vid, languages):
            if vid == "v3":
                raise type("IpBlocked", (Exception,), {})("blocked")
            return super().fetch(vid, languages)

    s = FakeSession(ITEMS)
    with pytest.raises(yt.TranscriptBlocked):
        yt.collect_channel("UC" + "a" * 22, "k", out, "2018-09-01", "2025-03-31", transcript_api=Blocking(),
                           session=s, sleep_seconds=0, checkpoint_every=10, log=lambda m: None)
    saved = pd.read_csv(out, encoding="utf-8-sig")
    assert list(saved["video_id"]) == ["v4"]          # 차단 전까지 받은 것은 남아 있음


def test_missing_api_key_message(monkeypatch):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API 키"):
        yt.resolve_api_key()
