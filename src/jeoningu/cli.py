"""명령줄 진입점.  python -m jeoningu <명령>   (또는 pip install -e . 후 jeoningu <명령>)

단계:  collect → preprocess → mentions → tone → analyze → report   (한 번에: all)
점검:  demo (합성 데이터), label-sample / validate (사람 라벨 검증)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import Config, load_config


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장: {path} ({len(df):,}행)")


# ------------------------------------------------------------------ 단계별 함수


def cmd_collect(cfg: Config, args) -> None:
    from .collect import collect_channel, resolve_api_key

    c = cfg["collect"]
    key = resolve_api_key(key_file=args.key_file)
    collect_channel(
        cfg["channel"]["id"], key, cfg.path("raw_videos"), c["start_date"], c["end_date"],
        max_videos=args.max_videos if args.max_videos is not None else c["max_videos"],
        languages=c["transcript_languages"], sleep_seconds=c["sleep_seconds"],
        checkpoint_every=c["checkpoint_every"], max_retries=c["max_retries"])


def _load_videos(cfg: Config, path: str | None) -> pd.DataFrame:
    return _read(Path(path) if path else cfg.path("raw_videos"))


def cmd_preprocess(cfg: Config, args) -> None:
    from .text.clean import build_sentence_table

    videos = _load_videos(cfg, args.videos)
    sentences = build_sentence_table(videos)
    n_guest = sentences.drop_duplicates("video_id")["guest"].sum()
    print(f"영상 {sentences['video_id'].nunique()}편 (게스트 영상 {int(n_guest)}편), 문장 {len(sentences):,}개")
    _write(sentences, cfg.path("sentences"))


def _load_stocks(cfg: Config):
    from .returns.data import load_korea_stocks

    return load_korea_stocks(cfg.path("korea_stocks"))


def cmd_mentions(cfg: Config, args) -> None:
    from .text.clean import build_sentence_table  # noqa: F401 (명시적으로 의존 단계 표시)
    from .pipeline import make_mentions

    videos = _load_videos(cfg, args.videos)
    mentions = make_mentions(videos, _load_stocks(cfg), cfg)
    print(mentions.groupby(["target_type", mentions["guest"]]).size().to_string())
    _write(mentions, cfg.path("mentions"))


def cmd_tone(cfg: Config, args) -> None:
    from .pipeline import score

    mentions = _read(cfg.path("mentions"))
    scored = score(mentions, cfg, method=args.method)
    _write(scored, cfg.path("scored"))


def _load_returns(cfg: Config):
    from .returns.data import build_panel, load_ff_korea, load_korea_stocks

    stocks = load_korea_stocks(cfg.path("korea_stocks"))
    ff = load_ff_korea(cfg.path("ff_korea"))
    return build_panel(stocks, ff), ff


def cmd_analyze(cfg: Config, args):
    from .analysis.run import run_analysis
    from .report import write_report

    scored = _read(cfg.path("scored"))
    method = args.method or cfg["tone"]["method"]
    panel, ff = _load_returns(cfg)
    res = run_analysis(scored, panel, ff, sample_cfg=cfg["sample"], tone_method=method,
                       mentions_cfg=cfg["mentions"], analysis_cfg=cfg["analysis"])
    path = write_report(res, cfg.path("outputs"), ff=ff, meaningful=cfg["analysis"]["meaningful_alpha"])
    for k, d in res.decisions.items():
        print(f"{k}: {d.label} — {d.reason}")
    print(f"보고서: {path}")


def cmd_all(cfg: Config, args) -> None:
    cmd_preprocess(cfg, args)
    cmd_mentions(cfg, args)
    cmd_tone(cfg, args)
    cmd_analyze(cfg, args)


def cmd_demo(cfg: Config, args) -> None:
    from .demo import run_demo

    res = run_demo(args.out, args.planted_alpha, args.planted_market, args.seed, cfg)
    for k, d in res.decisions.items():
        print(f"{k}: {d.label} — {d.reason}")
    if len(res.alpha):
        print(res.alpha[["portfolio", "alpha", "t_alpha", "p_alpha", "n"]].to_string(index=False))
    print(f"보고서: {Path(args.out) / 'report.md'}")


def cmd_label_sample(cfg: Config, args) -> None:
    from .analysis.validation import make_label_sample

    scored = _read(cfg.path("scored"))
    make_label_sample(scored, n=args.n, seed=args.seed, out_path=cfg.path("labels"))
    print(f"라벨링 표 저장: {cfg.path('labels')} — label_a, label_b 열에 −2~+2 를 채우세요.")


def cmd_validate(cfg: Config, args) -> None:
    from .analysis.validation import compare_to_human

    scored = _read(cfg.path("scored"))
    labels = _read(cfg.path("labels"))
    col = "llm_sentiment" if "llm_sentiment" in scored else "pos"
    if col == "pos":
        scored["pos_net"] = scored["pos"] - scored["neg"]
        col = "pos_net"
    for k, v in compare_to_human(scored, labels, model_col=col).items():
        print(f"{k}: {v}")


# ------------------------------------------------------------------ 파서


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jeoningu", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="설정 파일 경로(기본: config/config.toml)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("collect", help="유튜브 영상과 자막 수집(YOUTUBE_API_KEY 필요)")
    s.add_argument("--key-file", help="API 키가 든 텍스트 파일(없으면 환경변수 사용)")
    s.add_argument("--max-videos", type=int, help="시험용으로 개수 제한")
    s.set_defaults(fn=cmd_collect)

    for name, fn, h in [("preprocess", cmd_preprocess, "자막 정리와 문장 분리"),
                        ("mentions", cmd_mentions, "종목·시장 언급 문장 추출(수업 Case 1 종목명 필요)"),
                        ("all", cmd_all, "preprocess → mentions → tone → analyze")]:
        s = sub.add_parser(name, help=h)
        s.add_argument("--videos", help="수집 결과 CSV(기본: paths.raw_videos)")
        s.add_argument("--method", choices=["dictionary", "llm"], help="톤 방법(all 에서만 사용)")
        s.set_defaults(fn=fn)

    s = sub.add_parser("tone", help="언급 문장 톤 점수화")
    s.add_argument("--method", choices=["dictionary", "llm"], help="기본: config tone.method")
    s.set_defaults(fn=cmd_tone)

    s = sub.add_parser("analyze", help="H1·H2 분석과 보고서 생성(수업 Case 1 수익률 필요)")
    s.add_argument("--method", choices=["dictionary", "llm"])
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("demo", help="합성 데이터로 전체 파이프라인 점검")
    s.add_argument("--out", default="demo_outputs")
    s.add_argument("--planted-alpha", type=float, default=0.0, help="심어 둘 H−L 알파(예: -0.03)")
    s.add_argument("--planted-market", type=float, default=0.0, help="심어 둘 시장 효과(톤 부호당)")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("label-sample", help="사람 라벨링용 문장 표본 만들기")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_label_sample)

    s = sub.add_parser("validate", help="모델 점수와 사람 라벨의 일치도")
    s.set_defaults(fn=cmd_validate)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):   # 윈도우 콘솔(cp949)에서도 한글·기호가 깨지거나 죽지 않게
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    try:
        args.fn(cfg, args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
