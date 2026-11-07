"""명령줄 진입점.  python -m jeoningu <명령>   (또는 pip install -e . 후 jeoningu <명령>)

단계:  collect, prices(Yahoo 수익률) → preprocess → mentions → tone → analyze → report   (한 번에: all)
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


def cmd_mentions(cfg: Config, args) -> None:
    from .pipeline import make_mentions
    from .returns.data import load_universe

    videos = _load_videos(cfg, args.videos)
    universe = load_universe(cfg.path("universe"))
    print(f"종목 사전: {len(universe)}개 종목 ({cfg.path('universe')})")
    mentions = make_mentions(videos, universe, cfg)
    print(mentions.groupby(["target_type", mentions["guest"]]).size().to_string())
    _write(mentions, cfg.path("mentions"))


def cmd_prices(cfg: Config, args) -> None:
    """Yahoo Finance 에서 종목과 코스피 지수의 일별 가격을 받아 월 수익률 CSV 로 저장합니다."""
    from .months import month_str_to_ym
    from .returns.data import (build_returns_table, date_window, download_daily, load_universe,
                               save_csv, to_monthly)

    universe = load_universe(cfg.path("universe"))
    rcfg = cfg["returns"]
    start, end = date_window(month_str_to_ym(cfg["sample"]["start_month"]),
                             month_str_to_ym(cfg["sample"]["end_month"]))
    tickers = universe["code"].tolist()
    if args.max_tickers:
        tickers = tickers[:args.max_tickers]
    market = rcfg["market_ticker"]
    print(f"다운로드: 종목 {len(tickers)}개 + 시장({market}), 기간 {start} ~ {end}")
    daily, failed = download_daily(tickers + [market], start, end,
                                   sleep_seconds=float(rcfg["sleep_seconds"]),
                                   retries=int(rcfg["max_retries"]))
    if market not in daily:
        raise RuntimeError(f"시장 지수({market}) 를 받지 못했습니다. 잠시 후 다시 시도하세요.")
    market_df = to_monthly(daily.pop(market))
    returns = build_returns_table(daily)
    _write(returns, cfg.path("returns_monthly"))
    _write(market_df[["ym", "ret"]], cfg.path("market_monthly"))
    ok = returns["code"].nunique() if len(returns) else 0
    print(f"성공 {ok}개, 실패 {len(failed)}개" + (f": {', '.join(failed)}" if failed else ""))
    # 액면분할·분할상장 등을 Yahoo 가 제대로 반영하지 않으면 월 수익률이 비정상적으로 크게 나옵니다.
    odd = returns[returns["ret"].abs() > 0.4]
    if len(odd):
        print(f"주의: 월 수익률 절댓값이 40% 를 넘는 달 {len(odd)}개(액면분할·분할 미반영 가능성). 확인하세요:")
        print(odd.head(10)[["code", "ym", "ret"]].to_string(index=False))
    if not cfg.has_path("rf_monthly"):
        print(f"무위험수익률 파일이 없어서 연 {rcfg['rf_annual']:.2%} 상수를 씁니다(config 의 returns.rf_annual).")


def cmd_tone(cfg: Config, args) -> None:
    from .pipeline import score

    mentions = _read(cfg.path("mentions"))
    scored = score(mentions, cfg, method=args.method)
    _write(scored, cfg.path("scored"))


def _load_returns(cfg: Config):
    from .returns.data import load_panel

    return load_panel(cfg)


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

    s = sub.add_parser("prices", help="Yahoo Finance 에서 월 수익률 받기(config/universe.csv 의 종목, 네트워크 필요)")
    s.add_argument("--max-tickers", type=int, help="시험용으로 종목 수 제한")
    s.set_defaults(fn=cmd_prices)

    for name, fn, h in [("preprocess", cmd_preprocess, "자막 정리와 문장 분리"),
                        ("mentions", cmd_mentions, "종목·시장 언급 문장 추출(config/universe.csv 의 종목명 사용)"),
                        ("all", cmd_all, "preprocess → mentions → tone → analyze")]:
        s = sub.add_parser(name, help=h)
        s.add_argument("--videos", help="수집 결과 CSV(기본: paths.raw_videos)")
        s.add_argument("--method", choices=["dictionary", "llm"], help="톤 방법(all 에서만 사용)")
        s.set_defaults(fn=fn)

    s = sub.add_parser("tone", help="언급 문장 톤 점수화")
    s.add_argument("--method", choices=["dictionary", "llm"], help="기본: config tone.method")
    s.set_defaults(fn=cmd_tone)

    s = sub.add_parser("analyze", help="H1·H2 분석과 보고서 생성(prices 로 받은 수익률 필요)")
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
