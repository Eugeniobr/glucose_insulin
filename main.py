import argparse
import json

from config import load_config
from runner import run_forever, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pipeline glicose-insulina em tempo real")
    parser.add_argument("mode", choices=["once", "loop", "dashboard"], nargs="?", default="once")
    parser.add_argument("--skip-extract", action="store_true", help="Não chama extract.py antes da execução")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config()

    if args.mode == "loop":
        run_forever(config, skip_extract=args.skip_extract)
        return 0

    if args.mode == "dashboard":
        from dashboard_server import serve_dashboard

        serve_dashboard(host="127.0.0.1", port=8000, run_background_pipeline=True, skip_extract=args.skip_extract)
        return 0

    summary = run_pipeline(config, skip_extract=args.skip_extract)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
