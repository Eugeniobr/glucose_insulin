import argparse
import json
import threading
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from config import load_config
from runner import run_forever, run_pipeline


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, config=None, **kwargs):
        self.app_config = config
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self):
        return self._handle_request(send_body=True)

    def do_HEAD(self):
        return self._handle_request(send_body=False)

    def _handle_request(self, send_body: bool):
        parsed = urlparse(self.path)
        if parsed.path == "/api/metrics":
            return self._serve_json(self.app_config.metrics_path, send_body=send_body)
        if parsed.path == "/api/state":
            return self._serve_json(self.app_config.state_path, send_body=send_body)
        if parsed.path == "/api/series":
            return self._serve_json(self.app_config.series_path, send_body=send_body)
        if parsed.path.startswith("/outputs/"):
            return self._serve_file((self.app_config.base_dir / parsed.path.lstrip("/")).resolve(), send_body=send_body)
        if parsed.path == "/health":
            return self._serve_bytes(b'{"status":"ok"}', "application/json; charset=utf-8", send_body=send_body)
        if send_body:
            return super().do_GET()
        return super().do_HEAD()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, format, *args):
        return

    def _serve_json(self, path: Path, send_body: bool = True):
        if not path.exists():
            payload = json.dumps({"error": f"arquivo não encontrado: {path.name}"}).encode("utf-8")
            return self._serve_bytes(payload, "application/json; charset=utf-8", HTTPStatus.NOT_FOUND, send_body=send_body)
        return self._serve_bytes(path.read_bytes(), "application/json; charset=utf-8", send_body=send_body)

    def _serve_file(self, path: Path, send_body: bool = True):
        try:
            path.relative_to(self.app_config.base_dir)
        except ValueError:
            return self.send_error(HTTPStatus.FORBIDDEN, "Caminho inválido")
        if not path.exists() or not path.is_file():
            return self.send_error(HTTPStatus.NOT_FOUND, "Arquivo não encontrado")
        content_type = self.guess_type(str(path))
        return self._serve_bytes(path.read_bytes(), content_type, send_body=send_body)

    def _serve_bytes(self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK, send_body: bool = True):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        if send_body:
            self.wfile.write(content)


def start_pipeline_thread(skip_extract: bool) -> threading.Thread:
    config = load_config()
    worker = threading.Thread(target=run_forever, args=(config, skip_extract), daemon=True)
    worker.start()
    return worker


def serve_dashboard(host: str, port: int, run_background_pipeline: bool, skip_extract: bool) -> None:
    config = load_config()
    if not config.metrics_path.exists():
        run_pipeline(config, skip_extract=skip_extract)

    if run_background_pipeline:
        start_pipeline_thread(skip_extract=skip_extract)

    dashboard_dir = config.base_dir / "dashboard"
    handler = partial(DashboardHandler, directory=str(dashboard_dir), config=config)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Dashboard disponível em http://{host}:{port}")
    server.serve_forever()


def build_parser():
    parser = argparse.ArgumentParser(description="Dashboard local do pipeline glicose-insulina")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--skip-extract", action="store_true", help="Não chama extract.py no refresh do pipeline")
    parser.add_argument(
        "--no-background-pipeline",
        action="store_true",
        help="Serve o dashboard sem rodar o pipeline em background",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    serve_dashboard(
        host=args.host,
        port=args.port,
        run_background_pipeline=not args.no_background_pipeline,
        skip_extract=args.skip_extract,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
