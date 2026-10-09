"""极简本地 REST API（个人使用，无鉴权）。

启动: python cli.py serve --port 8765
接口:
  GET  /api/jobs            任务列表
  POST /api/jobs            {url, refs, mode, max} 新建任务，返回 job_id
  GET  /api/jobs/<id>       任务状态(job.json)
  GET  /api/jobs/<id>/report 生产报告
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .config import Config
from .pipeline import ProductionPipeline
from .story_count import clip_limit
from .utils import read_json


class _Server:
    def __init__(self, config):
        self.config = config
        self.jobs = {}
        self.lock = threading.Lock()

    def start_job(self, url, refs, mode="", max_clips=None):
        override = {}
        if mode:
            override["review_mode"] = mode
        if max_clips is not None:
            override["max_clips"] = clip_limit({'max_clips':max_clips})
        pipe = ProductionPipeline(self.config.clone(), url, references=list(refs or []),
                                  params_override=override or None)
        with self.lock:
            self.jobs[pipe.job_id] = pipe
        threading.Thread(target=pipe.run, daemon=True).start()
        return pipe.job_id

    def list_jobs(self):
        jobs_dir = self.config.workspace_path() / "jobs"
        out = []
        if jobs_dir.exists():
            for job in sorted(jobs_dir.iterdir(), reverse=True):
                data = read_json(job / "job.json")
                if data:
                    out.append({"job_id": data.get("job_id"), "status": data.get("status"),
                                "url": data.get("url"), "updated_at": data.get("updated_at")})
        return out

    def job_state(self, job_id):
        return read_json(self.config.job_dir(job_id) / "job.json") or {"error": "job not found"}

    def report(self, job_id):
        return read_json(self.config.job_dir(job_id) / "reports" / "production_report.json") or {"error": "no report"}


def make_handler(server):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            if path == "/api/jobs":
                return self._send(200, {"jobs": server.list_jobs()})
            parts = [p for p in path.split("/") if p]
            if len(parts) >= 3 and parts[0] == "api" and parts[1] == "jobs":
                job_id = parts[2]
                if len(parts) == 4 and parts[3] == "report":
                    return self._send(200, server.report(job_id))
                return self._send(200, server.job_state(job_id))
            if path in ("/", "/api"):
                return self._send(200, {"service": "short-video-factory",
                                        "endpoints": ["GET /api/jobs", "POST /api/jobs",
                                                      "GET /api/jobs/<id>", "GET /api/jobs/<id>/report"]})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/")
            if path != "/api/jobs":
                return self._send(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                data = json.loads(raw or b"{}")
            except ValueError:
                return self._send(400, {"error": "invalid json"})
            url = (data.get("url") or "").strip()
            if not url:
                return self._send(400, {"error": "url required"})
            try:
                job_id = server.start_job(url, data.get("refs") or [], data.get("mode", ""), data.get("max"))
            except ValueError as exc:
                return self._send(400,{'error':str(exc)})
            return self._send(202, {"job_id": job_id})

    return Handler


def serve(config_path="", host="127.0.0.1", port=8765):
    from pathlib import Path

    path = config_path or "workspace/config.json"
    config = Config.load(path) if Path(path).exists() else Config()
    server = _Server(config)
    httpd = ThreadingHTTPServer((host, port), make_handler(server))
    print("REST API: http://%s:%d (Ctrl+C 停止)" % (host, port))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
