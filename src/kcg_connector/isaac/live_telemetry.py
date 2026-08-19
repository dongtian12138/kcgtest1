"""Live telemetry for the D38999 tabletop pipeline.

A tiny stdlib HTTP server that streams:
  * the wrist 6-DOF wrench and the current runtime phase (JSON polling),
  * palm/wrist camera video as MJPEG (browser <img> shows live video).
The Isaac thread publishes samples/frames into thread-safe buffers; the
server thread only reads.  No third-party packages; the server cannot
perturb the formal physics episode.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HTML_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<title>D38999 实时监控：掌心/腕部相机 + 腕力</title>
<style>
 body{background:#0d1117;color:#e6edf3;font-family:monospace;margin:0;padding:10px}
 #phase{font-size:20px;font-weight:bold;color:#58a6ff;margin-bottom:4px}
 #sub{font-size:12px;color:#8b949e;margin-bottom:8px}
 .vrow{display:grid;grid-template-columns:1fr 1fr;gap:10px}
 .panel{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:8px}
 .panel h2{font-size:13px;margin:2px 0 6px 0;color:#79c0ff}
 .panel img{width:100%;height:auto;display:block;background:#000;border-radius:4px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px}
 canvas{width:100%;height:160px;display:block}
 table{font-size:12px;width:100%;border-collapse:collapse}
 td,th{border-bottom:1px solid #30363d;padding:2px 6px;text-align:right}
 th{color:#8b949e}
 .val{font-weight:bold}
</style></head><body>
<div id="phase">等待数据…</div>
<div id="sub">global_step 0 · 0 samples</div>
<div class="vrow">
 <div class="panel"><h2>Palm 掌心相机（实时）</h2>
   <img src="/video?cam=palm" alt="palm"></div>
 <div class="panel"><h2>Wrist 腕部相机（实时）</h2>
   <img src="/video?cam=wrist" alt="wrist"></div>
</div>
<div class="grid">
 <div class="panel"><h2>腕力 F (N) —— 粉=Fx 青=Fy 黄=Fz</h2><canvas id="f" width="900" height="160"></canvas></div>
 <div class="panel"><h2>腕力矩 M (N·m) —— 粉=Tx 青=Ty 黄=Tz</h2><canvas id="m" width="900" height="160"></canvas></div>
</div>
<div class="grid">
 <div class="panel"><h2>力模 |F| 与 力矩模 |M|</h2><canvas id="n" width="900" height="160"></canvas></div>
 <div class="panel"><h2>当前值</h2><table id="t"></table></div>
</div>
<script>
const N=600, f=[[],[],[]], m=[[],[],[]], n=[[],[]];
let since=0, phase="", latest=null;
function draw(cv, series, colors){
  const ctx=cv.getContext('2d'); const W=cv.width, H=cv.height;
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#30363d'; ctx.lineWidth=1;
  for(let i=0;i<6;i++){ctx.beginPath();ctx.moveTo(0,i*H/6);ctx.lineTo(W,i*H/6);ctx.stroke();}
  const all=series.flat();
  if(all.length<2) return;
  const mn=Math.min(...all), mx=Math.max(...all);
  const lo=mn-(mx-mn||1)*0.1, hi=mx+(mx-mn||1)*0.1;
  series.forEach((arr,i)=>{
    ctx.strokeStyle=colors[i%3]; ctx.lineWidth=1.6; ctx.beginPath();
    arr.forEach((v,j)=>{const x=j/(N-1)*W; const y=H-((v-lo)/(hi-lo))*H;
      j?ctx.lineTo(x,y):ctx.moveTo(x,y);});
    ctx.stroke();
  });
  ctx.fillStyle='#8b949e'; ctx.font='10px monospace';
  ctx.fillText('max '+mx.toFixed(3)+'  min '+mn.toFixed(3), 6, 12);
}
async function poll(){
  try{
    const r=await fetch('/data?since='+since);
    const d=await r.json();
    since=d.since; phase=d.phase||phase;
    for(const s of d.samples){
      const w=s.w;
      f[0].push(w[0]);f[1].push(w[1]);f[2].push(w[2]);
      m[0].push(w[3]);m[1].push(w[4]);m[2].push(w[5]);
      n[0].push(Math.hypot(w[0],w[1],w[2]));
      n[1].push(Math.hypot(w[3],w[4],w[5]));
      for(const arr of [f[0],f[1],f[2],m[0],m[1],m[2],n[0],n[1]])
        if(arr.length>N) arr.shift();
      latest=s;
    }
    if(latest){
      document.getElementById('phase').textContent='阶段: '+phase;
      document.getElementById('sub').textContent='global_step '+latest.step+' · '+since+' samples';
      const w=latest.w;
      document.getElementById('t').innerHTML=
        '<tr><th></th><th>Fx</th><th>Fy</th><th>Fz</th><th>Tx</th><th>Ty</th><th>Tz</th></tr>'+
        '<tr><td>补偿后</td>'+w.map(v=>'<td class="val">'+v.toFixed(3)+'</td>').join('')+'</tr>'+
        '<tr><td>|F|/|M|</td><td colspan="3">'+Math.hypot(w[0],w[1],w[2]).toFixed(3)+' N</td><td colspan="3">'+Math.hypot(w[3],w[4],w[5]).toFixed(3)+' N·m</td></tr>';
    }
    draw(document.getElementById('f'),f,['#ff7b72','#79c0ff','#d2a8ff']);
    draw(document.getElementById('m'),m,['#ff7b72','#79c0ff','#d2a8ff']);
    draw(document.getElementById('n'),n,['#ffa657','#56d364']);
  }catch(e){}
  setTimeout(poll,200);
}
poll();
</script></body></html>"""

_BOUNDARY = b"--d38999frame"


class _LiveTelemetry:
    """Thread-safe sample/frame buffers + HTTP server for one pipeline run."""

    def __init__(self, port: int):
        self._lock = threading.Lock()
        self._samples: deque[dict] = deque()
        self._phase = "boot"
        self._frames: dict[str, bytes | None] = {"palm": None, "wrist": None}
        self._frame_seq: dict[str, int] = {"palm": 0, "wrist": 0}
        self._frame_cond = threading.Condition(self._lock)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._port = int(port)
        self._started = False

    @property
    def port(self) -> int:
        return self._port

    def publish(self, *, global_step: int, phase: str,
                wrench: tuple[float, ...]) -> None:
        sample = {
            "step": int(global_step),
            "phase": str(phase),
            "w": [float(value) for value in wrench[:6]],
        }
        with self._lock:
            self._samples.append(sample)
            self._phase = str(phase)

    def publish_frame(self, cam: str, jpeg: bytes) -> None:
        with self._frame_cond:
            self._frames[cam] = bytes(jpeg)
            self._frame_seq[cam] += 1
            self._frame_cond.notify_all()

    def snapshot(self, since: int) -> dict:
        with self._lock:
            items = list(self._samples)[max(0, since):]
            since = len(self._samples)
            phase = self._phase
        return {"since": since, "phase": phase, "samples": items}

    def _stream_mjpeg(self, wfile, cam: str) -> None:
        """Blocking MJPEG loop for one client; ends when the client goes."""
        last_seq = -1
        while True:
            with self._frame_cond:
                while self._frame_seq[cam] == last_seq:
                    self._frame_cond.wait(timeout=0.1)
                frame = self._frames[cam]
                last_seq = self._frame_seq[cam]
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                wfile.write(_BOUNDARY + b"\r\n")
                wfile.write(b"Content-Type: image/jpeg\r\n")
                wfile.write(
                    b"Content-Length: " + str(len(frame)).encode() + b"\r\n"
                )
                wfile.write(b"\r\n")
                wfile.write(frame)
                wfile.write(b"\r\n")
                wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    def start(self) -> bool:
        if self._started:
            return True
        telemetry = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self):  # noqa: N802
                if self.path.startswith("/video"):
                    cam = "palm"
                    if "cam=wrist" in self.path:
                        cam = "wrist"
                    elif "cam=palm" in self.path:
                        cam = "palm"
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=d38999frame",
                    )
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    telemetry._stream_mjpeg(self.wfile, cam)
                    return
                if self.path.startswith("/data"):
                    since = 0
                    if "since=" in self.path:
                        try:
                            since = int(self.path.split("since=")[1])
                        except ValueError:
                            since = 0
                    payload = json.dumps(
                        telemetry.snapshot(since), ensure_ascii=False
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                page = _HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)

            def log_message(self, *_args):  # silence request logs
                return

        try:
            self._httpd = ThreadingHTTPServer(
                ("127.0.0.1", self._port), _Handler
            )
        except OSError as error:
            print(f"[live_telemetry] cannot bind port {self._port}: "
                  f"{error}", flush=True)
            return False
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._thread.start()
        self._started = True
        print(f"[live_telemetry] 实时监控页: "
              f"http://127.0.0.1:{self._port}/", flush=True)
        return True


_singleton: _LiveTelemetry | None = None


def start_live_telemetry(port: int = 8790) -> bool:
    global _singleton
    _singleton = _LiveTelemetry(port)
    return _singleton.start()


def publish_live_wrench(global_step: int, phase: str,
                        wrench: tuple[float, ...]) -> None:
    if _singleton is not None and _singleton._started:
        _singleton.publish(global_step=global_step, phase=phase,
                           wrench=wrench)


def publish_live_frame(cam: str, jpeg: bytes) -> None:
    if _singleton is not None and _singleton._started:
        _singleton.publish_frame(cam, jpeg)


def live_telemetry_port() -> int | None:
    if _singleton is not None and _singleton._started:
        return _singleton.port
    return None
