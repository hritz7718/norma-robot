#!/usr/bin/env python3
"""Local, read-only NormaCore episode capture UI (no motor writes).

Install in: ~/norma-robot/local/episode_dashboard.py
Run: ~/norma-robot/software/ai/smolvla_py/.venv/bin/python ~/norma-robot/local/episode_dashboard.py
Open: http://127.0.0.1:8891

Station performs teleoperation separately. This dashboard only reads Station
history, exports bounded recordings via dataset-generator, and trains SmolVLA.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
SMOL = ROOT / "software/ai/smolvla_py"
PYTHON = SMOL / ".venv/bin/python"
DATASET_GENERATOR = ROOT / "runtime/dataset-generator"
OUTPUT_DIR = ROOT / "datasets/elrobot-local"
STATE_FILE = ROOT / "local/episode_dashboard_state.json"
LOG_DIR = ROOT / "local/episode_dashboard_logs"
FOLLOWER = "5B8E113018"
PORT = 8891

# This deliberately NEVER calls station_py.send_commands or run_policy.run_one_tick.
# The only Station API operation in preview mode is reading a fresh frame.
PREVIEW_CODE = r"""
import asyncio, logging, sys
from pathlib import Path
import torch
from scripts import run_policy as rp

checkpoint = Path(sys.argv[1])
task = sys.argv[2]

async def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading {checkpoint} on {device}...', flush=True)
    stats = {k: v.to(device) for k, v in rp.load_stats(checkpoint / 'stats.safetensors').items()}
    policy = rp.SmolVLAPolicy.from_pretrained(checkpoint,
        config_overrides={'load_vlm_weights': False}, strict=False).to(device)
    policy.eval()
    cli = await rp.new_station_client('localhost', logging.getLogger('safe-preview'))
    frame = await rp.fetch_frame(cli, 10.0)
    joints = frame.get_joints() or []
    images = frame.get_images() or []
    if len(joints) != 8 or len(images) < 2:
        raise RuntimeError(f'Expected 8 joints and 2 cameras, got {len(joints)} joints and {len(images)} cameras')
    batch, ranges = rp.frame_to_batch(frame, stats, rp.IMAGE_KEYS, device)
    tokens, mask = policy.tokenize_task(task, device=device)
    batch['observation.language.tokens'] = tokens
    batch['observation.language.attention_mask'] = mask
    with torch.inference_mode():
        pred = policy.predict_action_chunk(batch)[0]
        goals = rp.unnormalize_action(pred, stats)[0].detach().cpu().tolist()
    if len(goals) != 8: raise RuntimeError(f'Expected 8 predictions; got {len(goals)}')
    print(f'\nTask: {task}\nNO MOTOR COMMANDS ARE SENT')
    print(f'{"Joint":>5} {"Current":>10} {"Predicted":>10} {"Delta":>10}')
    for i, (joint, goal) in enumerate(zip(joints, goals), 1):
        current = float(joint.get_position_norm())
        print(f'{i:>5} {current:>10.3f} {goal:>10.3f} {goal-current:>+10.3f}')
    print('\nRead-only prediction complete. NO movement commands sent.')

asyncio.run(main())
"""

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "software/station/shared"))
from station_py import new_station_client  # noqa: E402
from target.gen_python.protobuf.drivers.inferences import normvla  # noqa: E402


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


class StationReader:
    """Single long-lived Station client in a dedicated asyncio thread."""

    def __init__(self):
        self.lock = threading.RLock()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.client = None
        self.latest: dict = {}
        self.cameras: list[bytes] = []
        self.error = "Connecting to Station…"
        self.observation_count = 0
        self.thread = threading.Thread(target=self._thread_main, daemon=True, name="station-reader")
        self.thread.start()

    def _thread_main(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        loop.run_until_complete(self._worker())

    async def _get_tail(self, queue: str):
        if self.client is None:
            raise RuntimeError("Station is not connected")
        qr = self.client.read_from_tail(queue, offset=b"\x00", limit=1, step=1, buf_size=1)
        entry = await asyncio.wait_for(qr.data.get(), timeout=6)
        if entry is None:
            raise RuntimeError(f"No data in {queue}: {qr.err}")
        return entry

    async def _worker(self):
        while True:
            try:
                self.client = await new_station_client("localhost", logging.getLogger("station-reader"))
                with self.lock:
                    self.error = ""
                while True:
                    entry = await self._get_tail("inference/normvla")
                    frame = normvla.FrameReader(memoryview(bytes(entry.Data)))
                    joints = frame.get_joints() or []
                    images = frame.get_images() or []
                    frameid = int.from_bytes(bytes(frame.get_global_frame_id()), "little")
                    observation = {
                        "time": time.time(),
                        "frame_pointer": frameid,
                        "joint_count": len(joints),
                        "camera_count": len(images),
                        "joints": [round(float(j.get_position_norm()), 4) for j in joints],
                        "ranges_valid": len(joints) == 8 and all(
                            int(j.get_range_min()) != int(j.get_range_max()) for j in joints
                        ),
                    }
                    jpeg_bytes = [bytes(im.get_jpeg()) for im in images[:2]]
                    with self.lock:
                        if frameid != self.latest.get("frame_pointer"):
                            self.observation_count += 1
                        self.latest = observation
                        self.cameras = jpeg_bytes
                        self.error = ""
                    await asyncio.sleep(0.45)
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                self.client = None
                await asyncio.sleep(2)

    async def _current_pointer(self):
        entry = await self._get_tail("inference-states")
        return int.from_bytes(bytes(entry.ID.ID), "little")

    def marker(self) -> int:
        loop = self.loop
        if loop is None or self.client is None:
            raise RuntimeError("Station not ready")
        future = asyncio.run_coroutine_threadsafe(self._current_pointer(), loop)
        return future.result(timeout=9)

    def snapshot(self):
        with self.lock:
            data = dict(self.latest)
            data["observation_count"] = self.observation_count
            data["error"] = self.error
        data["live"] = bool(data.get("time", 0) and time.time() - data["time"] < 4)
        return data

    def camera(self, i: int) -> bytes | None:
        with self.lock:
            return self.cameras[i] if i < len(self.cameras) else None


@dataclass
class Task:
    kind: str
    process: subprocess.Popen | None
    log_path: str
    started_at: str
    record_id: int | None = None
    checkpoint: str | None = None
    returncode: int | None = None
    file_handle: object | None = field(default=None, repr=False)

    def public(self):
        result = {
            "kind": self.kind, "started_at": self.started_at,
            "log_path": self.log_path, "returncode": self.returncode,
            "record_id": self.record_id, "checkpoint": self.checkpoint,
        }
        if self.process is not None:
            result["running"] = self.process.poll() is None
            if not result["running"]:
                result["returncode"] = self.process.returncode
        else:
            result["running"] = False
        return result


class Dashboard:
    def __init__(self):
        self.lock = threading.RLock()
        self.station = StationReader()
        self.recordings = []
        self.active = None
        self.jobs: dict[str, Task] = {}
        self.last_checkpoint = None
        self.alerts = []
        self._load()
        self._check_paths()

    def _check_paths(self):
        if not PYTHON.exists():
            raise SystemExit(f"Missing Python environment: {PYTHON}")
        if not DATASET_GENERATOR.exists():
            raise SystemExit(f"Missing dataset-generator binary: {DATASET_GENERATOR}")
        cfg = ROOT / "local/station.yaml"
        if not cfg.exists() or not re.search(r'^\s*st3215-bus:\s*["\']?' + FOLLOWER, cfg.read_text(), re.M):
            raise SystemExit(f"Set st3215-bus to {FOLLOWER} in {cfg} before starting UI")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _load(self):
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            self.recordings = data.get("recordings", [])
            self.active = data.get("active")
            self.last_checkpoint = data.get("last_checkpoint")

    def _persist(self):
        save_json(STATE_FILE, {
            "recordings": self.recordings,
            "active": self.active,
            "last_checkpoint": self.last_checkpoint,
        })

    def _good_count(self):
        return sum(1 for r in self.recordings if r.get("status") == "ready" and Path(r.get("file", "")).is_file())

    def _validate_station(self):
        s = self.station.snapshot()
        if not s["live"]:
            raise RuntimeError("No fresh observation frame. Check Station and both cameras.")
        if s.get("joint_count") != 8 or s.get("camera_count", 0) < 2 or not s.get("ranges_valid"):
            raise RuntimeError("Need 8 calibrated follower joints + 2 working cameras")
        return s

    def _busy_export(self):
        return any(job.kind == "export" and job.process and job.process.poll() is None for job in self.jobs.values())

    def start(self, task_name: str):
        with self.lock:
            if self.active:
                raise RuntimeError("An episode is already being recorded")
            if self._busy_export():
                raise RuntimeError("Let the current export finish before recording again")
            self._validate_station()
            task_name = task_name.strip()
            if not task_name or len(task_name) > 120:
                raise RuntimeError("Task name must be between 1 and 120 characters")
            pointer = self.station.marker()
            idx = 1 + max([r["id"] for r in self.recordings] + [0])
            self.active = {
                "id": idx, "task": task_name, "start_ptr": pointer,
                "started_at": now(), "start_time": time.time(),
                "start_obs_count": self.station.snapshot()["observation_count"],
            }
            self._persist()
            return f"Episode {idx} started. Demonstrate using Leader → Follower."

    def discard(self):
        with self.lock:
            if not self.active:
                raise RuntimeError("No episode to discard")
            idx = self.active["id"]
            self.active = None
            self._persist()
            return f"Episode {idx} discarded (no dataset exported)"

    def finish(self):
        with self.lock:
            if not self.active:
                raise RuntimeError("No active episode")
            self._validate_station()
            end = self.station.marker()
            active = dict(self.active)
            if end <= active["start_ptr"]:
                raise RuntimeError("Recording has no newer frames")
            length = time.time() - active["start_time"]
            if length < 7:
                raise RuntimeError("Episode too short. Make a complete demonstration and hold the final pose for 5–6 seconds.")
            active.update({
                "end_ptr": end, "finished_at": now(),
                "seconds": round(length, 1), "status": "exporting",
            })
            self.active = None
            self.recordings.append(active)
            self._persist()
            self._export(active)
            return f"Episode {active['id']} complete. Export started; wait for READY before recording again."

    def _launch(self, kind: str, args: list[str], cwd: Path, log: Path, record_id=None, checkpoint=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = log.open("w", buffering=1)
        fh.write("Command: " + " ".join(args) + "\n\n")
        fh.flush()
        try:
            process = subprocess.Popen(args, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        except Exception:
            fh.close()
            raise
        job = Task(kind, process, str(log), now(), record_id, checkpoint, file_handle=fh)
        self.jobs[kind] = job
        threading.Thread(target=self._wait_for_job, args=(job,), daemon=True).start()
        return job

    def _export(self, record):
        idx = record["id"]
        target = OUTPUT_DIR / f"episode_{idx:03d}"
        log = LOG_DIR / f"export_{idx:03d}.log"
        args = [
            str(DATASET_GENERATOR), "--robot", "localhost",
            "--queue", "inference/normvla",
            "--from", str(record["start_ptr"]),
            "--to", str(record["end_ptr"]),
            "--task", record["task"],
            "--episode.duration", "10",
            "--episode.min-commands", "5",
            "--output", str(target), "--no-tui",
        ]
        self._launch("export", args, ROOT, log, record_id=idx)

    def _wait_for_job(self, job: Task):
        code = job.process.wait()
        if job.file_handle:
            job.file_handle.close()
        with self.lock:
            job.returncode = code
            if job.kind == "export":
                record = next((r for r in self.recordings if r["id"] == job.record_id), None)
                if record:
                    path = OUTPUT_DIR / f"episode_{job.record_id:03d}.parquet"
                    record["status"] = "ready" if code == 0 and path.exists() and path.stat().st_size > 0 else "failed"
                    record["file"] = str(path) if path.exists() else ""
                    record["bytes"] = path.stat().st_size if path.exists() else 0
                    record["log"] = job.log_path
            if job.kind == "train" and code == 0:
                self.last_checkpoint = job.checkpoint
            self._persist()

    def train(self):
        with self.lock:
            if self.active or self._busy_export():
                raise RuntimeError("Finish and export the current recording first")
            files = [Path(r["file"]) for r in self.recordings if r.get("status") == "ready" and r.get("file") and Path(r["file"]).exists()]
            if len(files) < 5:
                raise RuntimeError(f"Need 5 exported episodes; currently have {len(files)}")
            job = self.jobs.get("train")
            if job and job.process and job.process.poll() is None:
                raise RuntimeError("Training is already running")
            checkpoint = SMOL / "checkpoints" / f"block-local-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            args = [
                str(PYTHON), "scripts/train.py",
                "--steps", "200", "--batch-size", "1", "--num-workers", "0",
                "--log-every", "10", "--save-every", "100",
                "--parquets", *map(str, files), "--output", str(checkpoint),
            ]
            self._launch("train", args, SMOL, LOG_DIR / "train.log", checkpoint=str(checkpoint / "final"))
            return f"Training on {len(files)} local episodes (200 steps)."

    def preview(self):
        with self.lock:
            if not self.last_checkpoint or not Path(self.last_checkpoint).exists():
                raise RuntimeError('Train successfully before previewing a policy')
            self._validate_station()
            train_job = self.jobs.get('train')
            preview_job = self.jobs.get('preview')
            if train_job and train_job.process and train_job.process.poll() is None:
                raise RuntimeError('Training is still running')
            if preview_job and preview_job.process and preview_job.process.poll() is None:
                raise RuntimeError('Prediction preview already running')
            self._launch('preview', [str(PYTHON), '-c', PREVIEW_CODE,
                self.last_checkpoint, 'pick up the block'], SMOL, LOG_DIR / 'preview.log')
            return 'Read-only VLA preview started. View the Prediction log. No motor commands will be sent.'

    def status(self):
        with self.lock:
            active = dict(self.active) if self.active else None
            if active:
                active["elapsed"] = round(time.time() - active["start_time"], 1)
                active["frames_seen"] = max(0, self.station.snapshot()["observation_count"] - active["start_obs_count"])
            jobs = {k: v.public() for k, v in self.jobs.items()}
            return {
                "station": self.station.snapshot(), "active": active,
                "records": self.recordings, "ready": self._good_count(),
                "target": 5, "jobs": jobs, "checkpoint": self.last_checkpoint,
            }

    def logs(self, kind: str) -> str:
        if kind not in ("export", "train", "preview"):
            return ""
        with self.lock:
            job = self.jobs.get(kind)
            if not job:
                return "No job started yet."
            path = Path(job.log_path)
        try:
            with path.open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 12000))
                return f.read().decode("utf-8", "replace")[-9000:]
        except OSError as e:
            return str(e)


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ElRobot · Episode Lab</title><style>
:root{color-scheme:dark;--bg:#0c111b;--card:#151f2e;--line:#29394c;--txt:#f1f6fc;--muted:#9cacbb;--good:#55e3b0;--blue:#66bdfd;--warn:#ffcd7c}
*{box-sizing:border-box}body{margin:0;padding:28px;font:15px/1.45 system-ui,Segoe UI,Arial;background:radial-gradient(circle at 20% -10%,#173148,var(--bg) 50%);color:var(--txt)}
main{max-width:1240px;margin:auto}h1{font-size:30px;margin:0 0 4px}p{color:var(--muted)}.sub{margin:0 0 23px}h2{font-size:18px;margin:0 0 14px}
.row{display:flex;gap:14px;flex-wrap:wrap}.panel{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px;flex:1;min-width:330px;box-shadow:0 8px 18px #0003}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px}.metric{background:#142032;border:1px solid #2a4156;padding:18px;border-radius:12px}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.12em}.value{font-weight:700;font-size:25px;margin-top:5px}
button{background:#245c91;color:white;border:0;border-radius:8px;padding:11px 17px;font-weight:650;font-size:14px;cursor:pointer}button:hover{filter:brightness(1.15)}button:disabled{opacity:.35;cursor:not-allowed}.green{background:#167d65}.red{background:#7d4049}.dark{background:#324355}input{background:#0c1623;color:white;border:1px solid #3a5163;border-radius:8px;padding:12px;min-width:250px;flex:1;font:inherit}
.pill{border-radius:20px;padding:3px 10px;background:#293c4e;color:var(--blue);font-size:12px}.ok{color:var(--good)}.bad{color:#ff9a9a}.warning{color:var(--warn)}.progress{height:9px;background:#25384d;border-radius:10px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#2d9bdb,#66e5b2);width:0%;transition:width .3s}
.cameras{display:flex;gap:12px}.camera{width:50%;min-height:150px;aspect-ratio:4/3;object-fit:contain;background:#0a0e12;border:1px solid var(--line);border-radius:10px}.infobox{background:#0a1827;border:1px solid var(--line);border-radius:8px;padding:12px;margin-top:12px;white-space:pre-wrap;min-height:48px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #28394b}th{color:#9ab0c5}pre{white-space:pre-wrap;word-break:break-word;max-height:320px;overflow:auto;background:#090e16;padding:14px;border-radius:8px;border:1px solid #263848;font-size:12px;line-height:1.4}
@media(max-width:750px){body{padding:12px}.metrics{grid-template-columns:repeat(2,1fr)}.row{flex-direction:column}.panel{min-width:0}.cameras{flex-direction:column}.camera{width:100%}}
</style></head><body><main>
<h1>◉ ElRobot · Episode Lab <span class="pill">LOCAL • NO MOTOR COMMANDS</span></h1>
<p class="sub">Capture teleoperated demonstrations → export Parquet → fine-tune SmolVLA. Record as many takes as needed; training requires at least 5 successful exports. Leader drives the Follower through Station.</p>
<div class="metrics"><div class="metric"><div class="label">Station</div><div class="value" id="connection">…</div></div><div class="metric"><div class="label">Ready episodes · minimum 5 to train</div><div class="value" id="count">0</div></div><div class="metric"><div class="label">Follower joints</div><div class="value" id="joints">…</div></div><div class="metric"><div class="label">Cameras</div><div class="value" id="camera_count">…</div></div></div>
<div class="row"><section class="panel"><h2>1 · Demonstration</h2><p>Keep teleoperation active in Station. Place the block and reset to the same starting pose before each take.</p>
<div class="row"><input type="text" id="task" value="pick up the block" aria-label="Task description"/></div>
<div class="row" style="margin-top:14px"><button class="green" id="start" onclick="action('start')">▶ Start episode</button><button id="finish" onclick="action('finish')">■ Finish & export</button><button class="red" id="discard" onclick="action('discard')">× Discard take</button></div>
<div class="infobox" id="episode_info">No active episode.</div><p class="warning" style="font-size:13px">After the final movement, hold the pose for 5–6 seconds before finishing. Don't touch AI movement controls.</p></section>
<section class="panel"><h2>Live observation</h2><div class="cameras"><img class="camera" id="cam0" alt="Camera 0"/><img class="camera" id="cam1" alt="Camera 1"/></div><div class="infobox" id="joint_text">Waiting for joint state…</div></section></div>
<section class="panel"><h2>2 · Recorded episodes</h2><div class="progress"><div class="fill" id="fill"></div></div><table><thead><tr><th>Take</th><th>Task</th><th>Duration</th><th>Export</th><th>Size</th></tr></thead><tbody id="records"><tr><td colspan="5">No recordings yet.</td></tr></tbody></table><div class="infobox" id="notice">Ready to connect.</div></section>
<div class="row"><section class="panel"><h2>3 · Train SmolVLA</h2><p>After five successful exports, train on your own robot's camera views and calibration. Uses RTX 2050-compatible batch size 1, 200 steps.</p><button class="green" id="train" onclick="action('train')">⚡ Train block-local model</button><p id="train_status">Waiting for 5 valid episodes.</p><p id="checkpoint"></p><button class="dark" id="preview" onclick="action('preview')">◎ Preview predictions (NO MOVEMENT)</button><p>Preview uses the trained checkpoint and current camera feeds. It never sends a target to a motor.</p></section><section class="panel"><h2>Export / training / preview logs</h2><div class="row"><button class="dark" onclick="logKind='export';refreshLogs()">Export log</button><button class="dark" onclick="logKind='train';refreshLogs()">Training log</button><button class="dark" onclick="logKind='preview';refreshLogs()">Prediction log</button></div><pre id="logs">No logs yet.</pre></section></div>
<p style="text-align:center">Station: <code>127.0.0.1:8888</code> · Dashboard: <code>127.0.0.1:8891</code> · Follower: <code>5B8E113018</code></p>
</main><script>
let logKind='export',busy=false,latest={};
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path,method='GET',body={}){let r=await fetch(path,{method,headers:{'Content-Type':'application/json'},...(method==='POST'?{body:JSON.stringify(body)}:{})});let o=await r.json();if(!r.ok||!o.ok&&method==='POST')throw Error(o.error||r.statusText);return o}
async function action(which){if(busy)return;busy=true;try{let res=await api('/api/'+which,'POST',{task:$('task').value});$('notice').textContent=res.message||'Done.';await refresh()}catch(e){$('notice').textContent='ERROR: '+e.message}finally{busy=false}}
function status(r){let s=r.station||{}, live=s.live, active=r.active, n=r.ready||0;$('connection').textContent=live?'Online':'Offline';$('connection').className='value '+(live?'ok':'bad');$('count').textContent=n;$('joints').textContent=s.joint_count??'—';$('camera_count').textContent=s.camera_count??'—';$('fill').style.width=(Math.min(5,n)*20)+'%';
$('start').disabled=!!active||!live||!!r.jobs?.export?.running;$('finish').disabled=!active;$('discard').disabled=!active;$('train').disabled=!!active||n<5||!!r.jobs?.train?.running||!!r.jobs?.export?.running;$('preview').disabled=!r.checkpoint||!!r.jobs?.preview?.running||!!r.jobs?.train?.running;
$('episode_info').textContent=active?'● RECORDING #'+active.id+'  |  '+active.elapsed+' s  |  '+active.frames_seen+' observed frames\nTask: '+active.task:'No active episode. '+(live?'Ready to start a demonstration.':'Waiting for Station…');
$('joint_text').textContent=live?'Joint positions: '+(s.joints||[]).map((x,i)=>'J'+(i+1)+': '+(100*x).toFixed(1)+'%').join(' | ')+'\nFrame pointer: '+s.frame_pointer:'Connection: '+(s.error||'waiting');
$('records').innerHTML=r.records?.length?r.records.map(x=>'<tr><td>#'+x.id+'</td><td>'+esc(x.task)+'</td><td>'+(x.seconds||'—')+' s</td><td class="'+(x.status==='ready'?'ok':x.status==='failed'?'bad':'warning')+'">'+esc(x.status)+'</td><td>'+(x.bytes?Math.round(x.bytes/1024)+' KB':'—')+'</td></tr>').join(''):'<tr><td colspan="5">No recordings yet.</td></tr>';
$('train_status').textContent=r.jobs?.train?(r.jobs.train.running?'Training running…':'Training finished (exit '+r.jobs.train.returncode+')'):(n>=5?'Ready to train on '+n+' episodes. You can keep recording.':'Collect '+(5-n)+' more exported episode(s).');
$('checkpoint').textContent=r.checkpoint?'Model: '+r.checkpoint:'';
}
async function refresh(){try{latest=await api('/api/status');status(latest);let t=Date.now();if(latest.station?.live){$('cam0').src='/camera/0?t='+t;$('cam1').src='/camera/1?t='+t}}catch(e){$('connection').textContent='UI error';$('notice').textContent=e.message}}
async function refreshLogs(){try{let o=await api('/api/logs?kind='+logKind);$('logs').textContent=o.log||'No logs.';$('logs').scrollTop=$('logs').scrollHeight}catch(e){}}
refresh();refreshLogs();setInterval(refresh,1800);setInterval(refreshLogs,3500);
</script></body></html>'''


app: Dashboard | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if not self.path.startswith("/api/status") and not self.path.startswith("/camera/"):
            super().log_message(fmt, *args)

    def _out(self, status: int, data: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def json(self, data: dict, status=200):
        self._out(status, json.dumps(data).encode(), "application/json; charset=utf-8")

    def do_GET(self):
        assert app
        path = urlparse(self.path).path
        if path == "/":
            self._out(200, PAGE.encode(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self.json({"ok": True, **app.status()})
        elif path == "/api/logs":
            kind = "train" if "kind=train" in self.path else "export"
            self.json({"ok": True, "log": app.logs(kind)})
        elif path in ("/camera/0", "/camera/1"):
            cam = app.station.camera(int(path[-1]))
            if cam:
                self._out(200, cam, "image/jpeg")
            else:
                self._out(404, b"camera unavailable", "text/plain")
        else:
            self.json({"error": "Not found"}, 404)

    def do_POST(self):
        assert app
        path = urlparse(self.path).path
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self.json({"ok": False, "error": "Content-Type must be application/json"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0 or length > 2048:
                raise ValueError("Invalid payload size")
            data = json.loads(self.rfile.read(length) or b"{}")
            if path == "/api/start":
                msg = app.start(str(data.get("task", "pick up the block")))
            elif path == "/api/finish":
                msg = app.finish()
            elif path == "/api/discard":
                msg = app.discard()
            elif path == "/api/train":
                msg = app.train()
            elif path == "/api/preview":
                msg = app.preview()
            else:
                self.json({"ok": False, "error": "Not found"}, 404)
                return
            self.json({"ok": True, "message": msg})
        except Exception as e:
            self.json({"ok": False, "error": str(e)}, 400)


def main():
    global app
    app = Dashboard()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print("\nElRobot Episode Lab | READ-ONLY Station access")
    print(f"Visit http://127.0.0.1:{PORT}")
    print(f"Follower {FOLLOWER} | training output {SMOL / 'checkpoints'}")
    print("Press Ctrl+C to close dashboard (Station runs separately).\n", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
