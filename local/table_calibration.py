"""Standalone four-point tabletop calibration. Reads Station; never moves motors.

Launch: software/ai/smolvla_py/.venv/bin/python local/table_calibration.py --port 8894
"""
import argparse
import asyncio
import io
import itertools
import json
import math
import logging
import secrets
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

# Support the explicit script launcher from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local.mission_control.config import ROOT, STATION, CAMERAS, MANUAL_CALIBRATION, FOLLOWER, MOTOR_IDS
from local.mission_control.cameras import CameraReader
from local.mission_control.station import decode_motors
from local.mission_control.positioning import TaughtMapping, homography_signature



HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>4-point camera calibration</title><script src="/calibration/app.js" defer></script>
<style>
:root{color-scheme:dark;font:15px system-ui;background:#10151c;color:#e5eaf0}
body{max-width:960px;margin:24px auto;padding:0 18px}h1{font-size:1.5rem}p{line-height:1.5}
button,input,select{font:inherit;border:1px solid #425065;border-radius:5px;padding:9px;background:#182331;color:inherit}
button{cursor:pointer}button:disabled{opacity:.45;cursor:default}input{width:110px}button:focus-visible,input:focus-visible,canvas:focus-visible{outline:2px solid #9bc5ff}
.buttons{display:flex;flex-wrap:wrap;gap:10px;margin:16px 0}.muted{color:#9baabb}.good{color:#8ed5b7}.error{color:#ff9e9e}
canvas{display:block;width:100%;height:auto;max-height:none;background:#080d14;cursor:crosshair;outline:1px solid #425065}
[hidden]{display:none!important}table{width:100%;border-collapse:collapse;margin:16px 0}th,td{text-align:left;padding:8px;border-bottom:1px solid #2c3746}
.table{overflow-x:auto}#coordinates{font-variant-numeric:tabular-nums;font-size:1.05rem}
</style></head><body>
<h1>4-point camera calibration</h1>
<p class="muted">Capture the workspace camera. Click four points and enter each point’s real-world X/Y in millimetres. Keep the camera fixed after saving.</p>
<label for="camera">Tabletop camera</label> <select id="camera"><option value="">Waiting for cameras…</option></select>
<div class="buttons"><button id="capture">Capture image</button><button id="recalibrate">Recalibrate</button><button id="save" disabled>Save calibration</button></div>
<p id="status" role="status">Connecting to Station…</p>
<canvas id="image" hidden aria-label="Captured tabletop image; click to select calibration points or inspect coordinates"></canvas>
<p id="coordinates" role="status"></p>
<div class="table"><table><thead><tr><th>Point</th><th>Pixel (u, v)</th><th>X · mm</th><th>Y · mm</th></tr></thead><tbody id="points"></tbody></table></div>
<p class="muted" id="saved-path"></p>
<div class="table"><table><thead><tr><th>Robot base</th><th>Pixel (u, v)</th><th>X · mm</th><th>Y · mm</th></tr></thead>
<tbody><tr><td><button id="pick-base" disabled>Select base point</button></td><td id="base-pixel">Not selected</td>
<td><input id="base-x" type="number" step="any" aria-label="Base origin table X in mm"></td>
<td><input id="base-y" type="number" step="any" aria-label="Base origin table Y in mm"></td></tr></tbody></table></div>
<p class="muted">Click the tabletop projection directly below the URDF base origin, or enter its measured X/Y. An elevated joint cannot be projected with the tabletop homography.</p>
<p><label>Base +X angle · degrees <input id="base-yaw" type="number" step="any"></label>
<label>Base origin height above table · mm <input id="base-height" type="number" step="any"></label></p>
<p class="muted">Angle 0 means robot +X points along table +X; positive angles turn toward table +Y. Height is positive above the tabletop, negative below it.</p>
<p><label><input id="base-fixed" type="checkbox" checked style="width:auto"> Base and tabletop are level; robot, camera and tabletop references will remain fixed.</label></p>
<button id="save-base" disabled>Save base alignment</button>
<p id="base-status" role="status">Save camera calibration first, then align the robot base.</p>
<h2>Teach Robot · 9 points</h2>
<p class="muted">Teaching points may extend beyond the four homography references. Enter their measured coordinates in the same table coordinate system.</p>
<p>Use your existing Leader → Follower teleoperation to hold the gripper above each reference point. Enter its table X/Y and measured gripper height Z, then click Save Point. Keep the same height, gripper orientation and opening for all nine points.</p>
<p class="muted">This page records Follower feedback only. It does not move the robot or change teleoperation or torque. Click the calibrated tabletop image to fill X/Y; click the table point directly below the gripper, rather than its elevated tip.</p>
<div class="buttons">
<label>Point <select id="teach-slot" aria-label="Teaching point number"></select></label>
<label>X · mm <input id="teach-x" type="number" step="any"></label>
<label>Y · mm <input id="teach-y" type="number" step="any"></label>
<label>Z above table · mm <input id="teach-z" type="number" step="any" min="1" max="500" placeholder="Measured height"></label>
</div>
<button id="teach-save" disabled>Save Point</button>
<button id="teach-reset" disabled>Start new teaching set</button>
<p id="teach-status" role="status">Save the camera calibration, then teach nine points across the workspace.</p>
<div class="table"><table><thead><tr><th>Point</th><th>Table X / Y / Z · mm</th><th>Follower joints 1–8 · ticks</th></tr></thead><tbody id="teach-points"></tbody></table></div>
<p class="muted" id="teach-path"></p>
<p>After all nine points are saved, enter an X/Y between points and preview the estimated joint targets. Preview does not move the robot.</p>
<button id="teach-preview" disabled>Preview interpolation</button>
<p id="teach-preview-result" class="muted" role="status"></p>
<p><label><input id="teach-validated" type="checkbox" style="width:auto"> I physically checked interpolated positions and clearance at this height, kept the gripper orientation/opening consistent, and kept the robot, camera and table fixed.</label></p>
<button id="teach-enable" disabled>Enable taught controller</button>
<p class="muted">Execute uses the validated taught height. Start from any taught pose and disable mirroring before Execute. Interpolation estimates height; it does not measure live Cartesian clearance. Changing a teaching point disables the controller until revalidated.</p>
</body></html>"""

JS = r"""(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const prefix = '/calibration';
  const canvas = $('image'), ctx = canvas.getContext('2d');
  let token, capture, image, saved, selecting = true, points = [], lookup = null, busy = false, cameraKey = '';
  let pickingBase = false, basePixel = null;
  let taught = null, teachingMatches = false, teachingError = '';
  let previewSignature = null;
  $('base-x').value='-170'; $('base-y').value='185'; $('base-yaw').value='0'; $('base-height').value='35';
  function status(message, error = false) {
    $('status').textContent = message;
    $('status').className = error ? 'error' : message.startsWith('Calibration Successful') ? 'good' : '';
  }
  async function api(path, payload) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch(prefix + path, payload === undefined ? {cache:'no-store',signal:controller.signal} : {
        method:'POST',headers:{'Content-Type':'application/json','X-Calibration-Token':token},
        body:JSON.stringify(payload),signal:controller.signal
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'Request failed.');
      return result;
    } finally { clearTimeout(timer); }
  }
  function world() {
    return [0,1,2,3].map(i => [$(`x${i}`).valueAsNumber,$(`y${i}`).valueAsNumber]);
  }
  function controls() {
    $('capture').disabled = busy || !token || !$('camera').value;
    $('recalibrate').disabled = busy || !token || !$('camera').value;
    $('camera').disabled = busy;
    $('save').disabled = busy || !selecting || points.length !== 4 || !world().flat().every(Number.isFinite);
    for (let i=0;i<4;i++) for (const axis of ['x','y']) $(`${axis}${i}`).disabled=busy || !selecting;
    $('pick-base').disabled=busy || selecting || !image || !saved;
    $('save-base').disabled=busy || selecting || !capture || !saved || !$('base-fixed').checked ||
      !['base-x','base-y','base-yaw','base-height'].every(id=>Number.isFinite($(id).valueAsNumber));
    $('teach-save').disabled=busy || !token || !saved || !teachingMatches ||
      !['teach-x','teach-y','teach-z'].every(id=>Number.isFinite($(id).valueAsNumber)) || !$('teach-z').checkValidity();
    $('teach-reset').disabled=busy || !token || !saved;
    const nineReady = teachingMatches && taught?.points.length === 9;
    $('teach-preview').disabled=busy || !token || !nineReady || !['teach-x','teach-y'].every(id=>Number.isFinite($(id).valueAsNumber));
    $('teach-enable').disabled=busy || !token || !nineReady || !previewSignature || !$('teach-validated').checked;
    $('teach-validated').disabled=busy;
    for (const id of ['teach-slot','teach-x','teach-y','teach-z']) $(id).disabled=busy;
  }
  function rows() {
    for (let i = 0; i < 4; i++) {
      $(`pixel${i}`).textContent = points[i] ? points[i].map(v => v.toFixed(1)).join(', ') : 'Click point ' + (i+1);
    }
    controls();
  }
  function draw() {
    if (!image) return;
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.drawImage(image,0,0);
    ctx.font = `${Math.max(10,canvas.width/55)}px sans-serif`;
    points.forEach(([u,v],i) => {
      ctx.strokeStyle='#62ffd0';ctx.fillStyle='#62ffd0';ctx.lineWidth=1.5;
      ctx.beginPath();ctx.arc(u,v,Math.max(3,canvas.width/140),0,Math.PI*2);ctx.stroke();
      ctx.fillText(String(i+1),u+5,v-5);
    });
    if (lookup) {
      ctx.strokeStyle='#ffc76b';ctx.lineWidth=1.5;ctx.beginPath();
      ctx.moveTo(lookup[0]-5,lookup[1]);ctx.lineTo(lookup[0]+5,lookup[1]);
      ctx.moveTo(lookup[0],lookup[1]-5);ctx.lineTo(lookup[0],lookup[1]+5);ctx.stroke();
    }
    if (basePixel) {
      ctx.fillStyle='#ff81d4';ctx.strokeStyle='#ff81d4';ctx.beginPath();
      ctx.arc(basePixel[0],basePixel[1],5,0,Math.PI*2);ctx.stroke();
      ctx.fillText('BASE',basePixel[0]+6,basePixel[1]-6);
    }
  }
  async function takePicture(recalibrate = false) {
    busy = true; controls(); status('Capturing tabletop camera…');
    try {
      const next = await api('/api/capture',{camera_queue:$('camera').value});
      const nextImage = new Image();
      await new Promise((resolve,reject) => {
        nextImage.onload=resolve; nextImage.onerror=() => reject(new Error('Captured image could not be loaded.'));
        nextImage.src=`${prefix}/api/capture.jpg?id=${encodeURIComponent(next.capture_id)}`;
      });
      capture=next;image=nextImage;pickingBase=false;basePixel=null;
      canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;canvas.hidden=false;
      selecting=recalibrate || !saved || saved.camera_source!==next.camera_source || saved.image_size[0]!==canvas.width || saved.image_size[1]!==canvas.height;
      points=selecting ? [] : saved.pixel_points.map(p=>p.slice());lookup=null;$('coordinates').textContent='';
      status(selecting ? 'Click point 1 of 4, then enter its X/Y coordinates.' : 'Calibration Successful — click the image to read X/Y coordinates.');
      rows();draw();
    } catch(error) { status(error.message || 'Capture failed. Wait for the camera, then retry.',true); }
    finally { busy=false;controls(); }
  }
  canvas.addEventListener('click',event => {
    if (!image || busy) return;
    const rect=canvas.getBoundingClientRect();
    const u=Math.min(canvas.width-1,Math.max(0,(event.clientX-rect.left)*canvas.width/rect.width));
    const v=Math.min(canvas.height-1,Math.max(0,(event.clientY-rect.top)*canvas.height/rect.height));
    if (selecting) {
      if (points.length>=4) { status('Enter the four X/Y pairs, then click Save calibration.');return; }
      points.push([u,v]);rows();draw();$(`x${points.length-1}`).focus();
      status(points.length===4 ? 'Four points selected. Enter their X/Y coordinates and save.' : `Click point ${points.length+1} of 4.`);
    } else if (saved) {
      const h=saved.image_to_table;
      const denominator=h[2][0]*u+h[2][1]*v+h[2][2];
      if (Math.abs(denominator)<1e-10) { status('This pixel is outside the valid projection.',true);return; }
      const x=(h[0][0]*u+h[0][1]*v+h[0][2])/denominator;
      const y=(h[1][0]*u+h[1][1]*v+h[1][2])/denominator;
      if (![x,y].every(Number.isFinite)) { status('This pixel has no finite tabletop coordinate.',true);return; }
      lookup=[u,v];draw();$('coordinates').textContent=`X: ${x.toFixed(2)} mm     Y: ${y.toFixed(2)} mm`;
      if (pickingBase) {
        basePixel=[u,v];pickingBase=false;
        $('base-x').value=x.toFixed(3);$('base-y').value=y.toFixed(3);
        $('base-pixel').textContent=`${u.toFixed(1)}, ${v.toFixed(1)}`;
        $('base-status').textContent='Base point selected. Enter measured orientation and height, then save.';
        controls();draw();
      } else {
        $('teach-x').value=x.toFixed(3);$('teach-y').value=y.toFixed(3);controls();
      }
    }
  });
  $('capture').addEventListener('click',()=>takePicture(false));
  $('camera').addEventListener('change',()=>{
    image=null;capture=null;points=[];lookup=null;selecting=true;pickingBase=false;basePixel=null;canvas.hidden=true;
    $('coordinates').textContent='';rows();status('Click Capture image for the selected camera.');
  });
  $('recalibrate').addEventListener('click',()=>takePicture(true));
  $('pick-base').addEventListener('click',()=>{
    pickingBase=true;$('base-status').textContent='Click the tabletop point directly below the URDF base origin.';
  });
  for (const id of ['base-x','base-y','base-yaw','base-height','base-fixed']) $(id).addEventListener('input',()=>{
    if (id==='base-x' || id==='base-y') { basePixel=null;$('base-pixel').textContent='Measured X/Y';draw(); }
    controls();
  });
  $('save-base').addEventListener('click',async()=>{
    busy=true;controls();$('base-status').textContent='Checking and saving base alignment…';
    try {
      const result=await api('/api/base-alignment',{
        capture_id:capture.capture_id,base_x_mm:$('base-x').valueAsNumber,base_y_mm:$('base-y').valueAsNumber,
        yaw_deg:$('base-yaw').valueAsNumber,base_height_mm:$('base-height').valueAsNumber,
        level_and_fixed:$('base-fixed').checked,pixel:basePixel
      });
      $('base-status').textContent=result.message;
      $('base-status').className=result.motion_ready ? 'good' : 'error';
    } catch(error) { $('base-status').textContent=error.message;$('base-status').className='error'; }
    finally { busy=false;controls(); }
  });
  $('save').addEventListener('click',async()=> {
    busy=true;controls();
    try {
      saved=await api('/api/save',{capture_id:capture.capture_id,pixel_points:points,table_points_mm:world()});
      selecting=false;status('Calibration Successful — click the image to read X/Y coordinates.');
      $('saved-path').textContent=`Saved: ${saved.output_path}`;
    } catch(error) { status(error.message || 'Calibration could not be saved.',true); }
    finally { busy=false;controls(); }
  });
  function renderTeaching(data) {
    taught=data.teaching;teachingMatches=data.teaching_matches;
    $('teach-path').textContent=`Teaching file: ${data.teaching_path}`;
    const items=taught?.points || [];
    $('teach-points').replaceChildren(...Array.from({length:9},(_,index)=>{
      const point=items.find(item=>item.slot===index+1),row=document.createElement('tr');
      for(const value of [index+1,point ? [point.table_mm.x,point.table_mm.y,point.table_mm.z].join(' / ') : 'Not taught',point ? point.joint_ticks.join(', ') : '—']) {
        const cell=document.createElement('td');cell.textContent=value;row.append(cell);
      }
      return row;
    }));
    if(!busy) {
      $('teach-status').textContent=teachingError || data.teaching_error || (!teachingMatches ?
        (saved || data.saved ? 'Camera calibration changed. Start a new teaching set before saving points.' : 'Save camera calibration before teaching robot points.') :
        `${items.length} / 9 points saved. ${taught?.motion_enabled && taught?.physically_validated ? 'Taught controller enabled. Start from a taught pose with mirroring disabled.' : items.length===9 ? 'Preview interpolation, physically validate it, then enable the controller.' : 'Hold the Follower still, enter X/Y/Z, then Save Point.'}`);
      if(!$('teach-z').value && items.length) $('teach-z').value=taught.safe_height_mm;
    }
  }
  for(let slot=1;slot<=9;slot++) {
    const option=document.createElement('option');option.value=slot;option.textContent=slot;$('teach-slot').append(option);
  }
  for(const id of ['teach-x','teach-y','teach-z']) $(id).addEventListener('input',controls);
  $('teach-slot').addEventListener('change',()=>{
    const point=taught?.points.find(item=>item.slot===Number($('teach-slot').value));
    if(point) for(const [id,axis] of [['teach-x','x'],['teach-y','y'],['teach-z','z']]) $(id).value=point.table_mm[axis];
    else { $('teach-x').value='';$('teach-y').value=''; }
    controls();
  });
  $('teach-save').addEventListener('click',async()=>{
    teachingError='';
    busy=true;controls();$('teach-status').textContent='Reading fresh Follower feedback. Hold the robot still…';
    try {
      const data=await api('/api/teach-point',{slot:Number($('teach-slot').value),x_mm:$('teach-x').valueAsNumber,
        y_mm:$('teach-y').valueAsNumber,z_mm:$('teach-z').valueAsNumber});
      previewSignature=null;$('teach-validated').checked=false;$('teach-preview-result').textContent='';
      renderTeaching(data);$('teach-status').textContent=`Point ${$('teach-slot').value} saved (${data.teaching.points.length} / 9).`;
      const next=Array.from({length:9},(_,i)=>i+1).find(slot=>!data.teaching.points.some(point=>point.slot===slot));
      if(next) { $('teach-slot').value=next;$('teach-x').value='';$('teach-y').value=''; }
    } catch(error) { teachingError=error.message;$('teach-status').textContent=error.message; }
    finally { busy=false;controls(); }
  });
  $('teach-reset').addEventListener('click',async()=>{
    if(!confirm('Discard the saved teaching points and start a new set for this camera calibration?')) return;
    teachingError='';
    busy=true;controls();
    try {
      const data=await api('/api/teach-reset',{confirm:true});
      previewSignature=null;$('teach-validated').checked=false;$('teach-preview-result').textContent='';
      renderTeaching(data);$('teach-slot').value=1;$('teach-x').value='';$('teach-y').value='';$('teach-z').value='';
      $('teach-status').textContent='New teaching set started. Enter the measured height and teach point 1.';
    } catch(error) { teachingError=error.message;$('teach-status').textContent=error.message; }
    finally { busy=false;controls(); }
  });
  $('teach-validated').addEventListener('change',controls);
  $('teach-preview').addEventListener('click',async()=>{
    teachingError='';busy=true;controls();
    try {
      const data=await api('/api/teach-preview',{x_mm:$('teach-x').valueAsNumber,y_mm:$('teach-y').valueAsNumber});
      previewSignature=data.signature;
      $('teach-preview-result').textContent=`Estimated table ${data.table_mm.join(', ')} mm → joints 1–8: ${data.joint_ticks.join(', ')} ticks. Preview only; no movement.`;
    } catch(error) { previewSignature=null;teachingError=error.message;$('teach-status').textContent=error.message; }
    finally { busy=false;controls(); }
  });
  $('teach-enable').addEventListener('click',async()=>{
    teachingError='';busy=true;controls();
    try {
      const data=await api('/api/teach-enable',{signature:previewSignature,confirmed:$('teach-validated').checked});
      renderTeaching(data);previewSignature=null;
      $('teach-status').textContent='Taught controller enabled. Start from a taught pose, disable mirroring, then use Mission Control Execute.';
    } catch(error) { teachingError=error.message;$('teach-status').textContent=error.message; }
    finally { busy=false;controls(); }
  });
  for (let i=0;i<4;i++) {
    const row=document.createElement('tr');
    const label=document.createElement('td');label.textContent=String(i+1);
    const pixel=document.createElement('td');pixel.id=`pixel${i}`;
    row.append(label,pixel);
    for (const axis of ['x','y']) {
      const cell=document.createElement('td'),input=document.createElement('input');
      input.id=`${axis}${i}`;input.type='number';input.step='any';input.required=true;
      input.setAttribute('aria-label',`Point ${i+1} ${axis.toUpperCase()} in millimetres`);
      input.addEventListener('input',controls);cell.append(input);row.append(cell);
    }
    $('points').append(row);
  }
  rows();
  async function cameraState() {
    try {
      const data=await api('/api/state');
      renderTeaching(data);
      if(!token) {
        token=data.token;saved=data.saved;
        if(data.base_alignment) {
          const base=data.base_alignment;
          for (const [id,key] of [['base-x','table_xy_mm'],['base-y','table_xy_mm'],['base-yaw','yaw_deg'],['base-height','height_above_table_mm']]) {
            $(id).value=key==='table_xy_mm' ? base[key][id==='base-x'?0:1] : base[key];
          }
          $('base-status').textContent=data.base_alignment_status;
        }
        if(saved) {
          saved.table_points_mm.forEach((xy,i)=>{ $(`x${i}`).value=xy[0];$(`y${i}`).value=xy[1]; });
          $('saved-path').textContent=`Saved: ${data.output_path}`;
        }
      }
      const cameras=data.cameras || [],key=cameras.map(camera=>camera.queue).join('|');
      if(!busy && key!==cameraKey) {
        const previous=$('camera').value;
        $('camera').replaceChildren(...cameras.map((camera,index)=>{
          const option=document.createElement('option');option.value=camera.queue;
          option.textContent=`${index+1}: ${camera.name}${camera.serial ? ' · '+camera.serial : ''}`;return option;
        }));
        const preferred=cameras.find(camera=>camera.queue===previous) || cameras.find(camera=>camera.source===saved?.camera_source) || cameras[Math.min(data.default_camera_index || 0,cameras.length-1)];
        if(preferred) $('camera').value=preferred.queue;
        cameraKey=key;
      }
      if(!image && !busy) status(data.camera_error || (cameras.length ? 'Choose the tabletop camera, then click Capture image.' : 'Waiting for Station camera streams…'),Boolean(data.camera_error && !/^(Connecting|Waiting)/.test(data.camera_error)));
      controls();
    } catch(error) { if(!image && !busy) status(error.message,true); }
    finally { setTimeout(cameraState,1000); }
  }
  void cameraState();
})();"""


class Calibration:
    def __init__(self, output, server=STATION):
        self.output = output
        self.station = CameraReader(server)
        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(32)
        self.capture = None
        self.saved = None
        self.teaching_path = output.with_name("robot_taught_points.json")
        self.teaching = None
        self.teaching_error = None
        try:
            saved = json.loads(output.read_text())
            if saved.get("method") == "manual_four_point":
                self.saved = saved
        except (OSError, ValueError, TypeError):
            pass
        try:
            self.teaching = json.loads(self.teaching_path.read_text())
            if not isinstance(self.teaching, dict) or not isinstance(self.teaching.get("points"), list):
                raise ValueError("Expected a saved teaching set with a points list.")
        except FileNotFoundError:
            pass
        except (OSError, ValueError, TypeError) as exc:
            self.teaching = None
            self.teaching_error = f"Teaching file cannot be loaded: {exc}. Start a new teaching set to replace it."

    def calibration_signature(self):
        if not self.saved:
            raise ValueError("Save the camera calibration first.")
        return homography_signature(self.saved)

    def teaching_state(self):
        with self.lock:
            matches = not self.teaching_error and bool(self.saved) and (self.teaching is None or (
                self.teaching.get("calibration_signature") == self.calibration_signature()
                and self.teaching.get("follower") == FOLLOWER and self.teaching.get("motor_ids") == list(MOTOR_IDS)))
            return {"teaching": self.teaching, "teaching_matches": matches,
                    "teaching_error": self.teaching_error,
                    "teaching_path": str(self.teaching_path.relative_to(ROOT))}

    def new_teaching(self):
        return {"version": 1, "method": "taught_xy_grid", "units": "mm", "follower": FOLLOWER,
                "motor_ids": list(MOTOR_IDS), "camera_unique_id": self.saved["camera_unique_id"],
                "image_size": self.saved["image_size"], "calibration_path": str(self.output.relative_to(ROOT)),
                "calibration_signature": self.calibration_signature(), "safe_height_mm": None,
                "physically_validated": False, "motion_enabled": False, "points": []}

    def write_teaching(self, record):
        self.teaching_path.parent.mkdir(parents=True, exist_ok=True)
        record["saved_at"] = time.time()
        temporary = self.teaching_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.teaching_path)
        self.teaching, self.teaching_error = record, None

    async def follower_pose(self):
        """Read two advancing, stationary samples; never send a motor command."""
        sys.path.insert(0, str(ROOT / "software/station/shared"))
        from station_py.client import Client
        from target.gen_python.protobuf.drivers.st3215 import st3215
        client = Client(self.station.server, logging.getLogger("teaching-feedback"))
        try:
            await client.wait_ready(timeout=2)

            async def sample():
                reader = client.read_from_tail("st3215/inference", offset=b"\x00", limit=1, step=1, buf_size=2)
                entry = await asyncio.wait_for(reader.data.get(), timeout=1.5)
                if entry is None:
                    raise ValueError(f"Follower telemetry is unavailable: {reader.err}")
                state = st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
                feedback = decode_motors(state)
                if [motor["id"] for motor in feedback] != list(MOTOR_IDS):
                    raise ValueError("All eight Follower motors must report exactly once.")
                if any(motor["status"] != "reported" for motor in feedback):
                    raise ValueError("Follower motor fault. Point was not saved.")
                if any(not 0 <= (time.time_ns() - motor["sample_wall_ns"]) / 1e9 < 2 for motor in feedback):
                    raise ValueError("Follower feedback is stale. Point was not saved.")
                for bus in state.get_buses():
                    serial = bus.get_bus().get_serial_number()
                    if isinstance(serial, (bytes, bytearray, memoryview)):
                        serial = bytes(serial).decode("utf-8", "replace")
                    if serial == FOLLOWER:
                        break
                else:
                    raise ValueError("Configured Follower is missing from telemetry.")
                motors = sorted((motor for motor in bus.get_motors() if motor.get_id() in MOTOR_IDS), key=lambda m: m.get_id())
                ticks, ranges = [], []
                for motor in motors:
                    raw = bytes(motor.get_state())
                    if len(raw) < 0x3a:
                        raise ValueError("Follower position registers are incomplete.")
                    encoded = int.from_bytes(raw[0x38:0x3a], "little")
                    position = ((4096 - (encoded & 4095)) if encoded & 0x8000 else encoded) & 4095
                    low, high = int(motor.get_range_min()), int(motor.get_range_max())
                    if not 0 <= low <= 4095 or not 0 <= high <= 4095 or low == high:
                        raise ValueError("Follower motor limits are not calibrated.")
                    if (position - low) % 4096 > (high - low) % 4096:
                        raise ValueError(f"Follower J{motor.get_id()} is outside its calibrated motor limits.")
                    ticks.append(position); ranges.append([low, high])
                return {"joint_ticks": ticks, "joint_ranges_ticks": ranges,
                        "motor_samples": [motor["sample"] for motor in feedback],
                        "sample_wall_ns": [motor["sample_wall_ns"] for motor in feedback]}

            first = await sample()
            await asyncio.sleep(.4)
            deadline = time.monotonic() + 1.5
            while True:
                second = await sample()
                if all(b[0] == a[0] and b[1] > a[1] for a, b in zip(first["motor_samples"], second["motor_samples"])):
                    break
                if time.monotonic() >= deadline:
                    raise ValueError("Follower feedback is not advancing. Point was not saved.")
                await asyncio.sleep(.1)
            if first["joint_ranges_ticks"] != second["joint_ranges_ticks"]:
                raise ValueError("Motor calibration changed while recording. Retry.")
            if max(abs((b - a + 2048) % 4096 - 2048) for a, b in zip(first["joint_ticks"], second["joint_ticks"])) > 8:
                raise ValueError("Follower is still moving. Hold it still, then click Save Point again.")
            return second
        finally:
            client.close_connection()

    def save_teach_point(self, payload):
        import numpy as np
        if set(payload) != {"slot", "x_mm", "y_mm", "z_mm"}:
            raise ValueError("Supply the point number and measured X/Y/Z in millimetres.")
        slot = payload["slot"]
        if type(slot) is not int or not 1 <= slot <= 9:
            raise ValueError("Select point 1 through 9.")
        xyz = [payload[key] for key in ("x_mm", "y_mm", "z_mm")]
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in xyz) or not 1 <= xyz[2] <= 500:
            raise ValueError("Enter finite X/Y and a measured height from 1 to 500mm.")
        with self.lock:
            if not self.teaching_state()["teaching_matches"]:
                raise ValueError(self.teaching_error or "Save camera calibration first, or start a new teaching set if it changed.")
            signature = self.calibration_signature()
            saved = self.saved
            # Homography reference points do not define the taught workspace.
        matches = [cam for cam in self.station.snapshot()["cameras"] if cam.get("unique_id") == saved["camera_unique_id"]]
        if len(matches) != 1:
            raise ValueError("The calibrated tabletop camera must be connected to teach a point.")
        jpeg, camera = self.station.capture_camera(matches[0]["queue"])
        with Image.open(io.BytesIO(jpeg)) as image:
            if list(image.size) != saved["image_size"]:
                raise ValueError("Camera resolution changed. Recalibrate before teaching.")
        try:
            feedback = asyncio.run(asyncio.wait_for(self.follower_pose(), timeout=5))
        except TimeoutError as exc:
            raise ValueError("Timed out reading fresh Follower feedback. Hold still and retry.") from exc
        with self.lock:
            if signature != self.calibration_signature() or not self.teaching_state()["teaching_matches"]:
                raise ValueError("Calibration changed while recording. Start a new teaching set.")
            record = json.loads(json.dumps(self.teaching)) if self.teaching is not None else self.new_teaching()
            others = [point for point in record["points"] if point["slot"] != slot]
            if others and any(abs(point["table_mm"]["z"] - xyz[2]) > 1 for point in others):
                raise ValueError("Teach every point at the same measured gripper height (within 1mm).")
            if others and any(point["joint_ranges_ticks"] != feedback["joint_ranges_ticks"] for point in others):
                raise ValueError("Follower motor calibration changed. Start a new teaching set.")
            if any(math.hypot(point["table_mm"]["x"]-xyz[0], point["table_mm"]["y"]-xyz[1]) < 1 for point in others):
                raise ValueError("Another point already has this X/Y. Spread the nine points across the workspace.")
            point = {"slot": slot, "table_mm": dict(zip(("x", "y", "z"), xyz)),
                     **feedback, "camera_frame_id": camera["frame_id"], "recorded_at": time.time()}
            record["points"] = sorted(others + [point], key=lambda point: point["slot"])
            if len(record["points"]) == 9:
                xy = np.array([[point["table_mm"]["x"], point["table_mm"]["y"]] for point in record["points"]])
                if np.linalg.matrix_rank(xy - xy[0]) < 2:
                    raise ValueError("The nine points must span table X and Y; they cannot all lie on one line.")
            record["safe_height_mm"] = record["points"][0]["table_mm"]["z"]
            record.update(physically_validated=False, motion_enabled=False)
            self.write_teaching(record)
            return self.teaching_state()

    def reset_teaching(self, payload):
        if payload != {"confirm": True}:
            raise ValueError("Confirm starting a new teaching set.")
        with self.lock:
            record = self.new_teaching()
            self.write_teaching(record)
            return self.teaching_state()

    def preview_teaching(self, payload):
        if set(payload) != {"x_mm", "y_mm"} or any(type(value) not in (int, float) or not math.isfinite(value) for value in payload.values()):
            raise ValueError("Enter finite target X/Y coordinates in millimetres.")
        with self.lock:
            mapping = TaughtMapping(path=self.teaching_path, calibration=self.saved, require_validation=False)
            xy = [payload["x_mm"], payload["y_mm"]]
            ticks = mapping.interpolate(xy)
            return {"signature": mapping.signature, "table_mm": [*xy, mapping.safe_height],
                    "joint_ticks": ticks.tolist(), "movement": False}

    def enable_teaching(self, payload):
        if set(payload) != {"signature", "confirmed"} or payload["confirmed"] is not True:
            raise ValueError("Physically validate interpolation and clearance, then confirm the checkbox.")
        with self.lock:
            mapping = TaughtMapping(path=self.teaching_path, calibration=self.saved, require_validation=False)
            if payload["signature"] != mapping.signature:
                raise ValueError("Teaching changed since preview. Preview and validate the current teaching set again.")
            record = mapping.config
            record.update(physically_validated=True, motion_enabled=True,
                          validation={"method": "operator_confirmation", "confirmed_at": time.time(),
                                      "preview_signature": mapping.signature})
            self.write_teaching(record)
            return self.teaching_state()

    def take_picture(self, queue=None):
        jpeg, snapshot = self.station.capture_camera(queue)
        with Image.open(io.BytesIO(jpeg)) as image:
            size = list(image.size)
        capture = {"capture_id": secrets.token_urlsafe(16), "jpeg": jpeg,
                   **snapshot, "image_size": size}
        with self.lock:
            self.capture = capture
        return {key: value for key, value in capture.items() if key != "jpeg"}

    def save(self, payload):
        import cv2
        import numpy as np
        if set(payload) != {"capture_id", "pixel_points", "table_points_mm"}:
            raise ValueError("Supply the captured image ID and four corresponding pixel and XY points.")
        with self.lock:
            capture = self.capture
        if not capture or payload["capture_id"] != capture["capture_id"]:
            raise ValueError("Capture changed or expired. Capture the image again.")
        for key in ("pixel_points", "table_points_mm"):
            points = payload[key]
            if (not isinstance(points, list) or len(points) != 4
                    or any(not isinstance(point, list) or len(point) != 2 for point in points)
                    or any(type(value) not in (int, float) or not math.isfinite(value) for point in points for value in point)):
                raise ValueError("Enter exactly four finite pixel points and four finite X/Y coordinate pairs.")
        pixels, table = np.asarray(payload["pixel_points"], dtype=float), np.asarray(payload["table_points_mm"], dtype=float)
        width, height = capture["image_size"]
        if np.any(pixels < 0) or np.any(pixels[:, 0] >= width) or np.any(pixels[:, 1] >= height):
            raise ValueError("Selected points must be inside the captured image.")
        for name, points in (("Image", pixels), ("Table", table)):
            for a, b, c in itertools.combinations(points, 3):
                first, second = b-a, c-a
                area = abs(first[0]*second[1]-first[1]*second[0])
                if area <= 1e-6 * max(1., np.linalg.norm(first)*np.linalg.norm(second)):
                    raise ValueError(f"{name} points overlap or are collinear. Pick four distinct, spread-out points.")
        matrix, _ = cv2.findHomography(pixels, table, 0)
        if matrix is None or not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e12:
            raise ValueError("Points do not produce a stable homography. Recalibrate with spread-out points.")
        inverse = np.linalg.inv(matrix)
        transformed = np.column_stack([pixels, np.ones(4)]) @ matrix.T
        denominator = transformed[:, 2]
        if np.any(abs(denominator) < 1e-10) or not (np.all(denominator > 0) or np.all(denominator < 0)):
            raise ValueError("Point correspondences cross the projection horizon. Check the X/Y point order.")
        error = float(np.linalg.norm(transformed[:, :2]/denominator[:, None]-table, axis=1).max())
        if error > .1:
            raise ValueError("Homography could not reproduce the supplied coordinates accurately.")
        record = {"method": "manual_four_point", "units": "mm", "camera_source": capture["camera_source"],
                  "camera_queue": capture["camera_queue"], "camera_unique_id": capture["camera_unique_id"],
                  "image_size": capture["image_size"], "frame_id": capture["frame_id"],
                  "pixel_points": pixels.tolist(), "table_points_mm": table.tolist(),
                  "image_to_table": matrix.tolist(), "table_to_image": inverse.tolist(),
                  "max_reprojection_error_mm": error, "saved_at": time.time(),
                  "output_path": str(self.output.relative_to(ROOT))}
        with self.lock:
            if self.capture is not capture:
                raise ValueError("Capture changed. Save again using the current image.")
            self.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.output)
            self.saved = record
        return record

    def save_base_alignment(self, payload):
        required = {"capture_id", "base_x_mm", "base_y_mm", "yaw_deg", "base_height_mm", "level_and_fixed", "pixel"}
        if set(payload) != required:
            raise ValueError("Base alignment fields are incomplete.")
        with self.lock:
            capture, record = self.capture, self.saved
            if not capture or payload["capture_id"] != capture["capture_id"]:
                raise ValueError("Capture changed. Capture the workspace image again.")
            if not record or record.get("camera_unique_id") != capture.get("camera_unique_id"):
                raise ValueError("Save the tabletop calibration for this camera first.")
            values = [payload[key] for key in ("base_x_mm", "base_y_mm", "yaw_deg", "base_height_mm")]
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                raise ValueError("Enter finite base X/Y, orientation, and height measurements.")
            if payload["level_and_fixed"] is not True:
                raise ValueError("Confirm the base/table are level and the setup remains fixed.")
            pixel = payload["pixel"]
            if pixel is not None and (not isinstance(pixel, list) or len(pixel) != 2 or
                                      any(type(v) not in (int, float) or not math.isfinite(v) for v in pixel)):
                raise ValueError("Selected base pixel is invalid.")
            updated = dict(record)
            updated["robot_base_alignment"] = {
                "table_xy_mm": values[:2], "yaw_deg": values[2], "height_above_table_mm": values[3],
                "pixel": pixel, "camera_unique_id": capture["camera_unique_id"],
                "homography_saved_at": record["saved_at"], "level_and_fixed": True,
                "verified": False,
                "note": "Coordinate transform saved; robot kinematics and servo calibration remain prerequisites for motion."}
            temporary = self.output.with_suffix(".tmp")
            temporary.write_text(json.dumps(updated, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.output)
            self.saved = updated
        return {"saved": True, "motion_ready": False,
                "message": "Base alignment saved. Motion remains blocked until robot kinematics, servo calibration, tool offset, and workspace limits are verified."}


class Server(ThreadingHTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "ElRobotCalibration/1"

    def send(self, status, body, mime="application/json"):
        if isinstance(body, dict):
            body = json.dumps(body, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-ElRobot-App", "tabletop-calibration")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def host_ok(self):
        return self.headers.get("Host") in (f"localhost:{self.server.server_port}", f"127.0.0.1:{self.server.server_port}")

    def do_GET(self):
        if not self.host_ok():
            return self.send(403, {"error": "Loopback Host required."})
        path = urlparse(self.path).path
        if path in ("/calibration", "/calibration/"):
            path = "/"
        elif path.startswith("/calibration/"):
            path = path[len("/calibration"):]
        app = self.server.app
        if path == "/":
            return self.send(200, HTML, "text/html; charset=utf-8")
        if path == "/app.js":
            return self.send(200, JS, "application/javascript; charset=utf-8")
        if path == "/api/state":
            cameras = app.station.snapshot()
            with app.lock:
                return self.send(200, {"token": app.token, "saved": app.saved, "output_path": str(app.output.relative_to(ROOT)),
                                       **app.teaching_state(),
                                       "cameras": cameras["cameras"], "camera_error": cameras["error"],
                                       "base_alignment": (app.saved or {}).get("robot_base_alignment"),
                                       "base_alignment_status": "Base alignment saved; robot motion still requires verified kinematics and servo calibration." if (app.saved or {}).get("robot_base_alignment") else "Save camera calibration first, then align the robot base.",
                                       "default_camera_index": CAMERAS["workspace"]})
        if path == "/api/capture.jpg":
            capture_id = parse_qs(urlparse(self.path).query).get("id", [""])[0]
            with app.lock:
                capture = app.capture
            if capture and capture["capture_id"] == capture_id:
                return self.send(200, capture["jpeg"], "image/jpeg")
            return self.send(404, {"error": "Capture expired."})
        self.send(404, {"error": "Not found."})

    def do_POST(self):
        if not self.host_ok():
            return self.send(403, {"error": "Loopback Host required."})
        origin = self.headers.get("Origin")
        if origin and origin not in (f"http://localhost:{self.server.server_port}", f"http://127.0.0.1:{self.server.server_port}"):
            return self.send(403, {"error": "Same-origin requests required."})
        if not secrets.compare_digest(self.headers.get("X-Calibration-Token", ""), self.server.app.token):
            return self.send(403, {"error": "Invalid session token. Reload the page."})
        try:
            self.connection.settimeout(5)
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192 or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Expected a small JSON request.")
            payload = json.loads(self.rfile.read(size))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            path = urlparse(self.path).path
            if path == "/calibration":
                path = "/"
            elif path.startswith("/calibration/"):
                path = path[len("/calibration"):]
            if path == "/api/capture" and not set(payload) - {"camera_queue"}:
                queue = payload.get("camera_queue")
                if queue is not None and not isinstance(queue, str):
                    raise ValueError("Select a camera from the dropdown.")
                return self.send(200, self.server.app.take_picture(queue))
            if path == "/api/save":
                return self.send(200, self.server.app.save(payload))
            if path == "/api/base-alignment":
                return self.send(200, self.server.app.save_base_alignment(payload))
            if path == "/api/teach-point":
                return self.send(200, self.server.app.save_teach_point(payload))
            if path == "/api/teach-reset":
                return self.send(200, self.server.app.reset_teaching(payload))
            if path == "/api/teach-preview":
                return self.send(200, self.server.app.preview_teaching(payload))
            if path == "/api/teach-enable":
                return self.send(200, self.server.app.enable_teaching(payload))
            self.send(404, {"error": "Not found."})
        except ImportError:
            self.send(503, {"error": "OpenCV and NumPy are required in this Python environment."})
        except (ValueError, RuntimeError) as exc:
            self.send(400, {"error": str(exc)[:500]})
        except Exception as exc:
            self.send(500, {"error": f"Calibration could not be saved or captured: {str(exc)[:300]}"})

    def log_message(self, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="Standalone four-point tabletop homography calibration; no motor commands")
    parser.add_argument("--port", type=int, default=8894)
    parser.add_argument("--station", default=STATION, help="Station address for raw camera frames (default: localhost:8888)")
    parser.add_argument("--output", type=Path, default=MANUAL_CALIBRATION)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("--output must be inside this repository.")
    station = args.station if ":" in args.station else args.station + ":8888"
    app = Calibration(output, station)
    try:
        server = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        raise SystemExit(f"Calibration tool could not bind port {args.port}: {exc}. Choose another --port; existing services were left running.") from None
    server.app = app
    app.station.start()
    print(f"Standalone 4-point calibration: http://127.0.0.1:{server.server_port}/calibration", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.station.close()
        server.server_close()


if __name__ == "__main__":
    main()
