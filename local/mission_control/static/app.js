/* One instruction, one Execute click, continuous execution of all eight joints. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const cameraNames = ['workspace', 'wrist'];
  const cameras = Object.fromEntries(cameraNames.map((name) => [name, { key: '', loaded: false }]));
  let state = null;
  let connected = false;
  let executePending = false;
  let stopRequested = false;
  let logsKey = '';
  let settingsSession = null;
  let pollTimer;
  let polling = false;
  let closed = false;
  let speechEnabled = false;
  let spokenRisk = '';
  let speechPending = null;
  let speechUtterance = null;
  let speechTimer;
  let speechGeneration = 0;
  const lastSpoken = new Map();
  let preferredVoice = 'auto';
  try { preferredVoice = localStorage.getItem('mission-alert-voice') || 'auto'; } catch (_) { /* Storage may be unavailable. */ }

  const text = (id, value) => { $(id).textContent = value; };
  const active = () => Boolean(state?.controller.active);
  const milliseconds = (value) => Number.isFinite(value) ? `${Math.round(value)} ms` : '—';
  function settingsError() {
    const duration = $('run-seconds');
    const ticks = $('max-delta-ticks');
    const height = $('safe-height-mm');
    if (!duration.checkValidity()) return `Duration must be a whole number from 1 to ${duration.max} seconds.`;
    if (!ticks.checkValidity()) return `Max step must be a whole number from 1 to ${ticks.max} ticks.`;
    if (!height.checkValidity()) return 'Approach height must be 1–500mm and must fit the verified robot workspace.';
    return '';
  }
  function initializeSettings() {
    if (settingsSession === state.csrf_token) return;
    const limits = state.config.execution_limits;
    if (limits) {
      $('run-seconds').value = limits.duration_seconds;
      $('run-seconds').max = limits.max_duration_seconds;
      $('max-delta-ticks').value = limits.max_delta_ticks;
      $('max-delta-ticks').max = limits.max_delta_ticks_limit;
      $('safe-height-mm').value = limits.safe_height_mm ?? 200;
    }
    settingsSession = state.csrf_token;
  }
  function element(tag, value, className = '') {
    const item = document.createElement(tag);
    if (value !== undefined) item.textContent = String(value);
    item.className = className;
    return item;
  }
  function showError(message = '') {
    text('request-error', message);
    $('request-error').hidden = !message;
  }

  async function api(path, payload) {
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), 5000);
    try {
      const options = { cache: 'no-store', credentials: 'same-origin', signal: abort.signal };
      if (payload !== undefined) {
        if (!state?.csrf_token) throw new Error('Waiting for Mission Control to connect.');
        options.method = 'POST';
        options.headers = { 'Content-Type': 'application/json', 'X-Mission-Token': state.csrf_token };
        options.body = JSON.stringify(payload);
      }
      const response = await fetch(path, options);
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || `Request failed (${response.status}).`);
      return body;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Mission Control is not responding.');
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }

  function blocker() {
    if (!connected) return 'Waiting for Mission Control to connect.';
    if (state.mode === 'simulation') return 'Simulation mode. Robot execution is disabled.';
    if (!state.config.checkpoint_available) return 'SmolVLA checkpoint is unavailable.';
    if (state.job?.status === 'running') return 'Waiting for the previous background job to finish.';
    const hand = state.controller.hand_monitor;
    if (!hand?.restart_allowed) return hand?.reason || 'Waiting for working hand detection.';
    if (!state.station.camera_live?.workspace) return state.station.camera_error || 'Waiting for a fresh workspace camera image.';
    return '';
  }

  function renderControls() {
    const running = state?.controller.state === 'RUNNING';
    const invalidSettings = settingsError();
    $('execute-button').disabled = executePending || active() || Boolean(blocker()) || Boolean(invalidSettings) || !$('instruction').value.trim();
    $('run-seconds').disabled = active() || executePending;
    $('max-delta-ticks').disabled = active() || executePending;
    $('safe-height-mm').disabled = active() || executePending;
    // Independent of Execute requests and inference; always available with a session token.
    $('stop-button').disabled = !state?.csrf_token;
    text('connection-state', connected ? state.mode === 'simulation' ? 'Simulation' : 'Connected' : 'Disconnected');
    const stage = state?.controller.stage;
    const label = state?.controller.error ? 'Error' : state?.controller.state === 'FINISHED' ? 'Finished' : running ?
      stage === 'Calibrating' ? 'Preparing' : stage === 'Detecting object' ? 'Detecting object' :
      stage === 'Moving to safe height' ? 'Moving to safe height' : stage === 'Running VLA' ? 'Running VLA' : stage : 'Stopped';
    text('execution-status', connected ? label : 'Status unavailable');
    $('execution-status').className = running && connected ? 'running' : '';
    text('action-count', `${state?.controller.commands_sent ?? 0} actions`);
    const controller = state?.controller;
    text('run-time', controller?.timer_started ? running ?
      `${Math.ceil(controller.remaining_seconds ?? 0)}s left` : `${Math.floor(controller.elapsed_seconds ?? 0)}s elapsed` : '');
    text('interpreted-instruction', controller?.prompt ? `${controller.interpretation_source || 'Interpreter'} → ${controller.prompt}` : '');
    $('interpreted-instruction').hidden = !controller?.prompt;
    let detail = 'Enter an instruction, then click Execute.';
    if (!connected) detail = 'Connection lost. Robot status is unavailable.';
    else if (state.controller.error) detail = state.controller.error;
    else if (active()) {
      const stage = state.controller.stage;
      detail = state.controller.state === 'STOPPED' ? 'Stopping…' :
        stage === 'Calibrating' ? 'Preparing execution…' :
        stage === 'Detecting object' ? 'Locating the blue block in a captured camera image…' :
        stage === 'Interpreting instruction' ? 'Codex is interpreting your instruction…' :
        stage === 'Loading model' ? 'Loading SmolVLA… execution will start automatically.' :
        stage === 'Waiting for live feedback' ? 'Waiting for fresh camera and motor feedback…' :
        stage === 'Preparing follower' ? 'Preparing robot…' : stage;
    } else if (blocker()) detail = blocker();
    else if (invalidSettings) detail = invalidSettings;
    else if (state.controller.started_at) detail = state.controller.stage;
    else if (executePending) detail = 'Starting execution…';
    text('status-detail', detail);
  }

  function renderCameras() {
    for (const name of cameraNames) {
      const image = $(`camera-${name}`);
      const camera = cameras[name];
      const live = connected && state?.station.camera_live?.[name];
      const frameId = (name === 'workspace' && state?.controller.hand_monitor?.overlay_frame_id) || state?.station.camera_frame_ids?.[name];
      const available = Boolean(frameId);
      image.closest('.camera-card').classList.toggle('stale', !live);
      text(`camera-status-${name}`, !connected ? 'Disconnected' : !available ? 'Waiting' :
        !live ? 'Stale' : camera.loaded ? state.mode === 'simulation' ? 'Simulated' : 'Live' : 'Loading…');
      if (!connected || !available) continue;
      const key = `${state.csrf_token}:${frameId}`;
      if (camera.key !== key) {
        camera.key = key;
        image.src = `/api/camera/${name}.jpg?frame=${encodeURIComponent(frameId)}`;
      }
    }
  }

  function renderHandMonitor() {
    const hand = state?.controller.hand_monitor;
    const risk = !connected || !hand ? 'ERROR' : hand.level;
    const stopped = hand?.latched || risk === 'RED' || risk === 'ERROR';
    const banner = $('hand-warning');
    banner.className = `hand-warning ${stopped ? 'red' : risk === 'YELLOW' ? 'yellow' : risk === 'CLEAR' ? 'clear' : ''}`;
    banner.textContent = !connected ? 'Hand monitor status unavailable. Check the backend connection.' :
      !hand ? 'Waiting for hand monitor…' :
      hand.latched && hand.restart_allowed ? `${hand.reason} STOP remains latched; click Execute to restart deliberately.` : hand.reason;
    const card = $('camera-workspace').closest('.camera-card');
    card.classList.toggle('hand-red', Boolean(stopped));
    card.classList.toggle('hand-yellow', !stopped && risk === 'YELLOW');
    if (!speechEnabled || !('speechSynthesis' in window) || typeof SpeechSynthesisUtterance === 'undefined') return;
    if (risk === spokenRisk || risk === speechPending || speechPending === 'TEST') return;
    const messages = {
      YELLOW: 'Warning. Please keep your hands clear of the robot.',
      RED: 'Execution stopped. Please move your hand out of the work area.',
      ERROR: 'Camera monitoring is unavailable. Execution is blocked.'
    };
    const message = messages[risk];
    if (!message) { spokenRisk = risk; return; }
    const now = Date.now();
    // Speak on transitions only; suppress flickering detections and queued repeats.
    if (now-(lastSpoken.get(risk) || 0) < (risk === 'RED' ? 5000 : 15000)) return;
    speak(message, risk);
  }

  const voiceKey = (voice) => JSON.stringify([voice.voiceURI, voice.name, voice.lang]);
  function rankedVoices() {
    const voices = window.speechSynthesis.getVoices();
    const english = voices.filter(voice => /^en\b/i.test(voice.lang));
    const score = (voice) =>
      (/natural|neural|premium|enhanced/i.test(voice.name) ? 400 : 0) +
      (/google/i.test(voice.name) ? 150 : 0) +
      (/microsoft/i.test(voice.name) ? 100 : 0) +
      (!voice.localService ? 30 : 0) + (/^en-US$/i.test(voice.lang) ? 10 : 0) +
      (voice.default ? 2 : 0) - (/espeak|mbrola/i.test(voice.name) ? 300 : 0);
    return (english.length ? english : voices).sort((a, b) => score(b)-score(a) || a.name.localeCompare(b.name));
  }
  function populateVoices() {
    const picker = $('speech-voice');
    const voices = rankedVoices();
    const automatic = document.createElement('option');
    automatic.value = 'auto'; automatic.textContent = 'Automatic · best available voice';
    picker.replaceChildren(automatic, ...voices.map(voice => {
      const option = document.createElement('option');
      option.value = voiceKey(voice); option.textContent = `${voice.name} · ${voice.lang}`;
      return option;
    }));
    picker.value = voices.some(voice => voiceKey(voice) === preferredVoice) ? preferredVoice : 'auto';
    picker.disabled = !voices.length;
  }

  function speak(message, risk) {
    const generation = ++speechGeneration;
    clearTimeout(speechTimer);
    const synth = window.speechSynthesis;
    const fail = (reason) => {
      if (generation !== speechGeneration) return;
      clearTimeout(speechTimer);
      speechEnabled = false; speechPending = null; spokenRisk = ''; speechUtterance = null;
      const noVoices = synth.getVoices().length === 0;
      text('speech-status', noVoices ? 'No speech voice is available in this browser. Enable a system/browser voice, then click Test voice.' :
        `Speech failed (${reason}). Check browser sound permissions and audio output, then click Test voice.`);
    };
    try {
      synth.cancel(); synth.resume();
      const voices = rankedVoices();
      const voice = voices.find(item => voiceKey(item) === preferredVoice) || voices[0];
      const utterance = new SpeechSynthesisUtterance(message);
      // Retain the utterance until its end/error event and confirm actual playback.
      speechUtterance = utterance; speechPending = risk;
      if (voice) utterance.voice = voice;
      utterance.lang = voice?.lang || 'en-US';
      utterance.volume = 1; utterance.rate = .96; utterance.pitch = 1;
      utterance.onstart = () => {
        if (generation !== speechGeneration) return;
        clearTimeout(speechTimer);
        if (risk !== 'TEST') { spokenRisk = risk; lastSpoken.set(risk, Date.now()); }
        text('speech-status', `Voice warnings enabled${voice ? ` · ${voice.name}` : ''}.`);
      };
      utterance.onend = () => {
        if (generation !== speechGeneration) return;
        clearTimeout(speechTimer);
        speechPending = null; speechUtterance = null;
        renderHandMonitor();
      };
      utterance.onerror = (event) => {
        if (event.error !== 'canceled' && event.error !== 'interrupted') fail(event.error);
      };
      text('speech-status', risk === 'TEST' ? 'Testing voice…' : 'Speaking warning…');
      speechTimer = setTimeout(() => {
        if (generation !== speechGeneration) return;
        fail('playback did not start');
        ++speechGeneration; synth.cancel();
      }, 5000);
      synth.speak(utterance);
    } catch (error) { fail(error.message); }
  }

  function renderJoints() {
    const station = state.station;
    const ready = connected ? station.motors.filter((motor) => motor.status === 'reported').length : 0;
    text('joint-summary', connected ? `${ready} / 8 reporting` : 'Connection lost');
    const rows = [];
    for (let i = 0; i < 8; i += 1) {
      const joint = station.joints[i];
      const motor = station.motors.find((item) => item.id === i + 1);
      const row = element('tr');
      row.append(element('td', `J${i + 1}`));
      const position = element('td');
      const values = element('div', undefined, 'position');
      values.append(element('span', joint?.pos ?? '—'));
      if (Number.isFinite(joint?.pos_norm)) {
        const meter = element('meter');
        meter.min = 0; meter.max = 1; meter.value = joint.pos_norm;
        meter.setAttribute('aria-label', `Joint ${i + 1} normalized position ${joint.pos_norm.toFixed(3)}`);
        values.append(meter);
      }
      position.append(values);
      const torque = motor?.torque_enabled === true ? 'On' : motor?.torque_enabled === false ? 'Off' : '—';
      const status = !connected ? 'Stale' : motor?.status === 'reported' ? 'OK' :
        motor?.status === 'fault' ? 'Fault' : motor?.status === 'stale' ? 'Stale' : 'Waiting';
      const feedback = element('td', status, status === 'Fault' ? 'error' : status === 'OK' ? 'running' : 'muted');
      if (motor?.error) feedback.title = motor.error;
      const action = state.controller.last_action;
      const target = element('td', action?.sent_ticks[i] ?? '—');
      if (action?.clamped_joints?.includes(i + 1)) {
        target.className = 'warning';
        target.title = `Model output ${action.raw_predicted_norm[i].toFixed(4)} was clamped to ${action.predicted_norm[i].toFixed(4)} before the tick step limit.`;
      }
      const delta = action?.delta_ticks[i];
      const limited = action?.limited_joints.includes(i + 1);
      const step = element('td', Number.isFinite(delta) ? `${delta > 0 ? '+' : ''}${delta}` : '—', limited ? 'warning' : '');
      if (limited) step.title = `Model requested target ${action.requested_ticks[i]}; limited to ${action.sent_ticks[i]}.`;
      row.append(position, target, step, element('td', torque), feedback);
      rows.push(row);
    }
    $('joint-rows').replaceChildren(...rows);
  }

  function renderDiagnostics() {
    if (!state) return;
    const controller = state.controller;
    const station = state.station;
    const action = controller.last_action;
    const interpretation = controller.interpretation;
    text('diag-interpreter', controller.interpretation_source || (!state.interpreter.available ? 'OpenCV fallback available' :
      controller.stage === 'Interpreting instruction' && controller.active ? 'Interpreting…' :
      interpretation ? `Accepted ${interpretation.task_type.replaceAll('_', ' ')}` : 'Ready'));
    text('diag-interpretation-time', milliseconds(controller.interpretation_ms));
    text('diag-original-prompt', controller.original_prompt || '—');
    text('diag-interpretation-reason', interpretation?.reason || '—');
    const input = controller.image_input;
    text('diag-image-input', input ? `${input.interpreted ? 'Actual camera JPEG interpreted by Codex' : 'OpenCV fallback; Codex did not interpret the image'} · frame ${input.frame_id} · SHA ${input.sha256.slice(0, 12)}${input.reason ? ` · ${input.reason}` : ''}` : state.interpreter.image_supported ? 'Image attachment interface available; no image sent yet' : 'Codex images unavailable; OpenCV fallback');
    const detection = controller.detection;
    text('diag-detection', detection ? `${detection.object} · ${detection.source} · score ${detection.confidence.toFixed(2)} · pixel ${detection.pixel.u.toFixed(1)},${detection.pixel.v.toFixed(1)} · table ${detection.table_mm.x.toFixed(1)},${detection.table_mm.y.toFixed(1)}mm (approximate projection)` : 'No confident detection yet');
    const approach = controller.approach;
    text('diag-approach', approach ? `${approach.method} · ${approach.position_estimated ? 'estimated ' : ''}table ${approach.target_table_mm.map((value) => value.toFixed(1)).join(', ')}mm${approach.target_robot_mm ? ` · robot ${approach.target_robot_mm.map((value) => value.toFixed(1)).join(', ')}mm` : ' · operator-validated taught height; no live Cartesian height feedback'}` : /(?:ROBOT|TAUGHT) ALIGNMENT REQUIRED|START POSE REQUIRED/.test(controller.error || '') ? controller.error : 'A validated nine-point mapping or measured robot alignment is required');
    text('diag-handoff', `Approach ≤${Math.min(controller.max_delta_ticks, 50)} ticks; VLA ≤${Math.min(controller.max_delta_ticks, 100)} ticks, ≤20 seconds, using current observations`);
    text('diag-verification', controller.verification ? `${controller.verification.status}: ${controller.verification.reason}` : 'No grasp verification result');
    const feedbackAges = station.motors.map((motor) => motor.age_ms).filter(Number.isFinite);
    text('diag-model', `${controller.model_state.replaceAll('_', ' ')}${controller.device ? ` · ${controller.device}` : ''}`);
    text('diag-load', Number.isFinite(controller.load_ms) ? `${(controller.load_ms / 1000).toFixed(1)} s` : '—');
    text('diag-inference', milliseconds(controller.inference_ms));
    text('diag-cycle', milliseconds(controller.cycle_ms));
    text('diag-rate', Number.isFinite(controller.actions_per_second) ? controller.actions_per_second.toFixed(2) : '—');
    text('diag-write', `${controller.write_status || 'No write'}${Number.isFinite(controller.last_write_ms) ? ` · ${milliseconds(controller.last_write_ms)}` : ''}`);
    text('diag-observation', `${!connected ? 'Disconnected' : station.live ? 'Live' : 'Stale / unavailable'} · ${milliseconds(station.observation_age_ms)}${station.frame_id ? ` · frame ${station.frame_id.slice(0, 12)}` : ''}`);
    text('diag-feedback', feedbackAges.length ? `${milliseconds(Math.max(...feedbackAges))} · ${station.motors.length}/8 joints${!connected ? ' · disconnected' : ''}` : 'Waiting for feedback');
    text('diag-mirroring', controller.teleoperation);
    text('diag-limited', action ? action.limited_joints.length ? action.limited_joints.map((id) => `J${id}`).join(', ') : 'None' : 'No action yet');
    const prediction = controller.last_prediction;
    const boundary = prediction?.clamped_joints.length ? prediction.clamped_joints.map((id) =>
      `J${id}: ${prediction.raw[id - 1].toFixed(4)} → ${prediction.bounded[id - 1].toFixed(4)}`).join(', ') : 'None in latest prediction';
    text('diag-clamped', prediction ? `${boundary} · ${controller.clamped_predictions} clamped predictions this run · tolerance ±${controller.prediction_tolerance}` : 'No prediction yet');
    text('diag-station', state.config.station);
    text('diag-follower', state.config.follower);
    text('diag-checkpoint', state.config.checkpoint);
    text('diag-prompt', controller.prompt || '—');
    text('diag-stop', controller.stop_reason || '—');
    const faults = [...(controller.faults || []), station.error, station.motor_error];
    for (const motor of station.motors) {
      if (motor.error_kind || motor.servo_errors?.some(Boolean) || motor.error) {
        faults.push(`J${motor.id}: ${motor.error || 'motor fault'} · error ${motor.error_kind} · servo ${(motor.servo_errors || []).join(', ') || 'none'}`);
      }
    }
    const message = [...new Set(faults.filter(Boolean))].join(' · ');
    text('diag-faults', message);
    $('diag-faults').hidden = !message;
  }

  function renderLogs() {
    const key = `${state.csrf_token}:${state.logs.at(-1)?.id ?? 0}`;
    if (key === logsKey) return;
    logsKey = key;
    const list = $('logs');
    const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 30;
    list.replaceChildren(...state.logs.slice(-50).map((entry) => {
      const row = element('li', undefined, entry.level);
      const time = element('time', new Date(entry.time).toLocaleTimeString());
      time.dateTime = entry.time;
      row.append(time, element('span', entry.message));
      return row;
    }));
    if (atBottom) list.scrollTop = list.scrollHeight;
  }

  async function refresh() {
    if (polling || closed) return;
    polling = true;
    try {
      state = await api('/api/state');
      connected = true;
      initializeSettings();
      renderJoints();
      renderLogs();
    } catch (error) {
      connected = false;
      if (state) renderJoints();
    } finally {
      polling = false;
      renderHandMonitor();
      renderControls();
      renderCameras();
      renderDiagnostics();
    }
  }
  async function poll() {
    await refresh();
    if (!closed) pollTimer = setTimeout(poll, 500);
  }

  $('instruction').addEventListener('input', renderControls);
  const speechSupported = 'speechSynthesis' in window && typeof SpeechSynthesisUtterance !== 'undefined';
  const enableSpeech = () => {
    if (!speechSupported) return;
    speechEnabled = true; spokenRisk = ''; lastSpoken.clear();
    // Call speak inside the actual user gesture to unlock browser audio.
    speak('Voice alerts are ready.', 'TEST');
  };
  $('speech-button').disabled = !speechSupported;
  if (!speechSupported) {
    $('speech-voice').disabled = true;
    text('speech-status', 'This browser does not support text-to-speech. Visual warnings and backend STOP remain active.');
  }
  else {
    populateVoices();
    window.speechSynthesis.addEventListener('voiceschanged', () => {
      populateVoices();
      if (!speechEnabled && !speechPending && window.speechSynthesis.getVoices().length) {
        text('speech-status', 'Browser voice available. Click Test voice to enable spoken alerts.');
      }
    });
  }
  $('speech-button').addEventListener('click', enableSpeech);
  $('speech-voice').addEventListener('change', () => {
    preferredVoice = $('speech-voice').value;
    try { localStorage.setItem('mission-alert-voice', preferredVoice); } catch (_) { /* Keep the choice for this session. */ }
    enableSpeech();
  });
  const enableOnInteraction = (event) => {
    if (!speechEnabled && !speechPending && !event.target.closest('.voice-controls')) enableSpeech();
  };
  document.addEventListener('pointerdown', enableOnInteraction, { once: true });
  document.addEventListener('keydown', enableOnInteraction, { once: true });
  $('run-seconds').addEventListener('input', renderControls);
  $('max-delta-ticks').addEventListener('input', renderControls);
  $('safe-height-mm').addEventListener('input', renderControls);
  $('execute-button').addEventListener('click', async () => {
    if ($('execute-button').disabled) return;
    executePending = true;
    stopRequested = false;
    showError();
    renderControls();
    try {
      await api('/api/execute', {
        prompt: $('instruction').value.trim(),
        duration_seconds: $('run-seconds').valueAsNumber,
        max_delta_ticks: $('max-delta-ticks').valueAsNumber,
        safe_height_mm: $('safe-height-mm').valueAsNumber,
      });
      // Also stop a start request that was still in flight when STOP was clicked.
      if (stopRequested) await api('/api/stop', {});
      await refresh();
    } catch (error) {
      showError(error.message);
    } finally {
      executePending = false;
      renderControls();
    }
  });
  $('stop-button').addEventListener('click', async () => {
    stopRequested = true;
    showError();
    try {
      await api('/api/stop', {});
      await refresh();
    } catch (error) {
      showError(`STOP failed: ${error.message}`);
    }
  });
  for (const name of cameraNames) {
    const image = $(`camera-${name}`);
    image.addEventListener('load', () => {
      cameras[name].loaded = true;
      image.hidden = false;
      $(`camera-placeholder-${name}`).hidden = true;
      renderCameras();
    });
    image.addEventListener('error', () => {
      cameras[name].loaded = false;
      cameras[name].key = '';
      image.hidden = true;
      $(`camera-placeholder-${name}`).hidden = false;
      text(`camera-placeholder-${name}`, 'Camera unavailable');
      text(`camera-status-${name}`, 'Unavailable');
    });
  }
  window.addEventListener('pagehide', () => {
    closed = true; clearTimeout(pollTimer);
    ++speechGeneration; clearTimeout(speechTimer); speechPending = null; speechUtterance = null;
    if ('speechSynthesis' in window) window.speechSynthesis.cancel();
  });
  window.addEventListener('pageshow', (event) => { if (event.persisted) { closed = false; void poll(); } });
  void poll();
})();
