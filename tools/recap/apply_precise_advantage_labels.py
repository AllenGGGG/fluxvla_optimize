#!/usr/bin/env python
"""Phase 2, final design: precise per-frame advantage labeling, fused against
an existing (rough, imprecise) advantage=0 signal -- never inventing a new
error window the rough signal never pointed at, and never asking a VLM to
guess "when exactly" or "what happened" in one holistic shot. Instead, the
VLM is only ever asked objective per-checkpoint facts at specific times state
data already supplies (a grasp closure, a push-motion peak, the clip's last
frame, or -- in rough_pad mode -- an extra evenly-spaced timepoint); the
(status, precise window) decision is derived from those facts in plain code,
using the checkpoints' own exact timestamps as the window boundary. See
CycleCheckpointObservation / resolve_cycle_checkpoints for the grasp/drop/
reposition/failed path, and PushCheckpointObservation / resolve_push_retry
for the push-only path.

Two independently comparable window strategies, since neither has an
obviously-correct answer without real data to test against:

--window-strategy cycle:
    Take the STATE-SEGMENTED cycle containing/overlapping the rough window
    (segment_episode) and hand the VLM the WHOLE cycle clip -- more context
    (a real "before" reference), coarser (event-driven) checkpoint density.

--window-strategy rough_pad:
    Take the rough advantage=0 run directly, pad it by ROUGH_PAD seconds on
    each side, and hand the VLM only that -- less context, but denser
    checkpoints (event-driven ones plus an evenly-spaced grid across the
    padded window), since this mode's whole point is pinpointing exactly
    where within an already-roughly-known region the state actually changed.

Fusion gate (the actual safety mechanism, applies regardless of window
strategy): a cycle's VLM classification is only trusted -- and only then
converted into a precise window via state data (the checkpoint timestamps
themselves) -- if it overlaps an EXISTING rough advantage=0 region already
present in the episode's parquet. No overlap = discarded, not a new window.
On this old `lerobot_data/recap_converted` dataset, episodes with no rough
data at all (nothing seeded by build_fake_rough_advantage.py) therefore
converge to all advantage=1 automatically, with no per-episode special
casing needed.

pilot_state_guided.py and state_segmentation.py are both read-only
dependencies -- unmodified, and no longer imported either: the specific
functions this script needs from them (and from pilot_parcel_sort.py) are
inlined verbatim below instead of imported, so this file has no cross-file
dependency on the other recap scripts. This is a snapshot, not a live
symlink -- if the originals change, these copies do not update automatically.

Usage:
    python tools/recap/apply_precise_advantage_labels.py \
        --dataset-root lerobot_data/recap_converted \
        --model /data/wqz/openpi/models/Qwen3-VL-8B-Instruct \
        --device cuda:3 \
        --episodes 0 1 2 3 4 5 6 7 \
        --window-strategy cycle \
        --output-tag cycle
"""

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, Field
from scipy.signal import find_peaks
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_model(model_path: str, device: str, dtype: torch.dtype):
    print(f'Loading model from {model_path} on {device}...')
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True)
    print('Model loaded.')
    return model, processor


# --- Grasp/drop/reposition/failed classification: checkpoint-based ---------
# prompt, response schema, VLM call. Never asks "what happened" or "when
# exactly" in one holistic shot -- only objective per-checkpoint facts at
# times state data already supplies (see module docstring).

GRASP_ORIGIN_VALUES = ['not_yet_secured', 'placed_on_surface',
                       'dropped_on_floor']
PACKAGE_LOCATION_VALUES = ['in_gripper', 'on_table_or_surface',
                          'on_floor_dropped', 'on_conveyor']


class CycleCheckpointObservation(BaseModel):
    description: str = Field(
        description='3-5 sentences describing what happens across the '
        'WHOLE clip, start to finish, in chronological order -- written '
        'BEFORE looking at the specific checkpoint questions below')
    grasp_answers: list[Literal['not_yet_secured', 'placed_on_surface',
                                'dropped_on_floor']] = Field(
        default_factory=list,
        description='For each grasp checkpoint listed, in the same order: '
        'what state the package was in IMMEDIATELY BEFORE that specific '
        'gripper closure')
    push_answers: list[bool] = Field(
        default_factory=list,
        description='For each push checkpoint listed, in the same order: '
        'true if the package has already reached/entered the conveyor '
        'chute by that moment, false if still on the table / not there yet')
    final_answer: Literal['in_gripper', 'on_table_or_surface',
                          'on_floor_dropped', 'on_conveyor'] = Field(
        description='Where the package ends up by the very last frame of '
        'the clip')
    grid_answers: list[Literal['in_gripper', 'on_table_or_surface',
                               'on_floor_dropped', 'on_conveyor']] = Field(
        default_factory=list,
        description='For each extra timepoint listed (if any), in the same '
        'order: where the package is at that moment -- only used to narrow '
        'down exactly when a transition happens, not to judge the outcome')
    evidence: str = Field(
        description='One short sentence per checkpoint answered above '
        '(grasp checkpoints, then push checkpoints, then the final '
        'checkpoint, then any extra timepoints), in that order')


def build_cycle_checkpoint_prompt(grasp_times_relative: list,
                                  push_times_relative: list,
                                  final_time_relative: float,
                                  grid_times_relative: list,
                                  is_padded_window: bool) -> str:
    """Two-stage, narrative-free prompt: Step 1 asks for a chronological
    description of the whole clip (context, matching build_push_retry_prompt
    below); Step 2 asks only objective per-checkpoint facts, never "what
    happened overall" or "when exactly" -- resolve_cycle_checkpoints derives
    both the status and its precise (state-timestamp) boundary from those
    facts afterward.
    """
    role_context = (
        'This clip is a padded window extended around an existing rough '
        'advantage=0 mark from an earlier, imprecise labeling pass -- your '
        'job is to pinpoint EXACTLY when the package\'s state changes '
        'inside this window, not to judge the whole clip as one atomic '
        'attempt.' if is_padded_window else
        'This clip shows ONE complete attempt at a bimanual parcel-sorting '
        'cycle: a hand picks up a package, places it on a table, then a '
        'hand pushes it onto a conveyor belt.')

    location_options = (
        '"in_gripper" (still held) / "on_table_or_surface" (resting on a '
        'table/surface, not being held) / "on_floor_dropped" (fallen on the '
        'floor) / "on_conveyor" (already on the conveyor belt)')

    sections = []
    n_grasp = len(grasp_times_relative)
    if grasp_times_relative:
        times_str = ', '.join(f'{t:.1f}s' for t in grasp_times_relative)
        sections.append(f"""\
# Grasp checkpoints ({times_str}) -- {n_grasp} of them
The robot's gripper closes at each of these moments (state data, not a
guess). For EACH one, in order, answer with EXACTLY one of these three
strings (nothing else) describing the package's state IMMEDIATELY BEFORE
that specific closure:
- "not_yet_secured": nothing was being held yet -- this is either the first
  pickup of the cycle, or an earlier attempt at this same pickup that had
  not yet succeeded.
- "placed_on_surface": the package had already been set down on a table/
  surface (not dropped) and is now being re-grasped, e.g. to reposition it.
- "dropped_on_floor": the package had fallen/been dropped onto a surface
  below and is now being re-grasped after that drop.
This must produce a list of exactly {n_grasp} string(s), in order.""")
    n_push = len(push_times_relative)
    if push_times_relative:
        times_str = ', '.join(f'{t:.1f}s' for t in push_times_relative)
        sections.append(f"""\
# Push checkpoints ({times_str}) -- {n_push} of them
At each of these moments, has the package already reached / entered the
conveyor belt chute? Answer with EXACTLY the JSON boolean true or false for
each -- true if already on the conveyor, false if still on the table / not
there yet. Answer independently for each -- do not assume the answer is the
same for every moment. This must produce a list of exactly {n_push}
boolean(s), in order.""")
    sections.append(f"""\
# Final checkpoint ({final_time_relative:.1f}s, the last moment of the clip)
Where does the package end up? Answer with EXACTLY one of these four
strings (nothing else): {location_options}.""")
    n_grid = len(grid_times_relative)
    if grid_times_relative:
        times_str = ', '.join(f'{t:.1f}s' for t in grid_times_relative)
        sections.append(f"""\
# Extra timepoints ({times_str}) -- {n_grid} of them
For each of these additional moments, where is the package? Answer with
EXACTLY one of these four strings (nothing else) per timepoint, in order:
{location_options}. These exist only to help narrow down exactly when a
transition happens, evenly spaced across the clip. This must produce a list
of exactly {n_grid} string(s), in order.""")
    checkpoints_block = '\n\n'.join(sections)

    return f"""\
# Role
You are a Robotics Vision System. {role_context}

# Step 1 - Describe the whole clip
Before answering anything else, describe in 3-5 sentences what happens
across the ENTIRE clip, from the first frame to the last, in chronological
order -- track the package continuously.

# Step 2 - Answer the checkpoints below
Using your Step 1 description, answer each checkpoint question below,
using ONLY the exact strings/booleans specified for that checkpoint -- do
not paraphrase or invent your own wording. Do NOT over-report: normal
settling, a small shift on contact, and a single continuous (if slightly
uneven) motion are not anomalies -- only report a transition away from
"not_yet_secured" if you see CLEAR, UNAMBIGUOUS visual evidence of it.

{checkpoints_block}

# Output
Write your Step 1 description, then one short sentence of evidence per
checkpoint answered above (grasp, then push, then final, then extra
timepoints, in that order), then output ONLY this JSON, with EXACTLY
{n_grasp} entries in "grasp_answers", EXACTLY {n_push} entries in
"push_answers", and EXACTLY {n_grid} entries in "grid_answers" (an empty
list [] if a count above is 0):
{{"description": "...", "grasp_answers": [...], "push_answers": [...],
"final_answer": "...", "grid_answers": [...], "evidence": "..."}}
"""


def _run_vlm_json(model, processor, device: str, clip_path: Path,
                  clip_duration: float, prompt: str, response_model,
                  max_retries: int = 3):
    """Shared by classify_cycle and classify_push_retry: send one video clip
    + prompt to the model, parse the JSON reply into response_model, retry up
    to max_retries times on parse failure. Returns (raw_text, parsed | None).

    Passes return_video_metadata=True through to the processor so it can
    find the clip's true sample fps (10.0 below) instead of silently
    defaulting to fps=24 when computing each frame's "<X.X seconds>" prompt
    label -- confirmed this compresses the model's own sense of elapsed time
    by ~2.4x for an fps=10-sampled clip when metadata is missing.
    do_sample_frames=False (in video_kwargs) additionally stops the
    processor from re-sampling frames qwen_vl_utils already sampled.
    """
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'type': 'video',
                    'video': str(clip_path),
                    'fps': 10.0,
                },
                {
                    'type': 'text',
                    'text': f'{prompt}\n\nThis clip is exactly '
                    f'{clip_duration:.1f} seconds long.'
                },
            ],
        },
    ]

    last_response = ''
    for _ in range(max_retries):
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, return_video_metadata=True)
        videos = [v[0] for v in video_inputs]
        video_metadata = [v[1] for v in video_inputs]
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadata,
            padding=True,
            return_tensors='pt',
            **video_kwargs,
        ).to(device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, max_new_tokens=768, do_sample=False)
        response = processor.batch_decode(
            [
                out[len(inp):]
                for inp, out in zip(inputs.input_ids, generated_ids, strict=True)
            ],
            skip_special_tokens=True,
        )[0].strip()
        last_response = response

        json_str = response
        if '```json' in response:
            json_str = response.split('```json')[1].split('```')[0]
        elif '```' in response:
            json_str = response.split('```')[1].split('```')[0]
        try:
            return last_response, response_model.model_validate(
                json.loads(json_str))
        except Exception:
            match = re.search(r'\{.*\}', json_str, re.DOTALL)
            if match:
                try:
                    return last_response, response_model.model_validate(
                        json.loads(match.group()))
                except Exception:
                    pass
    return last_response, None


def classify_cycle_checkpoints(model, processor, device: str, clip_path: Path,
                               clip_duration: float,
                               grasp_times_relative: list,
                               push_times_relative: list,
                               final_time_relative: float,
                               grid_times_relative: list,
                               is_padded_window: bool, max_retries: int = 3):
    prompt = build_cycle_checkpoint_prompt(
        grasp_times_relative, push_times_relative, final_time_relative,
        grid_times_relative, is_padded_window)
    raw, observation = _run_vlm_json(
        model, processor, device, clip_path, clip_duration, prompt,
        CycleCheckpointObservation, max_retries)
    if observation is not None and (
            len(observation.grasp_answers) != len(grasp_times_relative) or
            len(observation.push_answers) != len(push_times_relative) or
            len(observation.grid_answers) != len(grid_times_relative)):
        observation = None  # malformed length(s) -- treat like a parse failure
    return raw, observation


# --- State-signal detection and cycle segmentation --------------------------

STATE_NAMES = [
    'body_joint1', 'body_joint2', 'body_joint3', 'body_joint4', 'head_joint1',
    'head_joint2', 'left_joint1', 'left_joint2', 'left_joint3', 'left_joint4',
    'left_joint5', 'left_joint6', 'left_joint7', 'right_joint1',
    'right_joint2', 'right_joint3', 'right_joint4', 'right_joint5',
    'right_joint6', 'right_joint7', 'left_hand_thumb_joint1',
    'left_hand_thumb_joint2', 'left_hand_index_joint',
    'left_hand_middle_joint', 'left_hand_ring_joint', 'left_hand_pinky_joint',
    'right_hand_thumb_joint1', 'right_hand_thumb_joint2',
    'right_hand_index_joint', 'right_hand_middle_joint',
    'right_hand_ring_joint', 'right_hand_pinky_joint'
]
GRASP_JOINT = 'left_hand_index_joint'
CLOSE_THRESHOLD = 0.25
RIGHT_ARM_JOINTS = [f'right_joint{i}' for i in range(1, 8)]

# Right-arm push-motion peak detection (smoothed joint-space speed).
PUSH_SMOOTH_WINDOW = 5
PUSH_PEAK_MIN_DISTANCE_S = 1.5
PUSH_PEAK_PROMINENCE = 0.03

# Fixed lookback applied to each cycle's start to cover the reach-in motion
# before the grasp actually closes (state alone doesn't give a clean,
# consistent "arm starts moving" signal).
REACH_IN_LOOKBACK_S = 1.0

# Originally set to 0.5 to try to filter "incidental" small bumps next to a
# dominant peak, but a 448-episode statistical sweep found no clean bimodal
# separation on this feature (or on arm-position shape features) - it does
# not reliably distinguish genuine second push attempts from a single push's
# internal two-phase speed profile at this sample size. Left deliberately
# loose (0.0 = no filtering) - the state signal's job is only to flag "maybe
# look here", with the actual normal/anomaly call left to the VLM.
PUSH_RETRY_MIN_HEIGHT_RATIO = 0.0


def _load_state(parquet_path: Path):
    df = pd.read_parquet(parquet_path)
    state = np.stack(df['observation.state'].values)
    t = df['timestamp'].values
    return state, t


def detect_grasp_close_events(state: np.ndarray, t: np.ndarray):
    grasp_idx = STATE_NAMES.index(GRASP_JOINT)
    closed = (state[:, grasp_idx] > CLOSE_THRESHOLD).astype(int)
    rising = np.where(np.diff(closed) == 1)[0] + 1
    return t[rising]


def detect_right_push_peaks(state: np.ndarray, t: np.ndarray):
    """Returns (peak_times, peak_heights)."""
    right_arm_idx = [STATE_NAMES.index(j) for j in RIGHT_ARM_JOINTS]
    right_arm = state[:, right_arm_idx]
    vel = np.linalg.norm(np.diff(right_arm, axis=0), axis=1)
    vel = np.concatenate([[0.0], vel])
    k = PUSH_SMOOTH_WINDOW
    smooth = np.convolve(vel, np.ones(k) / k, mode='same')

    fps = 1.0 / np.median(np.diff(t)) if len(t) > 1 else 30.0
    min_distance = max(1, int(PUSH_PEAK_MIN_DISTANCE_S * fps))
    peaks, _ = find_peaks(smooth, distance=min_distance,
                          prominence=PUSH_PEAK_PROMINENCE)
    return t[peaks], smooth[peaks]


def cluster_into_cycles(grasp_events: np.ndarray, push_times: np.ndarray,
                        push_heights: np.ndarray, episode_end: float):
    if len(grasp_events) == 0:
        return []

    cycles = [[float(grasp_events[0])]]
    for prev_ev, ev in zip(grasp_events[:-1], grasp_events[1:]):
        push_happened = np.any((push_times > prev_ev) & (push_times < ev))
        if push_happened:
            cycles.append([float(ev)])
        else:
            cycles[-1].append(float(ev))

    result = []
    for i, cycle_events in enumerate(cycles):
        raw_start = cycle_events[0]
        if i == 0:
            start = 0.0
        else:
            prev_end = cycles[i - 1][-1]
            start = max(prev_end, raw_start - REACH_IN_LOOKBACK_S)
        end = cycles[i + 1][0] if i + 1 < len(cycles) else episode_end
        if i + 1 < len(cycles):
            push_confirmed = True
        else:
            push_confirmed = bool(
                np.any((push_times > cycle_events[-1]) & (push_times < end)))
        in_window = (push_times >= start) & (push_times < end)
        window_heights = push_heights[in_window]
        if len(window_heights) > 0:
            height_threshold = window_heights.max() * PUSH_RETRY_MIN_HEIGHT_RATIO
            push_count = int(np.sum(window_heights >= height_threshold))
        else:
            push_count = 0
        result.append({
            'grasp_events': cycle_events,
            'has_retry': len(cycle_events) > 1,
            'start': start,
            'end': end,
            'push_confirmed': push_confirmed,
            'push_peak_count': push_count,
            'has_push_retry': push_count > 1,
        })
    return result


def segment_episode(parquet_path: Path):
    state, t = _load_state(parquet_path)
    grasp_events = detect_grasp_close_events(state, t)
    push_times, push_heights = detect_right_push_peaks(state, t)
    return cluster_into_cycles(grasp_events, push_times, push_heights,
                               float(t[-1]))


# --- Video clip extraction ---------------------------------------------------

CLIP_PAD = 0.6  # matches pilot_state_guided.py's own --pad default


def extract_clip(video_path: Path, start: float, end: float, pad: float,
                 out_path: Path) -> tuple[float, float]:
    padded_start = max(0.0, start - pad)
    duration = (end - start) + 2 * pad
    subprocess.run(
        [
            # -ss AFTER -i (not before): frame-accurate output seeking, not
            # fast/keyframe-snapping input seeking. -ss before -i can start
            # the real output up to a keyframe-interval early, desyncing the
            # clip's true t=0 from `padded_start` -- every relative-second
            # value computed against padded_start elsewhere in this script
            # (push_times_relative, check_times_relative) then points at the
            # wrong frame. No extra cost since this command already
            # re-encodes (-c:v libx264) rather than stream-copying.
            'ffmpeg', '-y', '-i',
            str(video_path), '-ss',
            str(padded_start), '-t',
            str(duration), '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
            str(out_path)
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True)
    return duration, padded_start


# --- Rough advantage=0 window discovery (the fusion gate's input) ----------


def find_rough_windows(parquet_path: Path) -> list:
    """Contiguous advantage=0 runs currently in the parquet -- the "existing,
    imprecise signal" this whole script fuses against. Read BEFORE this
    script's own write_advantage_column() call overwrites the column, so
    call this once per episode up front, not after refinement.
    """
    table = pq.read_table(parquet_path)
    if 'advantage' not in table.column_names:
        return []
    advantage = table.column('advantage').to_numpy()
    timestamps = table.column('timestamp').to_numpy()
    order = np.argsort(timestamps)
    advantage, timestamps = advantage[order], timestamps[order]

    windows = []
    in_run = False
    run_start = None
    for i in range(len(advantage)):
        if advantage[i] == 0 and not in_run:
            in_run = True
            run_start = timestamps[i]
        elif advantage[i] == 1 and in_run:
            in_run = False
            windows.append((float(run_start), float(timestamps[i])))
    if in_run:
        windows.append((float(run_start), float(timestamps[-1]) + 1e-6))
    return windows


def overlaps_any(start: float, end: float, rough_windows: list) -> bool:
    return any(start < r_end and end > r_start
              for r_start, r_end in rough_windows)


# --- Push-retry precise classification --------------------------------------

PUSH_COMPLETION_DELAY = 1.2  # seconds a push motion takes to finish after
# its detected speed-peak (empirically ~1.0-1.2s on this dataset, from
# frame-level ground truth on episode 2 cycle3: package reached the conveyor
# ~1.1s after its (sole, genuine) push peak). Checking position AT the peak
# itself catches the push mid-motion -- of course the package "hasn't
# arrived yet" at that exact instant even for a perfectly normal single
# push, which earlier testing showed the model was misreading as "the push
# failed." Checking peak_time + PUSH_COMPLETION_DELAY instead asks about
# push OUTCOME (did this attempt finish successfully), not push-in-progress.


class PushCheckpointObservation(BaseModel):
    """Deliberately NOT a status label. See build_push_retry_prompt's
    docstring: telling the model about "push attempts"/"speed peaks" and
    asking it to classify normal-vs-retry directly gave it a narrative
    template to echo instead of forcing it to actually look -- verified via
    raw text where a real retry and a genuine false positive got near-
    identical "continuous push" evidence. Asking only for a per-timestamp
    objective fact (is the package on the conveyor yet, yes/no) and doing
    the normal/retry classification in plain code afterward removes that
    narrative bias from the decision entirely.
    """
    description: str = Field(
        description='2-4 sentences describing what happens across the WHOLE '
        'clip, start to finish -- the package\'s movement throughout, '
        'written BEFORE looking at the specific check-point moments below')
    on_conveyor: list[bool] = Field(
        description='For each check-point time listed, in the same order: '
        'true if the package has already reached/entered the conveyor '
        'chute by that moment, false if it is still on the table / not '
        'there yet')
    evidence: str = Field(
        description='One short sentence per check-point describing exactly '
        'what is seen at that moment')


def build_push_retry_prompt(check_times_relative: list) -> str:
    """Deliberately minimal and narrative-free -- no mention of "sensor
    signal", "speed peaks", or "push attempts" (see PushCheckpointObservation
    docstring for why). Just gives the model concrete timestamps (state
    data, not a guess -- push_peak_time + PUSH_COMPLETION_DELAY so each
    check-point lands after that push motion should have finished, not
    mid-motion).

    Two-step structure (describe the WHOLE clip first, THEN answer the
    specific check-points) -- added after finding the single-step version
    jumping straight to the check-point questions caused the model to
    fixate near the requested seconds instead of actually watching the
    clip's full package trajectory, misreading a mid-push "not there yet"
    moment for one cycle the same way it misread a genuinely-already-
    finished push for another.
    """
    times_str = ', '.join(f'{t:.1f}s' for t in check_times_relative)
    return f"""\
# Role
You are a Robotics Vision System watching a short clip of a robot placing a
package onto a conveyor belt.

# Step 1 - Describe the whole clip
Before answering anything else, describe in 2-4 sentences what happens
across the ENTIRE clip, from the first frame to the last -- track the
package continuously: when does it move, when does it pause, when (if ever)
does it reach the conveyor belt chute.

# Step 2 - Answer specific moments
Using your Step 1 description, answer for EACH of the following specific
moments -- {times_str} (measured from the start of the clip, i.e. t=0 is the
first frame): has the package already reached / entered the conveyor belt
chute by that moment, or is it still on the table / not there yet?

Answer independently for each moment, in the order listed. Do not assume the
answer is the same for every moment.

# Output
Write your Step 1 description, then one short sentence per check-point for
Step 2, then output ONLY this JSON:
{{"description": "...", "on_conveyor": [true_or_false, ...], "evidence": "..."}}
The "on_conveyor" list must have exactly {len(check_times_relative)} entries,
in the same order as the moments listed above.
"""


def classify_push_retry(model, processor, device: str, clip_path: Path,
                        clip_duration: float,
                        push_peak_times_relative: list, max_retries: int = 3):
    """Runs build_push_retry_prompt over the clip and returns
    (raw_text, PushCheckpointObservation | None). Kept here (not in
    pilot_state_guided.py, which stays unmodified) since it is specific to
    this fusion pipeline's push-only candidates.

    push_peak_times_relative are the raw detected speed-peak times; this
    function converts them to "did the push finish" check-points (peak +
    PUSH_COMPLETION_DELAY, clamped to the clip) before building the prompt --
    see PUSH_COMPLETION_DELAY's docstring for why checking exactly at the
    peak (mid-motion) was misleading the model.
    """
    check_times_relative = [
        min(t + PUSH_COMPLETION_DELAY, clip_duration)
        for t in push_peak_times_relative]
    prompt = build_push_retry_prompt(check_times_relative)
    raw, observation = _run_vlm_json(model, processor, device, clip_path,
                                     clip_duration, prompt,
                                     PushCheckpointObservation, max_retries)
    if observation is not None and len(observation.on_conveyor) != len(
            check_times_relative):
        observation = None  # malformed length -- treat like a parse failure
    return raw, observation


def resolve_push_retry(observation, push_times_in_window):
    """Plain-code decision from PushCheckpointObservation's objective
    per-check-point facts -- no VLM-authored status label involved, so the
    prompt-narrative bias documented above can't influence the outcome.

    Window starts at push_times_in_window[0] (the FIRST, failed, push peak)
    -- not the cycle's clip_start, which also covers the normal pick-and-
    place phase before any push is even attempted. Anchoring to clip_start
    was verified to inflate the window by several seconds of entirely
    normal motion (episode 2 cycle2: clip_start=11.23s vs the real first
    push attempt at 14.87s).
    """
    if observation is None or not len(push_times_in_window):
        return 'PARSE_FAILED', None
    if observation.on_conveyor[0]:
        return 'normal', None  # first push attempt already succeeded
    for i, reached in enumerate(observation.on_conveyor):
        if reached:
            return ('recovered_after_push_retry',
                    (float(push_times_in_window[0]),
                     float(push_times_in_window[i])))
    # Never confirmed on the conveyor at any check-point -- conservative:
    # this branch's job is detecting push retries specifically, not general
    # cycle failures, so it does not invent a 'failed' window here.
    return 'normal', None


def resolve_cycle_checkpoints(observation, grasp_times_abs, push_times_abs,
                              clip_start: float, clip_end: float):
    """Plain-code decision from CycleCheckpointObservation's objective
    per-checkpoint facts -- mirrors resolve_push_retry's philosophy: the VLM
    never reports a status label or a raw timestamp, only what it sees at
    specific state-supplied moments. Both the status AND its precise window
    boundary come from where those per-checkpoint facts change, using the
    checkpoints' own exact timestamps -- never a [-1]-index heuristic.
    """
    if observation is None:
        return 'PARSE_FAILED', None

    if observation.final_answer != 'on_conveyor':
        return 'failed', (clip_start, clip_end)

    for i in range(1, len(observation.grasp_answers)):
        answer = observation.grasp_answers[i]
        if answer == 'placed_on_surface':
            return ('recovered_after_reposition',
                    (float(grasp_times_abs[0]), float(grasp_times_abs[i])))
        if answer == 'dropped_on_floor':
            return ('recovered_after_drop',
                    (float(grasp_times_abs[0]), float(grasp_times_abs[i])))
    if len(observation.grasp_answers) > 1:
        # Every recovery closure still reads as "not yet secured" -- a
        # repeated-miss pattern rather than a confirmed reposition/drop.
        return ('recovered_after_regrasp',
                (float(grasp_times_abs[0]), float(grasp_times_abs[-1])))

    if observation.push_answers and not observation.push_answers[0]:
        for i, reached in enumerate(observation.push_answers):
            if reached:
                return ('recovered_after_push_retry',
                        (float(push_times_abs[0]), float(push_times_abs[i])))

    return 'normal', None


# --- Grid checkpoint generation (rough_pad mode only) -----------------------

GRID_STEP_S = 1.0  # evenly-spaced extra timepoints across a padded window,
# only where an event checkpoint (grasp/push) isn't already close by -- these
# exist purely to help resolve_cycle_checkpoints narrow a transition down
# further than the event checkpoints alone would, they never drive the
# status decision themselves.
GRID_MIN_GAP_S = 0.3  # skip a grid point this close to an existing checkpoint


def generate_grid_checkpoints(clip_start: float, clip_end: float,
                              grasp_events_in_window, push_times_in_window,
                              step: float = GRID_STEP_S,
                              min_gap: float = GRID_MIN_GAP_S) -> list:
    existing = ([clip_start, clip_end] + list(grasp_events_in_window) +
               list(push_times_in_window))
    candidates = np.arange(clip_start, clip_end, step)
    return [
        float(c) for c in candidates
        if not any(abs(c - float(e)) < min_gap for e in existing)
    ]


# --- Single-cycle orchestration: one VLM call, anchored to a precise window -

def classify_and_anchor(model, processor, device: str, video_path: Path,
                        clip_start: float, clip_end: float,
                        grasp_events_in_window, push_times_in_window,
                        use_grid_checkpoints: bool, work_dir: Path, tag: str):
    """One VLM call over [clip_start, clip_end), returning (status, window)
    where window is the state-anchored precise (start, end) if status is a
    confirmed non-normal outcome, else None. Never asks the VLM for a
    timestamp or an overall verdict -- see module docstring.

    Push-only candidates (no grasp-side evidence) route to
    classify_push_retry's timestamp-anchored prompt instead of the grasp
    checkpoint prompt -- see build_push_retry_prompt's docstring for why.
    """
    grasp_retry_hint = len(grasp_events_in_window) > 1
    push_retry_hint = len(push_times_in_window) > 1

    clip_path = work_dir / f'_tmp_{tag}.mp4'
    clip_duration, padded_start = extract_clip(video_path, clip_start,
                                               clip_end, CLIP_PAD, clip_path)

    if not grasp_retry_hint and push_retry_hint:
        push_times_relative = [
            float(pt) - padded_start for pt in push_times_in_window]
        _, observation = classify_push_retry(
            model, processor, device, clip_path, clip_duration,
            push_times_relative)
        clip_path.unlink(missing_ok=True)
        return resolve_push_retry(observation, push_times_in_window)

    grasp_times_relative = [
        float(gt) - padded_start for gt in grasp_events_in_window]
    push_times_relative = [
        float(pt) - padded_start for pt in push_times_in_window]
    grid_times_abs = (
        generate_grid_checkpoints(clip_start, clip_end, grasp_events_in_window,
                                  push_times_in_window)
        if use_grid_checkpoints else [])
    grid_times_relative = [gt - padded_start for gt in grid_times_abs]

    _, observation = classify_cycle_checkpoints(
        model, processor, device, clip_path, clip_duration,
        grasp_times_relative, push_times_relative, clip_duration,
        grid_times_relative, is_padded_window=use_grid_checkpoints)
    clip_path.unlink(missing_ok=True)

    return resolve_cycle_checkpoints(observation, grasp_events_in_window,
                                     push_times_in_window, clip_start,
                                     clip_end)


# --- Window-strategy implementations ----------------------------------------


def run_cycle_strategy(model, processor, device, video_path, parquet_path,
                       rough_windows, work_dir):
    cycles = segment_episode(parquet_path)
    state, t = _load_state(parquet_path)
    push_times, _ = detect_right_push_peaks(state, t)

    results = []
    for i, cycle in enumerate(cycles):
        if not overlaps_any(cycle['start'], cycle['end'], rough_windows):
            continue  # no rough mark here: not a candidate at all
        in_cycle_push = push_times[(push_times >= cycle['start']) &
                                   (push_times < cycle['end'])]
        status, window = classify_and_anchor(
            model, processor, device, video_path, cycle['start'],
            cycle['end'], cycle['grasp_events'], in_cycle_push,
            False, work_dir, f'cycle{i:02d}')
        results.append({
            'cycle_index': i, 'clip_start': cycle['start'],
            'clip_end': cycle['end'], 'status': status, 'window': window,
        })
        print(f'  [cycle] cycle {i} [{cycle["start"]:.2f},{cycle["end"]:.2f}]: '
             f'{status} -> {window}')
    return results


# For --window-strategy rough_pad: deliberately >> a plausible human
# reaction delay (padding too close to the delay leaves ~0 real "before"
# footage and the model anchors every answer at frame 0 regardless of
# content).
ROUGH_PAD = 4.0


def run_rough_pad_strategy(model, processor, device, video_path,
                           parquet_path, rough_windows, work_dir):
    state, t = _load_state(parquet_path)
    grasp_events_all = detect_grasp_close_events(state, t)
    push_times_all, _ = detect_right_push_peaks(state, t)
    episode_end = float(t[-1])
    # Cycles fully partition the episode (state_segmentation.py's
    # cluster_into_cycles sets each cycle's end to the next cycle's start,
    # or episode_end for the last one -- confirmed no gaps). Clamping padding
    # to whichever cycle a rough window falls in is therefore equivalent to
    # "don't let padding bleed into the neighboring cycle" -- verified
    # necessary: a flat +/-ROUGH_PAD clamp only to episode bounds let the
    # padded clip cross into the next cycle for this dataset's short
    # (5-9s), back-to-back cycles, and "last grasp event in the padded
    # window" then grabbed a grasp from that NEXT cycle instead of the
    # current one, inflating the window 2-4x past the true boundary.
    cycles = segment_episode(parquet_path)

    def containing_cycle(r_start, r_end):
        # Adjacent cycles touch with no gap, so a rough window can overlap
        # two neighbors by a sliver right at the shared boundary -- picking
        # the FIRST overlapping cycle (as opposed to the one with the most
        # overlap) can pick the wrong neighbor by a hair. Max-overlap is
        # robust to that; ties don't matter since the sliver case has one
        # side near-zero overlap anyway.
        best_cycle, best_overlap = None, 0.0
        for cycle in cycles:
            overlap = min(cycle['end'], r_end) - max(cycle['start'], r_start)
            if overlap > best_overlap:
                best_cycle, best_overlap = cycle, overlap
        return best_cycle

    results = []
    for i, (r_start, r_end) in enumerate(rough_windows):
        cycle = containing_cycle(r_start, r_end)
        bound_start = cycle['start'] if cycle else 0.0
        bound_end = cycle['end'] if cycle else episode_end
        clip_start = max(bound_start, r_start - ROUGH_PAD)
        clip_end = min(bound_end, r_end + ROUGH_PAD)
        in_window_grasp = grasp_events_all[
            (grasp_events_all >= clip_start) & (grasp_events_all < clip_end)]
        in_window_push = push_times_all[
            (push_times_all >= clip_start) & (push_times_all < clip_end)]
        status, window = classify_and_anchor(
            model, processor, device, video_path, clip_start, clip_end,
            in_window_grasp, in_window_push, True, work_dir,
            f'rough{i:02d}')
        results.append({
            'rough_index': i, 'clip_start': clip_start, 'clip_end': clip_end,
            'status': status, 'window': window,
        })
        print(f'  [rough_pad] rough window {i} '
             f'(orig=[{r_start:.2f},{r_end:.2f}], padded=[{clip_start:.2f},'
             f'{clip_end:.2f}]): {status} -> {window}')
    return results


# --- Final fusion and persistence -------------------------------------------


def resolve_final_windows(rough_windows: list, results: list) -> list:
    """Fusion output: for each existing rough window, use the VLM+state
    confirmed refinement if an overlapping candidate cycle produced one;
    otherwise fall back to the ORIGINAL rough window unchanged. Without this,
    a candidate cycle that overlaps a real rough advantage=0 mark but comes
    back 'normal'/PARSE_FAILED silently drops that mark entirely (window=
    None never reaches write_advantage_column's `windows` list, which starts
    all-1) -- erasing an already-confirmed human-marked error instead of
    just failing to refine it. A rough window only loses its fallback once
    something has actually confirmed a replacement for it.
    """
    confirmed = [r['window'] for r in results if r['window'] is not None]
    final = list(confirmed)
    for r_start, r_end in rough_windows:
        covered = any(
            r['clip_start'] < r_end and r['clip_end'] > r_start and
            r['window'] is not None for r in results)
        if not covered:
            final.append((r_start, r_end))
    return final


def write_advantage_column(parquet_path: Path, windows: list) -> None:
    table = pq.read_table(parquet_path)
    for name in ('advantage', 'intervention'):
        if name in table.column_names:
            table = table.drop_columns([name])
    timestamps = table.column('timestamp').to_numpy()

    advantage = np.ones(len(timestamps), dtype=np.int8)
    for window in windows:
        if window is None:
            continue
        start, end = window
        advantage[(timestamps >= start) & (timestamps < end)] = 0
    intervention = np.zeros(len(timestamps), dtype=np.int8)

    table = table.append_column('advantage', pa.array(advantage, type=pa.int8()))
    table = table.append_column('intervention', pa.array(intervention, type=pa.int8()))
    pq.write_table(table, parquet_path)


def main():
    parser = argparse.ArgumentParser(
        description='Precise per-frame advantage labeling, fused against an '
        'existing rough advantage signal')
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--video-key', type=str,
                        default='observation.images.base_0_rgb')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--episodes', type=int, nargs='+', required=True)
    parser.add_argument('--window-strategy', choices=('cycle', 'rough_pad'),
                        required=True)
    parser.add_argument('--output-tag', type=str, required=True,
                        help='Used to name the saved results JSON, e.g. '
                        'cycle_true, rough_pad_false.')
    parser.add_argument('--write', action='store_true',
                        help='Actually overwrite the parquet advantage '
                        'column. Without this, only compute + report.')
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    video_dir = dataset_root / 'videos' / 'chunk-000' / args.video_key
    data_dir = dataset_root / 'data' / 'chunk-000'
    out_dir = Path('tools/recap/boundary_correction_test')
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model(args.model, args.device, torch.bfloat16)

    summary = {}
    for ep in args.episodes:
        video_path = video_dir / f'episode_{ep:06d}.mp4'
        parquet_path = data_dir / f'episode_{ep:06d}.parquet'

        rough_windows = find_rough_windows(parquet_path)
        print(f'Episode {ep}: {len(rough_windows)} existing rough window(s) '
             f'-> {rough_windows}')

        if args.window_strategy == 'cycle':
            results = run_cycle_strategy(
                model, processor, args.device, video_path, parquet_path,
                rough_windows, parquet_path.parent)
        else:
            results = run_rough_pad_strategy(
                model, processor, args.device, video_path, parquet_path,
                rough_windows, parquet_path.parent)

        windows = resolve_final_windows(rough_windows, results)
        summary[ep] = {'rough_windows': rough_windows, 'results': results}

        if args.write:
            write_advantage_column(parquet_path, windows)
            print(f'  wrote {len(windows)} refined window(s) to {parquet_path}')

    out_path = out_dir / f'refined_windows_{args.output_tag}.json'
    out_path.write_text(json.dumps(summary, indent=2))
    print(f'\nWrote summary to {out_path}')


if __name__ == '__main__':
    main()
