#!/usr/bin/env python3
"""
LectureCanvas Audio Transcription Server

Local, GPU-accelerated transcription using OpenAI Whisper, streamed over
Server-Sent Events so a two-hour lecture reports progress while it works instead
of going silent until the end.
"""

import os
import sys
import json
import time
import tempfile
import threading
from typing import Any, Dict, Generator, Iterable, List

from flask import Flask, Response, jsonify, request

import torch


def _select_device() -> str:
    override = os.environ.get('LECTURECANVAS_DEVICE')
    if override:
        return override
    if torch.backends.mps.is_available():
        return 'mps'
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


device = _select_device()

# Half precision is reliable on CUDA; on MPS and CPU it is either unsupported or
# a source of NaN decodes, so those stay in fp32 and still run on the GPU.
USE_FP16 = device == 'cuda'

# Audio is transcribed in windows so results can be streamed. Each window is
# re-anchored to the end of its last complete segment, so words are not cut in half.
WINDOW_SECONDS = float(os.environ.get('LECTURECANVAS_WINDOW_SECONDS', 120))

ALLOWED_MODELS = (
    'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium',
)

# Priming the decoder with subject-matter vocabulary measurably improves technical terms.
ACADEMIC_PROMPT = (
    'This is a classroom lecture with technical and academic vocabulary, '
    'concepts, definitions, explanations, formulas, and questions.'
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1 GB, for 2+ hour recordings

# Whisper weights are hundreds of MB; loading them per request would dominate runtime.
MODELS_CACHE: Dict[str, Any] = {}
cache_lock = threading.Lock()


def get_whisper_model(model_name: str):
    with cache_lock:
        if model_name not in MODELS_CACHE:
            import whisper
            print(f"[Whisper] Loading model '{model_name}' on device '{device}'...")
            MODELS_CACHE[model_name] = whisper.load_model(model_name, device=device)
            print(f"[Whisper] Model '{model_name}' loaded.")
        return MODELS_CACHE[model_name]


@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Range'
    response.headers['Access-Control-Expose-Headers'] = 'Content-Length, Content-Range'
    return response


@app.route('/api/health', methods=['GET', 'OPTIONS'])
def health():
    if request.method == 'OPTIONS':
        return ('', 204)
    return jsonify({
        'status': 'ok',
        'device': device,
        'gpu_accelerated': device in ('mps', 'cuda'),
        'fp16': USE_FP16,
        'loaded_models': sorted(MODELS_CACHE.keys()),
        'available_models': list(ALLOWED_MODELS),
    })


def sse(payload: Dict[str, Any]) -> str:
    """Formats one Server-Sent Event."""
    return f"data: {json.dumps(payload)}\n\n"


def format_clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def transcribe_windows(
    model,
    audio,
    duration_seconds: float,
    sample_rate: int,
) -> Iterable[Dict[str, Any]]:
    """
    Walks the recording window by window, yielding progress and segment events.

    Whisper's own `transcribe` is a single blocking call over the whole file, which
    is why the previous progress hook could never report anything: there is nothing
    to observe until it returns. Slicing the audio first gives real incremental
    output, at the cost of re-priming the decoder once per window.
    """
    cursor = 0.0
    segment_index = 0
    # Carries the tail of the previous window so the decoder keeps its context
    # across the seam.
    context = ACADEMIC_PROMPT

    while cursor < duration_seconds:
        window_end = min(cursor + WINDOW_SECONDS, duration_seconds)
        chunk = audio[int(cursor * sample_rate):int(window_end * sample_rate)]
        if chunk.size == 0:
            break

        result = model.transcribe(
            chunk,
            initial_prompt=context,
            verbose=False,
            fp16=USE_FP16,
            # The default temperature ladder lets Whisper retry a window that
            # decoded badly; pinning it to 0.0 removes that safety net and is a
            # common cause of repetition loops on long lectures.
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        )

        segments: List[Dict[str, Any]] = result.get('segments') or []
        last_end = 0.0

        for seg in segments:
            text = (seg.get('text') or '').strip()
            seg_start = cursor + float(seg.get('start', 0.0))
            seg_end = cursor + float(seg.get('end', 0.0))
            last_end = max(last_end, float(seg.get('end', 0.0)))
            if not text:
                continue

            yield {
                'type': 'segment',
                'id': f'seg-{segment_index}',
                'start': round(seg_start, 2),
                'end': round(seg_end, 2),
                'text': text,
                'percent': round(min(99.0, (seg_end / duration_seconds) * 100), 1),
                'elapsedLabel': f'{format_clock(seg_end)} of {format_clock(duration_seconds)}',
            }
            segment_index += 1

        window_length = window_end - cursor
        # Resume at the end of the last complete segment so no word is split. If the
        # window decoded to very little, the remainder was silence — skip it whole.
        if last_end >= window_length * 0.5:
            advance = last_end
        else:
            advance = window_length
        cursor += max(advance, 1.0)

        tail = (result.get('text') or '').strip()
        if tail:
            context = tail[-200:]

        yield {
            'type': 'progress',
            'percent': round(min(99.0, (cursor / duration_seconds) * 100), 1),
            'currentSecond': round(min(cursor, duration_seconds), 1),
            'totalSeconds': round(duration_seconds, 1),
            'message': (
                f'Transcribing on {device.upper()} — '
                f'{format_clock(cursor)} of {format_clock(duration_seconds)}'
            ),
        }


@app.route('/api/transcribe', methods=['POST', 'OPTIONS'])
def transcribe_stream():
    if request.method == 'OPTIONS':
        return ('', 204)

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    uploaded_file = request.files['file']
    if not uploaded_file.filename:
        return jsonify({'error': 'Empty filename'}), 400

    model_name = request.form.get('model', 'base.en')
    if model_name not in ALLOWED_MODELS:
        model_name = 'base.en'

    # The client controls this filename; keep only a plain extension so it can
    # never steer where the temporary file lands.
    raw_suffix = os.path.splitext(os.path.basename(uploaded_file.filename))[1]
    suffix = raw_suffix if raw_suffix.isascii() and raw_suffix[1:].isalnum() else '.webm'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_path = temp_audio.name
        uploaded_file.save(temp_path)

    def generate_events() -> Generator[str, None, None]:
        started = time.time()
        try:
            import whisper

            file_size_mb = os.path.getsize(temp_path) / (1024 * 1024)
            yield sse({
                'type': 'init',
                'percent': 1,
                'message': f'File received ({file_size_mb:.1f} MB). Loading model {model_name}...',
            })

            model = get_whisper_model(model_name)

            # Decode the audio once and reuse the array for every window; decoding a
            # two-hour file twice (once for duration, once to transcribe) is minutes
            # of wasted work.
            audio = whisper.load_audio(temp_path)
            sample_rate = whisper.audio.SAMPLE_RATE
            duration_seconds = len(audio) / sample_rate

            if duration_seconds <= 0:
                yield sse({'type': 'error', 'message': 'That file contains no decodable audio.'})
                return

            yield sse({
                'type': 'metadata',
                'percent': 2,
                'duration': duration_seconds,
                'message': (
                    f'Audio duration {format_clock(duration_seconds)}. '
                    f'Transcribing on {device.upper()}...'
                ),
            })

            total_segments = 0
            for event in transcribe_windows(model, audio, duration_seconds, sample_rate):
                if event['type'] == 'segment':
                    total_segments += 1
                yield sse(event)

            yield sse({
                'type': 'complete',
                'totalSegments': total_segments,
                'duration': duration_seconds,
                'elapsedSeconds': round(time.time() - started, 1),
            })

        except Exception as e:  # noqa: BLE001 - surfaced to the client verbatim
            print(f'[Error] Transcription failed: {e}', file=sys.stderr)
            yield sse({'type': 'error', 'message': str(e)})
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    return Response(
        generate_events(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            # Stops nginx and friends from buffering the stream into one response.
            'X-Accel-Buffering': 'no',
        },
    )


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5175))
    print(f'Starting LectureCanvas Whisper Server on http://127.0.0.1:{port} (device: {device})')
    app.run(host='127.0.0.1', port=port, threaded=True)
