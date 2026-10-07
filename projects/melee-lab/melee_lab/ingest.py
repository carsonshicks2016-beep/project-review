"""Stream tournament replays from a public archive into a packed demonstration dataset.

The archive is larger than this machine's free disk, and the raw .slp files are not
worth keeping anyway: a replay is ~2.3 MB and the frames it contributes are ~0.7 MB
packed. So replays are downloaded a chunk at a time, parsed, written to a shard, and
deleted. Peak disk is one chunk of .slp plus the shards, never the whole archive.

Shards are resumable. A chunk whose shard already exists is skipped, so an interrupted
ingest continues rather than starting over.
"""
import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .config import ROOT

REPO = 'erickfm/slippi-public-dataset-v3.7'
API = f'https://huggingface.co/api/datasets/{REPO}/tree/main/'
FILES = f'https://huggingface.co/datasets/{REPO}/resolve/main/'
AGENT = {'User-Agent': 'melee-lab/0.1 (local research)'}

# The archive labels stages two ways: a short tag in parentheses, or a long name after
# the last ' - '. Both appear inside the same character folder.
STAGE_TAGS = {
    'FINAL_DESTINATION': ('(FD)', '[FD]', 'Final Destination'),
    'BATTLEFIELD': ('(BF)', '[BF]', 'Battlefield'),
    'YOSHIS_STORY': ('(YS)', '[YS]', 'Yoshi_s Story', 'Yoshi', '(YI)', '[YI]'),
    'DREAMLAND': ('(DL)', '[DL]', 'Dream Land N64', 'Dream Land'),
    'POKEMON_STADIUM': ('(PS)', '[PS]', 'Pokemon Stadium'),
    'FOUNTAIN_OF_DREAMS': ('(FoD)', '[FoD]', 'Fountain of Dreams'),
    'ALL': ()
}


def _get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=90) as r:
        return json.load(r), r.headers.get('Link', '')


def _next_cursor(link):
    if 'rel="next"' not in link: return None
    target = link[link.index('<') + 1:link.index('>')]
    return urllib.parse.parse_qs(urllib.parse.urlparse(target).query).get('cursor', [None])[0]


def listing(character='FOX'):
    """Every file under the character's folder, as (path, bytes)."""
    files, pending = [], [character]
    while pending:
        folder = pending.pop(0)
        cursor = None
        while True:
            entries, link = _get(API + urllib.parse.quote(folder) + '?limit=1000'
                                 + (f'&cursor={urllib.parse.quote(cursor)}' if cursor else ''))
            for entry in entries:
                if entry['type'] == 'directory': pending.append(entry['path'])
                else: files.append((entry['path'], entry['size']))
            cursor = _next_cursor(link)
            if not cursor: break
            time.sleep(.05)
    return files


def on_stage(files, stage='FINAL_DESTINATION'):
    if not stage or stage == 'ALL':
        return files
    tags = STAGE_TAGS.get(stage)
    if not tags: return []
    return [(p, s) for p, s in files if any(t in p for t in tags)]


def fetch(path, into, attempts=4):
    """Download one replay. Returns (local path, sha256) or None.

    Retried because the CDN truncates a small fraction of responses under concurrency,
    and a silently dropped replay is a silently smaller dataset. A short read raises
    IncompleteRead here rather than producing a corrupt .slp, so retrying is safe."""
    target = Path(into) / re.sub(r'[^A-Za-z0-9._-]', '_', path.rsplit('/', 1)[-1])
    request = urllib.request.Request(FILES + urllib.parse.quote(path), headers=AGENT)
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
        except Exception as exc:
            if attempt == attempts - 1:
                sys.stderr.write(f'download failed {path}: {exc}\n')
                return None
            time.sleep(1.5 * (attempt + 1))
            continue
        target.write_bytes(payload)
        return target, hashlib.sha256(payload).hexdigest()
    return None


def parse_one(argument):
    """Worker: one replay in, packed arrays out. Runs in its own process."""
    path, character, stage, stride = argument
    from .imitation import Demonstrations, parse_replay
    try:
        observations, actions, returns = parse_replay(path, character=character, stage=stage, stride=stride)
    except Exception as exc:
        sys.stderr.write(f'parse failed {Path(path).name}: {exc}\n')
        return None
    if not len(observations): return None
    packed = Demonstrations.from_dense(observations, actions, returns,
                                       np.zeros(len(observations), np.int32))
    return packed.continuous, packed.globals, packed.ids, packed.actions, packed.returns


def ingest(output, character='FOX', stage='FINAL_DESTINATION', stride=6, limit=None,
           chunk=64, workers=None, work_dir=None, target_samples=None):
    from .imitation import Demonstrations, concatenate

    work = Path(work_dir or ROOT / '.runtime/ingest')
    (work / 'slp').mkdir(parents=True, exist_ok=True)
    (work / 'shards').mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, min(10, (os.cpu_count() or 4) - 2))

    print(f'Listing {character} replays in {REPO} ...')
    every = listing(character)
    wanted = on_stage(every, stage)
    wanted.sort()
    if limit: wanted = wanted[:limit]
    size = sum(s for _, s in wanted)
    print(f'{len(every)} {character} replays; {len(wanted)} on {stage} '
          f'({size/1e9:.1f} GB remote). Stride {stride}, {workers} parse workers.')

    seen, samples, replays, started = {}, 0, 0, time.time()
    chunks = [wanted[i:i+chunk] for i in range(0, len(wanted), chunk)]
    for number, group in enumerate(chunks):
        shard_path = work / 'shards' / f'{number:05d}.npz'
        if shard_path.exists():
            print(f'[chunk {number+1}/{len(chunks)}] shard present, skipping')
            continue

        with futures.ThreadPoolExecutor(max_workers=8) as pool:
            fetched = [f for f in pool.map(lambda e: fetch(e[0], work / 'slp'), group) if f]
        # Deduplicate on content, not name: the archive stores some sets twice.
        unique = []
        for local, digest in fetched:
            if digest in seen: local.unlink(missing_ok=True)
            else: seen[digest] = local.name; unique.append(local)

        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            parsed = [p for p in pool.map(parse_one, [(str(p), character, stage, stride) for p in unique]) if p]

        for local in unique: local.unlink(missing_ok=True)

        if parsed:
            # One replay per part, numbered 0; concatenate() assigns the running number.
            joined = concatenate([Demonstrations(continuous, globals_, ids, actions, returns,
                                                 np.zeros(len(ids), np.int32))
                                  for continuous, globals_, ids, actions, returns in parsed])
            joined.save(shard_path)
            samples += len(joined); replays += len(parsed)
        rate = (time.time() - started) / max(1, number + 1)
        print(f'[chunk {number+1}/{len(chunks)}] {len(parsed)} replays parsed · '
              f'{samples:,} samples · {replays} replays total · '
              f'{rate:.0f}s/chunk · eta {rate*(len(chunks)-number-1)/60:.0f} min')
        if target_samples and samples >= target_samples:
            print(f'Reached the sample target ({target_samples:,}).')
            break

    return merge(work / 'shards', output)


def shard_length(path):
    """Sample count from the .npy header alone, without decompressing the shard."""
    import zipfile
    with zipfile.ZipFile(path) as archive, archive.open('ids.npy') as handle:
        version = np.lib.format.read_magic(handle)
        shape, _, _ = (np.lib.format.read_array_header_1_0(handle) if version == (1, 0)
                       else np.lib.format.read_array_header_2_0(handle))
    return int(shape[0])


def merge(shards_dir, output):
    """Join every shard into the final dataset, one at a time to bound memory.

    Reading every shard first and concatenating after would hold two full copies of a
    multi-gigabyte dataset at once, so the total is sized from the shard headers and
    each shard is copied in and released."""
    from .imitation import Demonstrations
    paths = sorted(Path(shards_dir).glob('*.npz'))
    if not paths: raise ValueError('No shards to merge.')
    print(f'\nMerging {len(paths)} shards ...')
    total = sum(shard_length(p) for p in paths)
    first = Demonstrations.load(paths[0])
    dataset = Demonstrations(
        np.empty((total, first.continuous.shape[1]), np.float32),
        np.empty((total, first.globals.shape[1]), np.float32),
        np.empty((total, first.ids.shape[1]), np.int16),
        np.empty((total, first.actions.shape[1]), np.int64),
        np.empty(total, np.float32), np.empty(total, np.int32))
    del first
    at = group = 0
    for index, path in enumerate(paths):
        part = Demonstrations.load(path)
        n = len(part)
        dataset.continuous[at:at+n] = part.continuous
        dataset.globals[at:at+n] = part.globals
        dataset.ids[at:at+n] = part.ids
        dataset.actions[at:at+n] = part.actions
        dataset.returns[at:at+n] = part.returns
        dataset.groups[at:at+n] = part.groups + group
        group += int(part.groups.max()) + 1
        at += n
        del part
        if (index + 1) % 10 == 0 or index + 1 == len(paths):
            print(f'  {index+1}/{len(paths)} shards · {at:,} samples · {group:,} replays')
    out = dataset.save(output)
    replays = int(dataset.groups.max()) + 1
    print(f'{len(dataset):,} samples from {replays:,} replays · '
          f'{dataset.nbytes/1e6:.0f} MB in memory · {out.stat().st_size/1e6:.0f} MB on disk')
    print(f'Returns: mean {dataset.returns.mean():.2f}, std {dataset.returns.std():.2f}')
    return dataset


def main():
    p = argparse.ArgumentParser(description='Stream tournament replays into a packed dataset')
    p.add_argument('--output', required=True)
    p.add_argument('--character', default='FOX')
    p.add_argument('--stage', default='FINAL_DESTINATION', choices=sorted(STAGE_TAGS))
    p.add_argument('--stride', type=int, default=6, help='keep one decision every N frames')
    p.add_argument('--limit', type=int, help='cap the number of replays considered')
    p.add_argument('--target-samples', type=int, help='stop once this many samples are collected')
    p.add_argument('--chunk', type=int, default=64, help='replays downloaded and parsed per shard')
    p.add_argument('--workers', type=int)
    p.add_argument('--work-dir')
    p.add_argument('--merge-only', action='store_true')
    args = p.parse_args()
    work = Path(args.work_dir or ROOT / '.runtime/ingest')
    if args.merge_only:
        merge(work / 'shards', args.output); return
    ingest(args.output, character=args.character, stage=args.stage, stride=args.stride,
           limit=args.limit, chunk=args.chunk, workers=args.workers, work_dir=args.work_dir,
           target_samples=args.target_samples)


if __name__ == '__main__':
    main()
