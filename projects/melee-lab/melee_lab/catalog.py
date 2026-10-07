"""Immutable policy snapshots and evaluation evidence, identified by checkpoint bytes."""
import hashlib
import json
import os
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from .execution import VERSION as EXECUTION_VERSION
from .storage import write_json


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def signature(path):
    s=Path(path).stat()
    return (s.st_ino, s.st_size, s.st_mtime_ns)


class Catalog:
    def __init__(self, root):
        self.root=Path(root)
        self.hashes={}
        self.evidence_seen={}

    def digest(self, path):
        path=Path(path)
        stamp=signature(path)
        cached=self.hashes.get(str(path))
        if cached and cached[0]==stamp: return cached[1]
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if stamp!=signature(path): raise ValueError('Checkpoint is being saved. Try again in a moment.')
        self.hashes[str(path)]=(stamp,digest)
        return digest

    def resolve(self, name):
        path=(self.root/name).resolve()
        allowed=any(path.is_relative_to((self.root/folder).resolve()) for folder in ('runs','champions'))
        if not allowed or path.suffix!='.zip' or not path.is_file() or '.tmp.' in path.name:
            raise ValueError('Choose an existing checkpoint or champion in this project.')
        if not path.with_suffix('.json').is_file():
            raise ValueError('Checkpoint metadata is missing.')
        return path

    def snapshot(self, source, target):
        source=self.resolve(str(source)); target=Path(target)
        meta_path=source.with_suffix('.json')
        before=(signature(source),signature(meta_path))
        metadata=read_json(meta_path)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        # A writer replaces the zip and metadata separately. Compare the embedded
        # SB3 counter as well as file signatures to catch that intermediate window.
        try:
            with zipfile.ZipFile(target) as archive:
                embedded=json.loads(archive.read('data'))
            if embedded.get('num_timesteps')!=metadata.get('steps'):
                raise ValueError('Checkpoint and metadata are between saves. Try again in a moment.')
            if before!=(signature(source),signature(meta_path)):
                raise ValueError('Checkpoint changed while it was being copied. Try again in a moment.')
            write_json(target.with_suffix('.json'),metadata)
            return dict(sha256=self.digest(target), source=str(source.relative_to(self.root)),
                        steps=metadata['steps'], config=metadata['config'])
        except Exception:
            target.unlink(missing_ok=True)
            raise

    def ingest_evaluations(self):
        directory=self.root/'evaluations'
        for path in (self.root/'runs').glob('*/evaluation.json'):
            stamp=signature(path)
            if self.evidence_seen.get(str(path))==stamp: continue
            report=read_json(path)
            request=read_json(path.parent/'request.json')
            identity=request.get('checkpoint_identity')
            # Older unlinked reports cannot reliably identify the evaluated policy.
            # They remain in their run, but must never attach to an arbitrary latest.zip.
            if not identity or request.get('mode')!='evaluate': continue
            if report.get('policy') not in ('deterministic neural network','sampled neural network'):
                continue
            results=report.get('results',[])
            if not results or len(results)!=report.get('episodes') or results.count('win')!=report.get('wins'):
                continue
            if report.get('three_stock_start_verified') is not True: continue
            # Execution labelling is recomputed from the per-match counters the shared
            # controller rules record. A report does not get to assert its own honesty:
            # before those counters existed there was no way to tell an assisted trial
            # from a clean one, and the worker hardcoded 'no overrides' either way.
            measured=[m for row in report.get('execution',[])
                      for m in (row.get('execution'),row.get('opponent_execution')) if m]
            decisions=sum(m.get('decisions',0) for m in measured)
            changed=sum(m.get('changed_decisions',0) for m in measured)
            verified=(report.get('execution_version')==EXECUTION_VERSION
                      and report.get('execution_mode') in ('assisted','raw')
                      and all(m.get('version')==EXECUTION_VERSION for m in measured))
            # Raw means the network's own controller vector reached the pad. Counters
            # that disagree make the trial evidence for neither mode.
            if verified and report.get('execution_mode')=='raw' and changed: continue
            # An unverifiable trial that declares overrides is kept out, exactly as
            # before. Only the claim of running clean now requires the counters.
            if not verified and report.get('scripted_overrides'): continue
            artifact=path.parent/'input-policy.zip'
            if not artifact.exists() or self.digest(artifact)!=identity['sha256']: continue
            # Recompute headline figures from explicit outcomes rather than trust stale totals.
            from .analytics import summarize
            stats=summarize([{'result':r} for r in results])
            evidence=dict(report, **{k:stats[k] for k in ('episodes','wins','win_rate','win_rate_95_interval','outcomes')},
                          id=path.parent.name, checkpoint_sha256=identity['sha256'],
                          source_checkpoint=identity['source'], checkpoint_steps=identity['steps'],
                          evaluated_at=path.stat().st_mtime, run_id=path.parent.name)
            evidence['execution_verified']=verified
            evidence['execution_decisions']=decisions
            evidence['measured_override_rate']=changed/decisions if decisions else None
            # A discrete-vocabulary policy never consults the rules, so it is unassisted
            # by construction and its decision count is zero.
            evidence['scripted_overrides']=(report.get('execution_mode')=='assisted' and decisions>0) if verified else None
            if not verified: evidence['execution_mode']='legacy-unverified'
            write_json(directory/(path.parent.name+'.json'),evidence)
            self.evidence_seen[str(path)]=stamp

    def evidence(self):
        self.ingest_evaluations()
        return sorted([read_json(p) for p in (self.root/'evaluations').glob('*.json')],
                      key=lambda e:e.get('evaluated_at',0), reverse=True)

    def checkpoints(self):
        evidence=self.evidence()
        result=[]
        paths=[p for p in list((self.root/'runs').glob('*/*.zip'))
                          +list((self.root/'champions').glob('*/policy.zip'))
               if '.tmp.' not in p.name and not p.parent.name.startswith('.')]
        def when(path):
            # A checkpoint is written to a .tmp. file and renamed into place, so any
            # path found by the glob can be gone by the time it is stat-ed. Filtering
            # after the sort key ran was enough to 500 the dashboard on every save.
            try: return path.stat().st_mtime
            except OSError: return -1.0
        for path in sorted(paths,key=when,reverse=True):
            updated=when(path)
            if updated<0: continue
            meta=read_json(path.with_suffix('.json'))
            if not meta or 'config' not in meta: continue
            try: digest=self.digest(path)
            except (OSError,ValueError): continue
            champion=read_json(path.parent/'champion.json')
            evaluations=[e for e in evidence if e.get('checkpoint_sha256')==digest]
            status=read_json(path.parent/'status.json')
            result.append(dict(path=str(path.relative_to(self.root)),
                name=champion.get('name') or status.get('display_name') or path.parent.name+'/'+('evaluated policy' if path.stem=='input-policy' else path.stem),
                run_id=path.parent.name if not champion else champion.get('source_run'),
                imported=status.get('imported', False),
                steps=meta.get('steps',0), bootstrap_samples=meta.get('bootstrap_samples',0),
                imitation_samples=meta.get('imitation_samples',0),
                imitation_val_accuracy=meta.get('imitation_val_accuracy'),
                training_cpu_level=meta.get('training_cpu_level',meta['config'].get('cpu_level')),
                architecture=meta.get('architecture'),
                actions=meta.get('actions'),
                config=meta['config'], sha256=digest, evaluations=evaluations,
                champion=champion or None, updated=updated))
        return result

    def promote(self, checkpoint, name, note):
        name=name.strip(); note=note.strip()
        if not name or len(name)>80 or len(note)>2000:
            raise ValueError('Use a name of 1–80 characters and a note of at most 2,000 characters.')
        source=self.resolve(checkpoint)
        self.ingest_evaluations()
        directory=self.root/'champions'; directory.mkdir(exist_ok=True)
        identifier=uuid.uuid4().hex[:12]
        temporary=directory/('.'+identifier); temporary.mkdir()
        try:
            identity=self.snapshot(source,temporary/'policy.zip')
            entry=dict(id=identifier,name=name,note=note,created=time.time(),
                       source_checkpoint=identity['source'],source_run=source.parent.name,
                       sha256=identity['sha256'],steps=identity['steps'])
            write_json(temporary/'champion.json',entry)
            os.replace(temporary,directory/identifier)
        except Exception:
            shutil.rmtree(temporary,ignore_errors=True)
            raise
        return entry
