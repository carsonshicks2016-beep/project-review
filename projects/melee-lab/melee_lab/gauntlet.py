"""Resumable frozen-policy comparisons with both physical port assignments.

Each job runs in an isolated worker. Candidates and their opponent panel are copied
before any games begin. This module never trains or silently replaces a champion.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import NormalDist
from .catalog import Catalog, read_json
from .config import Config
from .execution import VERSION as EXECUTION_VERSION
from .storage import write_json


def interval(wins, n, z=1.96):
    if not n:
        return [0., 1.]
    p=wins/n; denominator=1+z*z/n
    center=(p+z*z/(2*n))/denominator
    radius=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/denominator
    return [max(0.,center-radius), min(1.,center+radius)]


def make_jobs(count, episodes, roster, modes):
    from .slot import COMPETITIVE_ROSTER
    jobs=[]
    cpu=[('FOX',8),('FOX',9)]
    if roster:
        cpu += [(c,9) for c in COMPETITIVE_ROSTER if c!='FOX']
    for mode in modes:
        for candidate in range(count):
            for character,level in cpu:
                jobs.append(dict(kind='cpu',p1=candidate,p2=None,character=character,
                                 level=level,mode=mode,episodes=episodes))
        # Self mirrors calibrate each candidate; both assignments compare every pair.
        for a in range(count):
            for b in range(a,count):
                for p1,p2 in ([(a,b),(b,a)] if a!=b else [(a,a)]):
                    jobs.append(dict(kind='policy',p1=p1,p2=p2,mode=mode,episodes=episodes))
    return jobs


def summarize_job(job, report):
    results=report.get('results',[])
    valid={'win','loss','draw','timeout'}
    return dict(job,results=results,wins=results.count('win'),games=len(results),
                valid=(len(results)==job['episodes'] and all(r in valid for r in results)
                       and report.get('execution_version')==EXECUTION_VERSION
                       and report.get('execution_mode')==job['mode']
                       and report.get('strict_cpu_level') is True),
                interval=interval(results.count('win'),len(results)),
                execution=report.get('execution',[]))


def promotion_gate(rows, count, modes, jobs):
    """Conservative gate against candidate zero, not a tournament skill rating.

    Requires >=40 games per self mirror and head-to-head, confidence above 50%
    against incumbent, and CPU noninferiority within 15 percentage points. Wilson
    intervals use a Bonferroni adjustment across the comparisons checked here.
    Screening trials intentionally return 'insufficient evidence'.

    The family size comes from the planned jobs, not the finished ones. Deriving it
    from completed rows made the threshold tighten as the suite progressed, so the
    same evidence passed or failed depending on when the gate happened to be read.
    """
    conditions=len({(j['character'],j['level']) for j in jobs if j['kind']=='cpu'})
    checks=max(1,len(modes)*max(1,count-1)*(3+conditions))
    z=NormalDist().inv_cdf(1-.05/(2*checks))
    output=[]
    for candidate in range(1,count):
        reasons=[]
        if len(rows)!=len(jobs):
            reasons.append(f'Comparison incomplete: {len(rows)} of {len(jobs)} jobs finished.')
        if not rows or not all(r['valid'] for r in rows):
            reasons.append('Incomplete or invalid comparison jobs.')
        for mode in modes:
            subset=[r for r in rows if r['mode']==mode]
            for who in (0,candidate):
                mirrors=[r for r in subset if r['kind']=='policy' and r['p1']==r['p2']==who]
                n=sum(r['games'] for r in mirrors); w=sum(r['wins'] for r in mirrors)
                lo,hi=interval(w,n,z)
                if n<40: reasons.append(f'{mode}: insufficient self-mirror games for candidate {who}.')
                elif not lo<=.5<=hi: reasons.append(f'{mode}: self-mirror port imbalance for candidate {who}.')
            duels=[r for r in subset if r['kind']=='policy' and {r['p1'],r['p2']}=={0,candidate}]
            n=sum(r['games'] for r in duels)
            w=sum(sum(result==('win' if r['p1']==candidate else 'loss') for result in r['results']) for r in duels)
            if len(duels)!=2 or n<40 or interval(w,n,z)[0]<=.5:
                reasons.append(f'{mode}: no confident head-to-head improvement over incumbent.')
            baseline=[r for r in subset if r['kind']=='cpu' and r['p1']==0]
            if not baseline: reasons.append(f'{mode}: CPU baseline missing.')
            for base in baseline:
                match=next((r for r in subset if r['kind']=='cpu' and r['p1']==candidate
                            and (r['character'],r['level'])==(base['character'],base['level'])),None)
                if not match or min(base['games'],match['games'])<40 or interval(match['wins'],match['games'],z)[0]-interval(base['wins'],base['games'],z)[1]<-.15:
                    reasons.append(f"{mode}: CPU {base['level']} {base['character']} non-regression unproven.")
        output.append(dict(candidate=candidate,eligible=not reasons,reasons=reasons))
    return dict(incumbent=0,automatic_promotion=False,confidence_family=.95,
                cpu_noninferiority_margin=.15,comparisons=checks,minimum_games=40,
                candidates=output)


def run_suite(directory):
    directory=Path(directory).resolve()
    request=read_json(directory/'comparison.json')
    snapshots=request['snapshots']; jobs=request['jobs']; rows=[]
    started=time.time()
    def publish(**kw):
        write_json(directory/'status.json',dict(mode='compare',started=started,elapsed=time.time()-started,
                   jobs_completed=len(rows),jobs_total=len(jobs),**kw))
    publish(status='running',phase='comparing frozen policies')
    child=None
    try:
        for index,job in enumerate(jobs):
            control=read_json(directory/'control.json')
            if control.get('stop'):
                publish(status='stopped',phase='comparison stopped; completed jobs retained'); return
            case=directory/f'job-{index:04d}'
            existing=read_json(case/'evaluation.json')
            if existing:
                result=summarize_job(job,existing)
                if result['valid']:
                    rows.append(result); continue
            case.mkdir(exist_ok=True)
            # A restarted job must not inherit a previous stop request or results.
            write_json(case/'control.json',{})
            write_json(case/'status.json',{})
            cfg=Config(**request['config'])
            cfg.execution_mode=job['mode']; cfg.strict_evaluation=True; cfg.league_manifest=None
            cfg.randomize_opponent=False; cfg.seed=request['seed']
            cfg.character=snapshots[job['p1']]['config']['character']
            cfg.cpu_level=job.get('level',1)
            cfg.opponent=job.get('character') or snapshots[job['p2']]['config']['character']
            cfg.save(case/'config.json')
            command=[sys.executable,'-u','-m','melee_lab.worker','--mode','evaluate',
                     '--config',str(case/'config.json'),'--run-dir',str(case),
                     '--checkpoint',str(directory/snapshots[job['p1']]['path']),
                     '--episodes',str(job['episodes']),'--bootstrap','0']
            if job['p2'] is not None:
                command += ['--opponent-checkpoint',str(directory/snapshots[job['p2']]['path'])]
            publish(status='running',phase=f"comparison {index+1}/{len(jobs)} · {job['mode']} · {job['kind']}")
            with (case/'worker.log').open('a') as log:
                child=subprocess.Popen(command,cwd=request['root'],stdout=log,stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
                mirrored=None
                while child.poll() is None:
                    control=read_json(directory/'control.json')
                    forwarded={k:control[k] for k in ('stop','pause') if k in control}
                    if forwarded!=mirrored:
                        write_json(case/'control.json',forwarded); mirrored=forwarded
                    live=read_json(case/'status.json')
                    if live: write_json(directory/'env0/status.json',dict(live,index=0))
                    time.sleep(.5)
            if read_json(directory/'control.json').get('stop'):
                publish(status='stopped',phase='comparison stopped; completed jobs retained'); return
            report=read_json(case/'evaluation.json')
            result=summarize_job(job,report)
            # The worker writes its atomic report and completed status before Python's
            # multiprocessing cleanup runs. A leaked Slippi semaphore can make that
            # cleanup return nonzero even though the requested games and evidence are
            # complete. Trust the validated artifact in that case; still reject a
            # nonzero worker that did not reach its completed state.
            case_status=read_json(case/'status.json')
            if not result['valid'] or (child.returncode and case_status.get('status')!='completed'):
                raise RuntimeError(f"Job {index+1} failed or produced invalid evidence: {read_json(case/'status.json').get('error','see job log')}")
            rows.append(result)
            write_json(directory/'results.json',dict(snapshots=snapshots,rows=rows,
                       gate=promotion_gate(rows,len(snapshots),request['modes'],jobs)))
        write_json(directory/'results.json',dict(snapshots=snapshots,rows=rows,
                   gate=promotion_gate(rows,len(snapshots),request['modes'],jobs)))
        publish(status='completed',phase='comparison complete; promotion decision available')
    except Exception as exc:
        publish(status='failed',phase='comparison needs attention',error=str(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            write_json(case/'control.json',{'stop':True})
            try: child.wait(timeout=45)
            except subprocess.TimeoutExpired: child.terminate()


def prepare(catalog, directory, candidates, config, episodes=10, roster=True, modes=('assisted','raw'), seed=17001):
    from .storage import validate_checkpoint
    from dataclasses import replace, asdict
    if not 2<=len(candidates)<=4: raise ValueError('Select two to four candidates, incumbent first.')
    if not 1<=episodes<=1000: raise ValueError('Games per condition must be between 1 and 1000.')
    if not modes or any(m not in ('raw','assisted') for m in modes): raise ValueError('Invalid execution modes.')
    directory=Path(directory); snapshots=[]
    first=read_json(catalog.resolve(candidates[0]).with_suffix('.json'))['config']
    config=replace(config,character=first['character'],action_frames=first['action_frames'],
                   action_set=first.get('action_set','legacy'),randomize_opponent=True,league_manifest=None)
    if config.action_set!='controller': raise ValueError('Fair comparison currently requires full-controller policies.')
    for i,source in enumerate(candidates):
        validate_checkpoint(catalog.resolve(source),config)
        identity=catalog.snapshot(source,directory/'candidates'/f'candidate-{i}.zip')
        snapshots.append(dict(identity,path=f'candidates/candidate-{i}.zip'))
    if len({s['sha256'] for s in snapshots})!=len(snapshots):
        raise ValueError('Choose distinct checkpoint weights; self mirrors are added automatically.')
    request=dict(version=1,root=str(catalog.root.resolve()),config=asdict(config),snapshots=snapshots,
                 jobs=make_jobs(len(candidates),episodes,roster,modes),modes=list(modes),seed=seed)
    write_json(directory/'comparison.json',request)
    return request


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    run_suite(args.run_dir)
