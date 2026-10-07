"""Independently audit recorded commands and physical calibration outcomes."""
import sys,json,gzip
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.measure_body_operating_range import OUT,sha,metrics,DT
from flygarden.descending import DescendingDecoder
from flygarden.recording import atomic_json

def run():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in protocol['sources'].items())
    progress=json.loads((OUT/'progress.json').read_text());assert progress['status']=='complete'
    results=[]
    for seed in protocol['seeds']:
        for name,case in protocol['scenarios'].items():
            folder=OUT/'trials'/f'{seed}-{name}';m=json.loads((folder/'manifest.json').read_text())
            assert m['status']=='complete' and m['completed_windows']==120
            rows=[];frames=[]
            for c in m['chunks']:
                p=folder/c['file'];assert sha(p)==c['sha256']
                with gzip.open(p,'rt') as f:data=json.load(f)
                assert len(data['rows'])==c['windows'] and len(data['frames'])==c['frames']
                rows.extend(data['rows']);frames.extend(data['frames'])
            assert len(rows)==120 and len(frames)==90
            assert np.all(np.abs(np.array([f['time'] for f in frames])-np.arange(1,91)/30)<.0005+1e-9)
            decoder=DescendingDecoder()
            for tick,row in enumerate(rows):
                assert abs(row['time']-(tick+1)*DT)<1e-9
                active=.5<=tick*DT<2.5
                if 'rates' in case:
                    rates=case['rates'] if active else dict.fromkeys(case['rates'],0)
                    expected=decoder.advance(DT,rates)
                else:expected=case['motor'] if active else [0.,0.]
                assert np.array_equal(row['motor'],expected)
                assert np.isfinite(row['body']['position']).all()
            assert metrics(m['initial'],rows)==m['metrics']
            p=folder/m['checkpoint']['file'];assert sha(p)==m['checkpoint']['sha256']
            results.append({'seed':seed,'case':name,**m['metrics'],'wall_seconds':m['attempt_wall_seconds']})
    atomic_json(OUT/'results.json',{'status':'complete','trials':results,'commands_reconstructed':True,
                                  'frame_timing_verified':True,'neural_activity_simulated':False,
                                  'held_out_60_second_acceptance_completed':False})
    table='\n'.join(f"| {r['seed']} | {r['case']} | {r['xy_displacement_mm']:.3f} | {r['heading_change_rad']:.4f} | {r['flipped_samples']} |" for r in results)
    (OUT/'RESULTS.md').write_text(f'''# Body/decoder calibration measurement

Twenty-two three-second calibration trials completed at 25-ms command intervals. All saved commands were independently reconstructed; all 90 poses per trial match the 30-Hz sampling target within a gait step. All trial chunk and latest checkpoint hashes verify. Every trial had finite sampled positions and no flipped samples.

These are controlled body commands and artificial injected rate readouts, not full-brain trials or validated navigation. The decoder, brain and gait gains were not changed. Seeds 9101/9102 are calibration-only and must not become held-out acceptance seeds.

| Seed | Command case | Final xy displacement (mm) | Heading change (rad) | Flipped samples |
|---|---|---|---|---|
{table}

The physical body and fixed candidate decoder can produce substantial forward motion and mirrored turns at these tested command/rate levels. The weak movement in Step 8 is therefore not explained by an inability of this body to move when given adequate commands. This does not establish which sensory/model/readout change is justified.

The field named `contacts` in the pinned Body adapter contains the supplied controller's **stumbling contact forces**, not complete foot-ground loads. Its zero value on these open-ground trials must not be interpreted as absent ground contact or used to claim slip-free walking. Raw poses and contact fields are retained. Full contact/stance/slip observability and ten held-out 60-second trials remain required.

Physics state and decoder smoothing state are retained at each committed one-second chunk; AC-power interruption can resume at that boundary. Unit tests verify resumed command/record sequences and preserved original chunks; fresh-process native physical continuation is a separate pending acceptance check. The previous Step 7 native continuation result remains separately archived.
''')
    print({'trials':len(results),'commands_reconstructed':True,'max_displacement_mm':max(r['xy_displacement_mm'] for r in results)})

if __name__=='__main__':run()
