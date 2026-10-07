"""Reconcile the released full synapse archive to the unchanged modeled graph."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import resource
import shutil
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
from flygarden.synapse_archive import select_root_rows, pair_counts, reconcile_counts

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v1'
OUT = ROOT/'reports/brain-integration/recovery/patchy-compartment-mapping-v2'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def main():
    start = time.monotonic()
    archive = OUT/'source/flywire_synapses_783.feather'
    identity = json.loads((OUT/'archive-identity.json').read_text())
    if (identity.get('md5') != 'f8f1b97c9d4b0ea9b4c8b287f6b99091'
            or not identity.get('published_checksum_verified')
            or archive.stat().st_size != identity['bytes']
            or sha(archive) != identity['sha256']):
        raise ValueError('Archive identity is not verified')
    neurons = json.loads((OLD/'neurons.json').read_text())
    roots = np.array([int(n['root_id']) for n in neurons], dtype=np.int64)
    edges = pd.read_csv(OLD/'imported-edges.csv', dtype={'Presynaptic_ID':str,'Postsynaptic_ID':str})
    imported = {(int(r.Presynaptic_ID),int(r.Postsynaptic_ID)):int(r.Connectivity) for r in edges.itertuples()}
    inputs = [OLD/'neurons.json', OLD/'imported-edges.csv', ROOT/'flygarden/synapse_archive.py', Path(__file__).resolve()]
    protocol = {'version':1,'materialization':783, 'archive_identity':identity,
                'source_hashes':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
                'selected_roots':list(map(str, roots)), 'coordinate_units':'nanometers, as specified in Zenodo record',
                'filter':'No additional filter; archive already thresholded by source authors. Preserve unmodeled partners separately.',
                'expected_imported_pairs':len(imported), 'expected_imported_sites':sum(imported.values()),
                'gates':{'every_imported_pair_exact':True, 'selected_synapse_ids_unique':True},
                'anatomical_assignment':'Unavailable coordinates or failed atlas registration remain unavailable',
                'application_changed':False}
    (OUT/'synapse-protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    observed = Counter(); selected = 0; offset = 0; ids = set(); min_score = None
    temporary = OUT/'selected-synapses.parquet.partial'
    writer = None
    with pa.OSFile(str(archive),'rb') as source:
        reader = ipc.open_file(source)
        for i in range(reader.num_record_batches):
            if shutil.disk_usage(OUT).free < 2*1024**3+128*1024**2:
                raise RuntimeError('Disk reserve reached; partial audit preserved')
            batch = reader.get_batch(i)
            table = select_root_rows(batch, roots, offset)
            offset += batch.num_rows
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression='zstd')
            if table.num_rows:
                synids = list(map(int, table['id'].to_numpy()))
                if len(set(synids)) != len(synids) or ids.intersection(synids):
                    raise ValueError('Duplicate selected synapse ID')
                ids.update(synids)
                observed.update(pair_counts(table)); selected += table.num_rows
                score = int(table['cleft_score'].to_numpy().min())
                min_score = score if min_score is None else min(score,min_score)
                writer.write_table(table)
            if i%20 == 0 or i+1 == reader.num_record_batches:
                (OUT/'synapse-progress.json').write_text(json.dumps({'status':'scanning','batches_done':i+1,
                    'batches':reader.num_record_batches,'rows_scanned':offset,'selected_rows':selected,
                    'elapsed_seconds':time.monotonic()-start})+'\n')
            del batch, table
    if writer is not None:
        writer.close()
    temporary.replace(OUT/'selected-synapses.parquet')
    audit = pd.DataFrame(reconcile_counts(imported,observed))
    audit.to_csv(OUT/'pair-count-audit.csv',index=False)
    modeled = audit[audit.imported_pair]
    missing = modeled[modeled.archive_count < modeled.imported_count]
    excess = modeled[modeled.archive_count > modeled.imported_count]
    result = {'status':'audit_complete', 'archive_rows_scanned':offset, 'selected_rows':selected,
              'selected_ids_unique':len(ids)==selected, 'selected_min_cleft_score':min_score,
              'imported_pairs':len(imported), 'imported_synapses':sum(imported.values()),
              'observed_imported_synapses':int(modeled.archive_count.sum()),
              'exact_imported_pairs':int(modeled.exact_match.sum()),
              'every_imported_pair_exact':bool(modeled.exact_match.all()),
              'deficient_pairs':len(missing), 'excess_pairs':len(excess),
              'unmodeled_partner_pairs':int((~audit.imported_pair).sum()),
              'unmodeled_partner_sites':int(audit.loc[~audit.imported_pair,'archive_count'].sum()),
              'missing_imported_sites':int((missing.imported_count-missing.archive_count).sum()),
              'extra_sites_on_imported_pairs':int((excess.archive_count-excess.imported_count).sum()),
              'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'wall_seconds':time.monotonic()-start, 'application_changed':False,
              'sources_reverified':all(sha(ROOT/p)==h for p,h in protocol['source_hashes'].items())
                                   and sha(archive)==identity['sha256']}
    (OUT/'synapse-results.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'synapse-progress.json').write_text(json.dumps({'status':'complete'})+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
