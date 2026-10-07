"""Bounded, exact-ID extraction from the released v783 synapse archive."""
from collections import Counter
import numpy as np
import pyarrow as pa


def select_root_rows(batch, roots, offset=0):
    """Retain both directions and all source columns, including unmodeled edges."""
    pre = batch.column(batch.schema.get_field_index('pre_pt_root_id')).to_numpy()
    post = batch.column(batch.schema.get_field_index('post_pt_root_id')).to_numpy()
    if pre.dtype != np.int64 or post.dtype != np.int64:
        raise ValueError('Root IDs must be exact int64')
    mask = np.isin(pre, roots) | np.isin(post, roots)
    indices = np.flatnonzero(mask)
    table = pa.Table.from_batches([batch]).take(pa.array(indices))
    return table.append_column('source_row', pa.array(indices + offset, type=pa.int64()))


def pair_counts(table):
    """One count per released synapse row; no threshold or partner substitution."""
    pre = table['pre_pt_root_id'].to_numpy()
    post = table['post_pt_root_id'].to_numpy()
    return Counter(zip(map(int, pre), map(int, post)))


def reconcile_counts(imported, observed):
    rows = []
    for pair in sorted(set(imported) | set(observed)):
        expected, actual = imported.get(pair, 0), observed.get(pair, 0)
        rows.append({'pre_root_id':str(pair[0]), 'post_root_id':str(pair[1]),
                     'imported_count':expected, 'archive_count':actual,
                     'exact_match':expected == actual, 'imported_pair':pair in imported})
    return rows


def directional_endpoints(sites, roots):
    """Input reception is post location; output release is pre location.

    An internal selected-selected synapse legitimately contributes one endpoint
    to each selected neuron. An autapse has two directional endpoints as well.
    These are anatomical endpoints, not two synapse-count records.
    """
    import pandas as pd
    rows = []
    for direction, field, position in [('input','post_pt_root_id','post'),('output','pre_pt_root_id','pre')]:
        a = sites[sites[field].isin(roots)]
        rows.append(pd.DataFrame({'synapse_id':a.id.astype(str),'root_id':a[field].astype(str),
                                 'pre_root_id':a.pre_pt_root_id.astype(str),'post_root_id':a.post_pt_root_id.astype(str),
                                 'direction':direction,'source_row':a.source_row,
                                 'x':a[f'{position}_pt_position_x'],'y':a[f'{position}_pt_position_y'],'z':a[f'{position}_pt_position_z'],
                                 'source_neuropil':a.neuropil.fillna('unavailable')}))
    return pd.concat(rows, ignore_index=True)
