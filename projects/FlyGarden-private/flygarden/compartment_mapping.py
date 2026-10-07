"""Read-only anatomical audit helpers. No inferred label becomes a controller compartment."""
import re
import numpy as np


class CoordinateDecoder:
    """Decode explicit/carry-forward decimal root IDs without float conversion.

    The sampled source encoding carries both columns independently. Raw line
    numbers and source hashes must accompany derived coordinates. Full pair-count
    comparisons are required before attributing these rows to imported edges.
    """
    def __init__(self):
        self.pre = None
        self.post = None

    def decode(self, row):
        if len(row) != 5:
            raise ValueError('Expected five coordinate columns')
        for i in (0, 1):
            if row[i] and (not row[i].isdigit() or int(row[i]) <= 0):
                raise ValueError('Invalid decimal root ID')
        pre = row[0] or self.pre
        post = row[1] or self.post
        if not pre or not post:
            raise ValueError('Missing initial root ID')
        point = tuple(int(s) for s in row[2:])
        if any(v < 0 for v in point):
            raise ValueError('Negative coordinate')
        self.pre, self.post = pre, post
        return pre, post, point


def partner_glomerulus(annotation):
    """Return only explicit single-glomerulus partner labels, never a branch label."""
    cell_type = annotation.get('cell_type', '') or ''
    if re.fullmatch(r'ORN_[A-Za-z0-9]+', cell_type):
        return cell_type[4:], 'ORN_annotation'
    if annotation.get('cell_class') == 'ALPN' and annotation.get('cell_sub_class') == 'uniglomerular':
        match = re.fullmatch(r'([A-Za-z0-9]+)_(?:adPN|lPN|vPN|lvPN|ilPN|l2PN|l3PN|lv2PN)', cell_type)
        if match:
            return match[1], 'uniglomerular_PN_annotation'
    return None, 'unavailable'


def unique_mesh_assignment(memberships, region_names):
    """Accept exactly one enclosed region; keep overlap and outside unavailable."""
    m = np.asarray(memberships, bool)
    if m.ndim != 2 or m.shape[1] != len(region_names):
        raise ValueError('Invalid membership matrix')
    counts = m.sum(axis=1)
    labels = np.full(len(m), 'unavailable_outside', dtype=object)
    labels[counts > 1] = 'unavailable_overlap'
    for i, name in enumerate(region_names):
        labels[(counts == 1) & m[:, i]] = str(name)
    return labels
