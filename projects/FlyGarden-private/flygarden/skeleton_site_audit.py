"""Exact geometric site-to-segment distances; no electrical compartment inference."""
import numpy as np
from scipy.spatial import cKDTree

def _project(points,starts,ends):
    vector=ends-starts;length2=np.einsum('ij,ij->i',vector,vector)
    t=np.divide(np.einsum('ij,ij->i',points-starts,vector),length2,
                out=np.zeros(len(vector)),where=length2>0)
    t=np.clip(t,0,1);delta=points-(starts+t[:,None]*vector)
    return np.einsum('ij,ij->i',delta,delta),t

def nearest_segments(points,vertices,edges):
    """Exact search using center-radius bounds grouped by segment length.

    For every edge with distance smaller than the current valid upper bound,
    its center is within upper_bound + half_length of the query point.
    Every such center is searched; this is not a nearest-vertex approximation.
    """
    points=np.asarray(points,float);vertices=np.asarray(vertices,float);edges=np.asarray(edges,np.int64)
    if points.ndim!=2 or points.shape[1]!=3 or vertices.ndim!=2 or vertices.shape[1]!=3:
        raise ValueError('Expected N by 3 coordinates')
    if edges.ndim!=2 or edges.shape[1]!=2 or not len(edges) or edges.min()<0 or edges.max()>=len(vertices):
        raise ValueError('Invalid skeleton edges')
    if not np.isfinite(points).all() or not np.isfinite(vertices).all():
        raise ValueError('Nonfinite coordinates')
    starts=vertices[edges[:,0]];ends=vertices[edges[:,1]];centers=(starts+ends)/2
    half=np.linalg.norm(ends-starts,axis=1)/2
    _,index=cKDTree(centers).query(points,workers=-1)
    best,t=_project(points,starts[index],ends[index])
    bins=np.digitize(half,[100,250,500,1000,2500,5000])
    for bin_id in sorted(set(bins),reverse=True):
        edge_ids=np.flatnonzero(bins==bin_id);tree=cKDTree(centers[edge_ids]);radius=float(half[edge_ids].max())
        for first in range(0,len(points),256):
            last=min(first+256,len(points))
            candidates=tree.query_ball_point(points[first:last],np.sqrt(best[first:last])+radius+1e-7,workers=-1)
            for local,ids in enumerate(candidates):
                if not ids:continue
                q=first+local;subset=edge_ids[ids]
                dist,fraction=_project(np.broadcast_to(points[q],(len(subset),3)),starts[subset],ends[subset])
                j=int(dist.argmin())
                if dist[j]<best[q]:best[q]=dist[j];index[q]=subset[j];t[q]=fraction[j]
    return np.sqrt(best),index,t
