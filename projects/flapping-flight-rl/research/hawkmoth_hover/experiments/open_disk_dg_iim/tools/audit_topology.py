#!/usr/bin/env python3
"""Independent topology/orientation audit for the deterministic open disk mesh."""
import argparse, collections, json, math
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--radius',type=float,required=True); p.add_argument('--radial-cells',type=int,required=True); p.add_argument('--azimuthal-cells',type=int,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
R,nr,nt=a.radius,a.radial_cells,a.azimuthal_cells
if R<=0 or nr<1 or nt<6: p.error('invalid disk parameters')
pts=[(0.,0.,0.)]
for ir in range(1,nr+1):
 r=R*ir/nr
 for j in range(nt):
  t=2*math.pi*j/nt; pts.append((0.,r*math.cos(t),r*math.sin(t)))
tris=[]
def tri(x,y,z): tris.append((x,y,z))
for j in range(nt): tri(0,1+j,1+(j+1)%nt)
for ir in range(1,nr):
 inner=1+(ir-1)*nt; outer=1+ir*nt
 for j in range(nt):
  jn=(j+1)%nt; tri(inner+j,outer+j,outer+jn); tri(inner+j,outer+jn,inner+jn)
edge_count=collections.Counter(tuple(sorted((t[i],t[(i+1)%3]))) for t in tris for i in range(3))
boundary=[e for e,n in edge_count.items() if n==1]; nonmanifold=[e for e,n in edge_count.items() if n>2]
adj=collections.defaultdict(set)
for u,v in boundary: adj[u].add(v); adj[v].add(u)
components=0; seen=set()
for u in adj:
 if u in seen: continue
 components+=1; stack=[u]; seen.add(u)
 while stack:
  q=stack.pop()
  for v in adj[q]:
   if v not in seen: seen.add(v); stack.append(v)
normal_x=[]; area=0.
for i,j,k in tris:
 A,B,C=pts[i],pts[j],pts[k]
 ux,uy,uz=[B[d]-A[d] for d in range(3)]; vx,vy,vz=[C[d]-A[d] for d in range(3)]
 cx=uy*vz-uz*vy; cy=uz*vx-ux*vz; cz=ux*vy-uy*vx
 normal_x.append(cx); area+=.5*math.sqrt(cx*cx+cy*cy+cz*cz)
record={'geometry':'zero_thickness_open_planar_disk','plane':'x=0','radius':R,'radial_cells':nr,'azimuthal_cells':nt,'vertices':len(pts),'triangles':len(tris),'unique_edges':len(edge_count),'euler_characteristic':len(pts)-len(edge_count)+len(tris),'boundary_edges':len(boundary),'boundary_components':components,'nonmanifold_edges':len(nonmanifold),'all_triangle_normals_positive_x':all(v>0 for v in normal_x),'minimum_oriented_normal_x_component':min(normal_x),'triangulated_area':area,'analytic_area':math.pi*R*R,'relative_area_error':area/(math.pi*R*R)-1,'status':'PASS_TOPOLOGY_ONLY' if len(pts)-len(edge_count)+len(tris)==1 and len(boundary)==nt and components==1 and not nonmanifold and all(v>0 for v in normal_x) else 'FAIL'}
a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n'); print(json.dumps(record,indent=2,sort_keys=True))
