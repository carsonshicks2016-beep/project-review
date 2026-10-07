"""Offline constrained terrain. Road boundary segments are triangulation constraints.
Only packaged JSON reaches Unity; historical reference media is never imported.
"""
from __future__ import annotations
import json, math, hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import shapely
from shapely.geometry import Polygon, box, LineString
from shapely.strtree import STRtree


def road_envelope(edges):
    left,right=(Polygon(e) for e in edges)
    if not left.is_valid or not right.is_valid:raise ValueError("Self-intersecting circuit edge")
    outer,inner=(left,right) if left.area>right.area else (right,left)
    if not outer.covers(inner):raise ValueError("Circuit edges cross")
    return outer.difference(inner)


def prepare(data, output: Path):
    points=np.array([[p['x'],p['y'],p['z']] for p in data['points']]); stations=np.array(data['stations']); xy=points[:,[0,2]]
    # Exactly the same central-difference direction used by Unity.
    def position(s):
        s=np.asarray(s)%data['length'];i=np.clip(np.searchsorted(stations,s,side='right')-1,0,len(stations)-2)
        u=(s-stations[i])/(stations[i+1]-stations[i]);return points[i]+(points[i+1]-points[i])*u[...,None]
    tangent=position(stations+1.5)-position(stations-1.5);tangent[:,1]=0;tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    right=np.stack([tangent[:,2],np.zeros(len(tangent)),-tangent[:,0]],axis=1)
    profiles=data['roadProfiles'];edges=[];heights=[]
    for side,key,kerb in [(-1,'left','leftKerb'),(1,'right','rightKerb')]:
        lateral=np.array([side*(p[key]+p[kerb]) for p in profiles]); y=lateral*np.tan(np.radians([p['crossfall'] for p in profiles]))
        for bowl in data['intervals']:
            blend=np.clip(np.minimum((stations-bowl['start'])/40,(bowl['end']-stations)/40),0,1);blend=blend*blend*(3-2*blend)
            into=np.clip(lateral*bowl['side']-(np.array([p['left' if bowl['side']<0 else 'right'] for p in profiles])-bowl['channelWidth']),0,bowl['channelWidth'])
            y-=bowl['channelWidth']*math.tan(math.radians(bowl['bankDegrees']))/1.3*(into/bowl['channelWidth'])**1.3*blend
        edge=points+right*lateral[:,None];edge[:,1]+=y;edges.append(edge[:,[0,2]]);heights.append(edge[:,1])
    shoulder_edges=[]
    for side,key,kerb,shoulder in [(-1,'left','leftKerb','leftShoulder'),(1,'right','rightKerb','rightShoulder')]:
        lateral=np.array([side*(p[key]+p[kerb]+p[shoulder]) for p in profiles])
        shoulder_edges.append((points+right*lateral[:,None])[:,[0,2]])
    shoulder_envelope=road_envelope(shoulder_edges)
    road=road_envelope(edges)
    if not road.is_valid: raise ValueError('Road envelope intersects itself; inspect source/profile before terrain preparation')
    # Nearby returns share this one domain, never independent swept shoulders.
    boundary_xy=np.concatenate(edges);boundary_y=np.concatenate(heights);tree=cKDTree(boundary_xy)
    starts=np.concatenate([e[:-1] for e in edges]);ends=np.concatenate([e[1:] for e in edges]);ys=np.concatenate([h[:-1] for h in heights]);ye=np.concatenate([h[1:] for h in heights]);segments=ends-starts;den=np.sum(segments*segments,axis=1)
    segtree=cKDTree((starts+ends)*.5)
    landscape_points=points[::67];landscape_tree=cKDTree(landscape_points[:,[0,2]])
    height_cache={}
    def height(x,z):
        key=(round(x,7),round(z,7))
        if key in height_cache:return height_cache[key]
        q=np.array([x,z]);_,inds=segtree.query(q,k=12);ab=segments[inds];t=np.clip(np.sum((q-starts[inds])*ab,axis=1)/den[inds],0,1)
        projected=starts[inds]+ab*t[:,None];dist=np.linalg.norm(q-projected,axis=1);j=int(np.argmin(dist));nearest=float(ys[inds[j]]*(1-t[j])+ye[inds[j]]*t[j]);d=float(dist[j])
        if d<.00001:y=nearest
        else:
            # Blend all nearby influences; exact edge constraints dominate smoothly.
            distances,ix=tree.query(q,k=128);weights=1/np.maximum(distances,.05)**4
            local=float(np.sum(boundary_y[ix]*weights)/weights.sum())
            far_dist,far_ix=landscape_tree.query(q,k=min(32,len(landscape_points)))
            far_weights=1/(far_dist**2+100**2);landscape=float(np.sum(landscape_points[far_ix,1]*far_weights)/far_weights.sum())
            profile_index=int(inds[j])%(len(points)-1);shoulder=profiles[profile_index]["leftShoulder" if int(inds[j])<len(points)-1 else "rightShoulder"]
            blend=min(1,max(0,d-shoulder)/12);blend=blend*blend*(3-2*blend)
            near=nearest*(1-blend)+local*blend
            distant=min(1,max(0,d-30)/100);distant=distant*distant*(3-2*distant)
            y=near*(1-distant)+landscape*distant+min(1,d/30)*(math.sin(x*.009)*math.cos(z*.011)*1.4)
        height_cache[key]=y;return y
    tile=64;tiles=set()
    for p in xy[::12]:
        k=np.floor(p/tile).astype(int)
        for a in range(-2,3):
            for b in range(-2,3):tiles.add((int(k[0]+a),int(k[1]+b)))
    domain=shapely.union_all([box(x*tile,z*tile,(x+1)*tile,(z+1)*tile) for x,z in sorted(tiles)])
    boundaries=[LineString(e) for e in edges]; chunks=[];total_area=0;triangle_count=0;min_area=float('inf');road_overlap=0
    for tile_index,(tx,tz) in enumerate(sorted(tiles)):
        vertices=[];triangles=[];lookup={}
        def vertex(x,z):
            key=(round(x,7),round(z,7))
            if key not in lookup:lookup[key]=len(vertices);vertices.append({'x':float(key[0]),'y':height(*key),'z':float(key[1])})
            return lookup[key]
        local_road=road.intersection(box(tx*tile-12,tz*tile-12,(tx+1)*tile+12,(tz+1)*tile+12))
        local_shoulders=shoulder_envelope.intersection(box(tx*tile-16,tz*tile-16,(tx+1)*tile+16,(tz+1)*tile+16))
        # Exact road/kerb and shoulder perimeters are triangulation constraints.
        for dz in range(0,tile,8):
            for dx in range(0,tile,8):
                x=tx*tile+dx;z=tz*tile+dz;cell=box(x,z,x+8,z+8)
                near=not local_road.is_empty and local_road.distance(cell)<12;step=4
                for zz in range(z,z+8,step):
                    for xx in range(x,x+8,step):
                        area=box(xx,zz,xx+step,zz+step).difference(local_road)
                        if area.is_empty:continue
                        partitions=[area.intersection(local_shoulders),area.difference(local_shoulders)]
                        for tri in [t for part in partitions if not part.is_empty for t in shapely.constrained_delaunay_triangles(part).geoms]:
                            if tri.area<1e-8:continue
                            coords=list(tri.exterior.coords)[:3];indices=[vertex(*p) for p in coords]
                            # Unity's upward winding is clockwise in the X/Z plane.
                            a,b,c=coords
                            if (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])>0:indices.reverse()
                            triangles.extend(indices);total_area+=tri.area;min_area=min(min_area,tri.area);triangle_count+=1
        # Skirts only on exterior tile edges. No internal seams or contact walls.
        skirts=[]
        for delta,a,b in [((-1,0),(tx*tile,tz*tile),(tx*tile,(tz+1)*tile)),((1,0),((tx+1)*tile,(tz+1)*tile),((tx+1)*tile,tz*tile)),((0,-1),((tx+1)*tile,tz*tile),(tx*tile,tz*tile)),((0,1),(tx*tile,(tz+1)*tile),((tx+1)*tile,(tz+1)*tile))]:
            if (tx+delta[0],tz+delta[1]) in tiles:continue
            for i in range(8):
                p=np.array(a)+(np.array(b)-a)*(i/8);q=np.array(a)+(np.array(b)-a)*((i+1)/8)
                skirts.append([{'x':float(p[0]),'y':height(*p),'z':float(p[1])},{'x':float(q[0]),'y':height(*q),'z':float(q[1])}])
        skirt_vertices=[];skirt_triangles=[]
        for p,q in skirts:
            b=len(skirt_vertices);skirt_vertices.extend([p,q,{**p,'y':-650},{**q,'y':-650}]);skirt_triangles.extend([b,b+2,b+1,b+1,b+2,b+3,b+1,b+2,b,b+3,b+2,b+1])
        chunks.append({'name':f'Ground_{tx}_{tz}','vertices':vertices,'triangles':triangles,'skirtVertices':[],'skirtTriangles':[]})
        if tile_index%200==0:print(f'Terrain tiles {tile_index}/{len(tiles)}',flush=True)
    # A route corridor leaves visible cliffs across the interior and at distant horizons.
    # Fill the whole surrounding domain beyond the 1200m camera far plane. The
    # existing exact near-road tiles remain authoritative; this disjoint extension
    # shares their 4m boundary vertices and never adds a stacked contact surface.
    x0,z0,x1,z1=domain.bounds
    outer=box(math.floor((x0-1500)/256)*256,math.floor((z0-1500)/256)*256,
              math.ceil((x1+1500)/256)*256,math.ceil((z1+1500)/256)*256)
    extension=outer.difference(domain)
    for tx in range(int(outer.bounds[0]),int(outer.bounds[2]),256):
        for tz in range(int(outer.bounds[1]),int(outer.bounds[3]),256):
            patch=extension.intersection(box(tx,tz,tx+256,tz+256))
            if patch.is_empty:continue
            vertices=[];triangles=[];lookup={}
            for z in range(tz,tz+256,64):
                for x in range(tx,tx+256,64):
                    area=patch.intersection(box(x,z,x+64,z+64))
                    if area.is_empty:continue
                    # Collinear constraints force matching edge vertices between
                    # coarse cells, outer tiles, and the original near-road tiles.
                    area=shapely.segmentize(area,4)
                    for tri in shapely.constrained_delaunay_triangles(area).geoms:
                        if tri.area<1e-8:continue
                        coords=list(tri.exterior.coords)[:3];indices=[vertex(*p) for p in coords]
                        a,b,c=coords
                        if (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])>0:indices.reverse()
                        triangles.extend(indices);total_area+=tri.area;min_area=min(min_area,tri.area);triangle_count+=1
            skirts=[];skirt_triangles=[]
            border=box(tx,tz,tx+256,tz+256).boundary.intersection(outer.boundary)
            for line in getattr(border,'geoms',[border]):
                if line.geom_type!='LineString':continue
                coords=list(shapely.segmentize(line,4).coords)
                for a,b in zip(coords,coords[1:]):
                    n=len(skirts);p={'x':a[0],'y':height(*a),'z':a[1]};q={'x':b[0],'y':height(*b),'z':b[1]}
                    skirts.extend([p,q,{**p,'y':-650},{**q,'y':-650}]);skirt_triangles.extend([n,n+2,n+1,n+1,n+2,n+3,n+1,n+2,n,n+3,n+2,n+1])
            chunks.append({'name':f'GroundOuter_{tx}_{tz}','vertices':vertices,'triangles':triangles,'skirtVertices':skirts,'skirtTriangles':skirt_triangles})
    expected=outer.difference(road).area

    if abs(total_area-expected)>max(.01,expected*1e-8):raise ValueError('Terrain triangulation coverage mismatch')
    if any(p['ground']=='soil' for p in profiles):
        partitioned=[]
        for chunk in chunks:
            verts=np.array([[v['x'],v['z']] for v in chunk['vertices']]);faces=np.array(chunk['triangles']).reshape(-1,3)
            if not len(faces):continue
            centers=verts[faces].mean(axis=1);dist,ix=segtree.query(centers,k=1)
            soil=np.array([profiles[int(i)%(len(points)-1)]['ground']=='soil' for i in ix]) & (dist<6)
            for mask,name in [(~soil,chunk['name']),(soil,chunk['name'].replace('Ground_','GroundSoil_'))]:
                if np.any(mask):partitioned.append({**chunk,'name':name,'triangles':faces[mask].reshape(-1).tolist(),'skirtVertices':chunk['skirtVertices'] if not name.startswith('GroundSoil') else [],'skirtTriangles':chunk['skirtTriangles'] if not name.startswith('GroundSoil') else []})
        chunks=partitioned
    # Average incident faces across tile boundaries before packaging normals.
    normal_sums={}
    for chunk in chunks:
        verts=np.array([[v['x'],v['y'],v['z']] for v in chunk['vertices']]);faces=np.array(chunk['triangles'],dtype=int).reshape(-1,3)
        if len(faces)==0:continue
        face_normals=np.cross(verts[faces[:,1]]-verts[faces[:,0]],verts[faces[:,2]]-verts[faces[:,0]])
        sums=np.zeros_like(verts)
        for lane in range(3):np.add.at(sums,faces[:,lane],face_normals)
        for v,n in zip(chunk['vertices'],sums):
            key=(round(v['x'],7),round(v['z'],7))
            if key in normal_sums:normal_sums[key]+=n
            else:normal_sums[key]=n.copy()
    for chunk in chunks:
        chunk['normals']=[]
        for v in chunk['vertices']:
            n=normal_sums[(round(v['x'],7),round(v['z'],7))];length=np.linalg.norm(n);n=n/length if length>1e-12 else np.array([0,1,0])
            chunk['normals'].append(dict(zip(('x','y','z'),map(float,n))))
    result={'schema':1,'revision':data['revision'],'generatorVersion':data['generatorVersion'],'chunks':chunks}
    output.write_text(json.dumps(result,separators=(',',':'))+'\n')
    report={'revision':data['revision'],'generatorVersion':data['generatorVersion'],'tiles':len(chunks),'outerPaddingMetres':1500,'shoulderBoundaryConstraints':True,'triangles':triangle_count,'minTriangleArea':min_area,'area':total_area,'expectedArea':expected,'areaError':abs(total_area-expected),'terrainHash':hashlib.sha256(output.read_bytes()).hexdigest(),'groundModel':'source-edge constrained, shared synthesized terrain','roadEnvelopeValid':road.is_valid,'maxBoundaryHeightMismatch':max(abs(height(float(p[0]),float(p[1]))-float(y)) for e,h in zip(edges,heights) for p,y in zip(e,h)),'gradeSeparation':'no nonlocal source road-envelope intersection; reference-supported external Döttinger Höhe overpass constructed separately with estimated dimensions'}
    (output.parent.parent.parent.parent/'art-source/circuits/nordschleife/terrain-report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    from rallylab.circuit import ROOT,convert
    print(json.dumps(prepare(convert(),ROOT/'Assets/Resources/Circuits/NordschleifeTerrain.json'),indent=2))
