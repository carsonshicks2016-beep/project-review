import hashlib
import json
import math
import unittest
from unittest.mock import patch
from copy import deepcopy
from fastapi.testclient import TestClient
from rallylab import circuit
from rallylab.api import app, CourseSpec


class CircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=circuit.convert()

    def test_source_snapshot_and_conversion_are_pinned(self):
        report=json.loads((circuit.SOURCE.parent/'import-report.json').read_text())
        self.assertEqual(report['sourceHash'],circuit.digest(circuit.SOURCE))
        self.assertEqual(self.data,json.loads((circuit.ROOT/'Assets/Resources/Circuits/Nordschleife.json').read_text()))
        self.assertEqual(self.data,circuit.convert())

    def test_periodic_seam_and_horizontal_length(self):
        d=self.data
        self.assertEqual(d['points'][0],d['points'][-1])
        self.assertAlmostEqual(d['stations'][-1],20832,delta=1)
        length=sum(math.hypot(b['x']-a['x'],b['z']-a['z']) for a,b in zip(d['points'],d['points'][1:]))
        self.assertAlmostEqual(length,d['length'],places=6)
        self.assertTrue(all(0<b-a<=1.500001 for a,b in zip(d['stations'],d['stations'][1:])))

    def test_source_vertices_preserved_after_translation(self):
        raw=json.loads(circuit.SOURCE.read_text());d=self.data;o=d['origin']
        points={(round(p['x'],4),round(p['y'],4),round(p['z'],4)) for p in d['points']}
        for xy,elev in zip(raw['center'],raw['elevation']):
            self.assertIn((round(xy[0]-o['x'],4),round(elev-o['y'],4),round(xy[1]-o['z'],4)),points)

    def test_all_sectors_partition_the_route(self):
        variants=[circuit.variant(self.data,i,1) for i in range(16)]
        self.assertEqual(variants[0]['startStation'],0)
        self.assertEqual(variants[-1]['endStation'],self.data['length'])
        for a,b in zip(variants,variants[1:]): self.assertEqual(a['endStation'],b['startStation'])
        self.assertTrue(all(v['episodeSeconds']==240 for v in variants))
        self.assertEqual(circuit.variant(self.data)['episodeSeconds'],1800)

    def test_variant_bounds_and_procedural_limits(self):
        for first,count in [(-1,1),(16,1),(15,2),(0,0),(0,17)]:
            with self.assertRaises(ValueError): circuit.variant(self.data,first,count)
        with self.assertRaises(ValueError):CourseSpec(name='Ring',seed=1,family='gentle',length=20832)

    def test_import_api_is_allowlisted_and_rejects_wrapping_range(self):
        client=TestClient(app)
        with patch('rallylab.api.core.create',return_value={'id':'test'}) as create:
            self.assertEqual(client.post('/api/circuits/import',json={'firstSector':9,'sectorCount':1}).status_code,200)
            self.assertEqual(create.call_args.args[0],'circuit')
            self.assertEqual(client.post('/api/circuits/import',json={'circuit':'arbitrary-path'}).status_code,422)
            self.assertEqual(client.post('/api/circuits/import',json={'firstSector':15,'sectorCount':2}).status_code,409)

    def test_visual_circuit_review_cannot_enable_training(self):
        client=TestClient(app)
        candidate={'id':'stage-c','suite':'library','definition':{'topology':'circuit'},'trainability':{'stage':'pending'}}
        with patch('rallylab.api.core.course_asset_valid',return_value=True), patch('rallylab.api.core.read',return_value=candidate), patch('rallylab.api.core.write') as write:
            response=client.post('/api/courses/stage-c/review')
            self.assertEqual(response.status_code,409)
            write.assert_not_called()

    def test_estimated_profiles_and_t13_identity_are_explicit(self):
        self.assertIn('not surveyed',self.data['startReference'])
        self.assertTrue(all(s['confidence']=='estimated' for s in self.data['intervals']))
        self.assertEqual(next(l['station'] for l in self.data['landmarks'] if l['name']=='T13'),0)

    def test_complete_profile_coverage_and_resolved_seam(self):
        profile=json.loads(circuit.PROFILES.read_text());circuit.validate_profiles(profile,self.data['length'])
        self.assertEqual(len(self.data['roadProfiles']),len(self.data['points']))
        self.assertEqual(self.data['roadProfiles'][0],self.data['roadProfiles'][-1])
        self.assertEqual(self.data['generatorVersion'],'circuit-geometry-v7')
        self.assertTrue(any(p['leftKerb']>0 for p in self.data['roadProfiles']))
        self.assertTrue(any(p['rightKerb']==0 for p in self.data['roadProfiles']))
        self.assertTrue(all(s['videoSeconds'] and s['evidence'] for s in profile['sections']))

    def test_profiles_reject_gaps_nonfinite_and_invalid_edges(self):
        original=json.loads(circuit.PROFILES.read_text())
        for change in ('gap','nan','edge','height'):
            p=deepcopy(original)
            if change=='gap':p['sections'][1]['start']+=1
            if change=='nan':p['sections'][0]['values']['left']=float('nan')
            if change=='height':p['edgeFeatures'][0]['height']=float('nan')
            if change=='edge':p['edgeFeatures'][0]['end']=p['edgeFeatures'][0]['start']-1
            with self.assertRaises(ValueError):circuit.validate_profiles(p,self.data['length'])

    def test_left_and_right_kerb_dimensions_do_not_override_each_other(self):
        import tempfile
        from pathlib import Path
        p=json.loads(circuit.PROFILES.read_text());left=dict(p['edgeFeatures'][0]);left.update(side=-1,height=.045,bevel=.12)
        right=dict(left);right.update(side=1,height=.11,bevel=.25)
        p['edgeFeatures']=[left,right]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json';path.write_text(json.dumps(p));d=circuit.convert(profiles=path)
        sample=next(v for s,v in zip(d['stations'],d['roadProfiles']) if left['start']+25<s<left['end']-25)
        self.assertEqual(sample['leftKerbHeight'],.045);self.assertEqual(sample['rightKerbHeight'],.11)
        self.assertEqual(sample['leftKerbBevel'],.12);self.assertEqual(sample['rightKerbBevel'],.25)

    def test_packaged_terrain_hash_and_coverage(self):
        report=json.loads((circuit.SOURCE.parent/'terrain-report.json').read_text())
        self.assertEqual(report['revision'],self.data['revision'])
        self.assertEqual(report['terrainHash'],circuit.digest(circuit.ROOT/'Assets/Resources/Circuits/NordschleifeTerrain.json'))
        self.assertLess(report['areaError'],.01)
        self.assertGreaterEqual(report['outerPaddingMetres'],1200)
        self.assertTrue(report['shoulderBoundaryConstraints'])
        self.assertLess(report['maxBoundaryHeightMismatch'],.01)
        self.assertGreater(report['minTriangleArea'],0)

    def test_closed_terrain_envelope_has_no_station_seam_cut(self):
        from rallylab.circuit_terrain import road_envelope
        from shapely.geometry import Point
        left=[(-12,-12),(12,-12),(12,12),(-12,12),(-12,-12)]
        right=[(-8,-8),(8,-8),(8,8),(-8,8),(-8,-8)]
        band=road_envelope((left,right))
        self.assertAlmostEqual(band.area,576-256)
        self.assertTrue(band.contains(Point(-10,-11)))
        self.assertFalse(band.contains(Point(0,0)))


if __name__=='__main__':unittest.main()
