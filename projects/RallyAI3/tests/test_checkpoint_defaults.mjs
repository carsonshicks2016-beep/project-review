import assert from 'node:assert/strict';
import test from 'node:test';
import {checkpointDefaults} from '../dashboard/src/checkpointDefaults.ts';

const courses = [
  {id: 'library-a', name: 'Reviewed course', suite: 'library', review: 'reviewed'},
  {id: 'library-b', name: 'Another course', suite: 'library', review: 'reviewed'},
];
const current = {
  checkpoint: '', courses: [], attempts: 20, seed: 2026, deterministic: false,
  timeScale: 10, startingGear: 'neutral',
};

test('specialist checkpoints default to their frozen course and run settings', () => {
  const checkpoint = {id: 'cp-specialist', run: 'run-specialist'};
  const jobs = [{
    id: checkpoint.run, kind: 'training', state: 'completed',
    spec: {mode: 'specialist', course: 'library-b', seed: 771, startingGear: 'first'},
  }];

  const result = checkpointDefaults(checkpoint, jobs, courses, current, 'viewer');

  assert.deepEqual(result.settings, {
    ...current, checkpoint: checkpoint.id, courses: ['library-b'], attempts: 1,
    seed: 771, deterministic: true, startingGear: 'first',
  });
  assert.equal(result.courseSource, 'specialist');
});

test('generalist checkpoints reuse the most recent completed evaluation course', () => {
  const checkpoint = {id: 'cp-generalist', run: 'run-generalist'};
  const jobs = [
    {id: checkpoint.run, kind: 'training', state: 'completed', spec: {mode: 'generalist', seed: 91}},
    {id: 'old', kind: 'evaluation', state: 'completed', updated: 10, summary: {complete: true}, spec: {checkpoint: checkpoint.id, courses: ['library-a']}},
    {id: 'latest', kind: 'evaluation', state: 'completed', updated: 20, summary: {complete: true}, spec: {checkpoint: checkpoint.id, courses: ['library-b']}},
    {id: 'partial', kind: 'evaluation', state: 'completed', updated: 30, summary: {complete: false}, spec: {checkpoint: checkpoint.id, courses: ['library-a']}},
  ];

  const result = checkpointDefaults(checkpoint, jobs, courses, current, 'evaluation');

  assert.deepEqual(result.settings.courses, ['library-b']);
  assert.equal(result.settings.seed, current.seed);
  assert.equal(result.settings.deterministic, current.deterministic);
  assert.equal(result.courseSource, 'last-evaluation');
});

test('unevaluated generalists get a reviewed preview course and honest note', () => {
  const checkpoint = {id: 'cp-new-generalist', run: 'run-generalist'};
  const jobs = [{id: checkpoint.run, kind: 'training', state: 'completed', spec: {mode: 'generalist'}}];
  const result = checkpointDefaults(checkpoint, jobs, courses, current, 'viewer');

  assert.deepEqual(result.settings.courses, ['library-a']);
  assert.match(result.courseNote, /preview/);
  assert.equal(result.courseSource, 'preview');
});
