export type CourseContext = {
  id: string;
  name: string;
  suite?: string;
  review?: string;
  legacyObstacleLayout?: boolean;
};

type JobContext = {
  id: string;
  kind: string;
  state: string;
  updated?: number;
  spec?: Record<string, unknown>;
  summary?: {complete?: boolean};
};

type CheckpointContext = {id: string; run: string};

export type CheckpointSettings = {
  checkpoint: string;
  courses: string[];
  attempts: number;
  seed: number;
  deterministic: boolean;
  timeScale: number;
  startingGear: 'neutral' | 'first';
};

export type CheckpointDefaults = {
  settings: CheckpointSettings;
  courseSource: 'specialist' | 'last-evaluation' | 'preview' | 'none';
  courseNote: string;
};

function text(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function validGear(value: unknown): 'neutral' | 'first' | undefined {
  return value === 'neutral' || value === 'first' ? value : undefined;
}

function reviewedPreviewCourse(courses: CourseContext[]): CourseContext | undefined {
  return courses.find(course => course.suite === 'library' && course.review === 'reviewed' && !course.legacyObstacleLayout)
    ?? courses.find(course => course.suite === 'library' && !course.legacyObstacleLayout)
    ?? courses.find(course => !course.legacyObstacleLayout);
}

export function checkpointDefaults(
  checkpoint: CheckpointContext,
  jobs: JobContext[],
  courses: CourseContext[],
  current: CheckpointSettings,
  intent: 'viewer' | 'evaluation',
): CheckpointDefaults {
  const sourceRun = jobs.find(job => job.id === checkpoint.run && job.kind === 'training');
  const mode = text(sourceRun?.spec?.mode);
  const courseById = new Map(courses.map(course => [course.id, course]));

  let course = mode === 'specialist'
    ? courseById.get(text(sourceRun?.spec?.course))
    : undefined;
  let courseSource: CheckpointDefaults['courseSource'] = course ? 'specialist' : 'none';

  if (!course) {
    const priorEvaluations = jobs
      .filter(job => job.kind === 'evaluation' && job.state === 'completed' && job.summary?.complete
        && job.spec?.checkpoint === checkpoint.id)
      .sort((a, b) => (b.updated ?? 0) - (a.updated ?? 0));
    for (const evaluation of priorEvaluations) {
      const ids = evaluation.spec?.courses;
      if (!Array.isArray(ids)) continue;
      course = ids.map(id => courseById.get(text(id))).find(Boolean);
      if (course) {
        courseSource = 'last-evaluation';
        break;
      }
    }
  }

  if (!course) {
    course = reviewedPreviewCourse(courses);
    if (course) courseSource = 'preview';
  }

  const courseNote = courseSource === 'specialist'
    ? 'Using the frozen course recorded by this specialist run.'
    : courseSource === 'last-evaluation'
      ? mode === 'generalist'
        ? 'This generalist has no fixed course; using its most recent completed evaluation course.'
        : 'Using the most recent completed evaluation course for this checkpoint.'
      : courseSource === 'preview'
        ? mode === 'specialist'
          ? 'The specialist course is unavailable; this reviewed course is shown as a preview.'
          : 'This generalist has no fixed course or completed evaluation; using a reviewed library course as a preview.'
        : 'No saved course is available for this checkpoint.';

  const settings: CheckpointSettings = {
    ...current,
    checkpoint: checkpoint.id,
    courses: course ? [course.id] : [],
    startingGear: validGear(sourceRun?.spec?.startingGear) ?? current.startingGear,
  };

  if (intent === 'viewer') {
    settings.attempts = 1;
    settings.deterministic = true;
    const seed = sourceRun?.spec?.seed;
    if (typeof seed === 'number' && Number.isSafeInteger(seed) && seed >= 0) settings.seed = seed;
  }

  return {settings, courseSource, courseNote};
}
