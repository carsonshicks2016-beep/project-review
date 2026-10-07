// Recorded simulation timestamps determine playback, regardless of sampling cadence.
export function recordingIndex(frames, elapsed) {
  if (!frames.length) return 0;
  const target = frames[0].world.time + Math.max(0, elapsed);
  let low = 0, high = frames.length;
  while (low < high) {
    const mid = Math.floor((low + high) / 2);
    if (frames[mid].world.time <= target) low = mid + 1;
    else high = mid;
  }
  return Math.max(0, low - 1);
}
