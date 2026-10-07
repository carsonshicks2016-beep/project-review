function wrapIndex(index, count) {
  return ((index % count) + count) % count;
}

/**
 * Project a simulator-space car position onto the nearby visual road segments.
 *
 * Python intentionally reports the nearest authoritative track sample, so its
 * road height advances in ~3 m increments. The rendered road is made from the
 * straight segments between those exact samples. Projecting onto the same
 * segments gives the viewer a continuous surface height without changing the
 * simulator, sensors, policy observations, or recorded telemetry.
 */
export function sampleTrackSurface(trackPoints, simPosition, progress = 0, searchRadius = 24) {
  const count = Array.isArray(trackPoints) ? trackPoints.length : 0;
  if (count < 2) return null;
  const x = Number(simPosition?.[0]);
  const y = Number(simPosition?.[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

  const wrappedProgress = ((Number(progress) || 0) % 1 + 1) % 1;
  const center = Math.round(wrappedProgress * count) % count;
  const radius = Math.max(2, Math.min(count - 1, Math.floor(searchRadius)));
  let best = null;

  for (let offset = -radius; offset <= radius; offset += 1) {
    const index = wrapIndex(center + offset, count);
    const nextIndex = (index + 1) % count;
    const a = trackPoints[index];
    const b = trackPoints[nextIndex];
    const dx = Number(b?.[0]) - Number(a?.[0]);
    const dy = Number(b?.[1]) - Number(a?.[1]);
    const lengthSquared = dx * dx + dy * dy;
    if (!(lengthSquared > 1e-9)) continue;
    const along = Math.max(0, Math.min(1,
      ((x - Number(a[0])) * dx + (y - Number(a[1])) * dy) / lengthSquared,
    ));
    const projectedX = Number(a[0]) + dx * along;
    const projectedY = Number(a[1]) + dy * along;
    const distanceSquared = (x - projectedX) ** 2 + (y - projectedY) ** 2;
    if (best && distanceSquared >= best.distanceSquared) continue;
    const aHeight = Number(a?.[2]) || 0;
    const bHeight = Number(b?.[2]) || aHeight;
    best = {
      height: aHeight + (bHeight - aHeight) * along,
      pitch: Math.atan2(bHeight - aHeight, Math.sqrt(lengthSquared)),
      index,
      along,
      distanceSquared,
    };
  }

  return best;
}
