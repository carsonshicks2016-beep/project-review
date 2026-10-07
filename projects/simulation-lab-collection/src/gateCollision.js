export function checkGatePassing(prevPos, curPos, gate) {
  const gPos = gate.position;
  const n = gate.normal;

  // Signed distance to gate plane for both positions
  const distPrev = (prevPos[0] - gPos[0]) * n[0] + (prevPos[1] - gPos[1]) * n[1] + (prevPos[2] - gPos[2]) * n[2];
  const distCur = (curPos[0] - gPos[0]) * n[0] + (curPos[1] - gPos[1]) * n[1] + (curPos[2] - gPos[2]) * n[2];

  // Check crossing in either direction (sign change between frames)
  const crossed = (distPrev > 0 && distCur <= 0) || (distPrev < 0 && distCur >= 0);
  if (crossed) {
    const denom = distPrev - distCur;
    if (Math.abs(denom) < 1e-10) return { status: 'NONE' };
    const t = distPrev / denom;
    const cx = prevPos[0] + t * (curPos[0] - prevPos[0]);
    const cy = prevPos[1] + t * (curPos[1] - prevPos[1]);
    const cz = prevPos[2] + t * (curPos[2] - prevPos[2]);

    const distToCenter = Math.sqrt(
      (cx - gPos[0])**2 +
      (cy - gPos[1])**2 +
      (cz - gPos[2])**2
    );

    if (distToCenter <= gate.radius) return { status: 'PASSED' };
    if (distToCenter <= gate.radius + 0.35) return { status: 'CRASHED_FRAME' };
    return { status: 'MISSED' };
  }

  return { status: 'NONE' };
}

export function distanceToGate(pos, gate) {
  return Math.sqrt(
    (pos[0] - gate.position[0])**2 +
    (pos[1] - gate.position[1])**2 +
    (pos[2] - gate.position[2])**2
  );
}
