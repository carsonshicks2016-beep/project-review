import * as THREE from 'three';

export function generateCircuit(numWaypoints = 18, baseRadius = 70, heightVariation = 25) {
    const points = [];
    
    for (let i = 0; i < numWaypoints; i++) {
        const angle = (i / numWaypoints) * Math.PI * 2;
        
        const r = baseRadius + Math.sin(angle * 3) * 18 + Math.cos(angle * 2) * 12 + Math.sin(angle * 5) * 6;
        
        const x = Math.cos(angle) * r;
        const z = Math.sin(angle) * r;
        
        let y = 8 + heightVariation * 0.4 + Math.sin(angle * 4) * heightVariation * 0.4 + Math.cos(angle * 2.5) * heightVariation * 0.2;
        y = Math.max(3.0, y);
        
        points.push(new THREE.Vector3(x, y, z));
    }
    
    const curve = new THREE.CatmullRomCurve3(points, true, 'centripetal');
    
    const gates = [];
    const numGates = 16;
    const spacedPoints = curve.getSpacedPoints(numGates);
    
    const upVector = new THREE.Vector3(0, 1, 0);
    const tangent = new THREE.Vector3();
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();
    
    for (let i = 0; i < numGates; i++) {
        const u = i / numGates;
        const pos = spacedPoints[i];
        
        tangent.copy(curve.getTangentAt(u)).normalize();
        
        right.crossVectors(upVector, tangent).normalize();
        up.crossVectors(tangent, right).normalize();
        
        gates.push({
            index: i,
            position: [pos.x, pos.y, pos.z],
            normal: [tangent.x, tangent.y, tangent.z],
            up: [up.x, up.y, up.z],
            radius: 3.0
        });
    }
    
    return { curve, gates };
}

// Rebuilds just the visual spline passing through an existing array of gates (for Editor)
export function rebuildTrackSpline(gates) {
    if (!gates || gates.length < 2) return null;
    const points = gates.map(g => new THREE.Vector3(g.position[0], g.position[1], g.position[2]));
    return new THREE.CatmullRomCurve3(points, true, 'centripetal');
}

