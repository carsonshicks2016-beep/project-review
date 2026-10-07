import * as THREE from 'three';
import { DRONE_SCALE } from './scene.js';

export class CameraSystem {
    constructor(droneGroup) {
        const aspect = window.innerWidth / window.innerHeight;
        
        this.fpvCamera = new THREE.PerspectiveCamera(100, aspect, 0.1, 1000);
        this.chaseCamera = new THREE.PerspectiveCamera(70, aspect, 0.1, 1000);
        this.tracksideCamera = new THREE.PerspectiveCamera(70, aspect, 0.1, 1500);
        this.broadcastCamera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1500);
        this.orbitCamera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1500);
        this.selfieCamera = new THREE.PerspectiveCamera(80, aspect, 0.1, 1000);
        
        // Position trackside camera way back in the corner of the room (static for editor)
        this.tracksideCamera.position.set(20, 25, 45);
        this.tracksideCamera.lookAt(0, 5, 0);
        
        // Broadcast camera starts in a nice high corner
        this.broadcastCamera.position.set(30, 30, 50);
        
        // The FPV camera is placed in world space by updateFPV(), using _fpvOffset
        // (derived from DRONE_SCALE so it clears the front props) rather than being
        // parented to the airframe.
        
        // Selfie Mount (In front of drone, looking backward)
        this.selfieMount = new THREE.Object3D();
        this.selfieMount.position.set(0, 4.0, -18.0);
        this.selfieMount.rotation.y = Math.PI; // Spin 180 degrees
        this.selfieMount.rotation.x = THREE.MathUtils.degToRad(-15); // Look slightly down
        this.selfieMount.add(this.selfieCamera);
        
        if (droneGroup) {
            droneGroup.add(this.selfieMount);
        }
        
        // FPV framing. Real goggle footage is rigid to the airframe, which is only
        // watchable when the pilot is smooth. Retaining a fraction of roll/pitch keeps the
        // bank-into-the-turn feel that makes FPV look like flying, while holding enough
        // horizon that you can tell which way is up.
        this.fpvStabilized = true;
        this.fpvRollRetain = 0.35;
        this.fpvPitchRetain = 0.55;
        this.fpvTau = 0.07;          // Orientation smoothing time constant, seconds
        this.fpvUpTilt = THREE.MathUtils.degToRad(30);

        this._fpvOffset = new THREE.Vector3(0, 0.11 * DRONE_SCALE, -0.34 * DRONE_SCALE);
        this._fpvTarget = new THREE.Quaternion();
        this._euler = new THREE.Euler(0, 0, 0, 'YXZ');
        this._tmpVec = new THREE.Vector3();
        this._fpvInit = false;

        // Updated Camera Cycle List
        this.modes = ['chase', 'fpv', 'trackside', 'broadcast', 'orbit', 'selfie'];
        this.modeIndex = 0;
        this.mode = this.modes[this.modeIndex];
        this.activeCamera = this.chaseCamera;
        
        this._droneQuat = new THREE.Quaternion();
        this._dronePos = new THREE.Vector3();
        this._targetPos = new THREE.Vector3();
        this._lookTarget = new THREE.Vector3();
        
        this.time = 0;
    }
    
    toggle() {
        this.modeIndex = (this.modeIndex + 1) % this.modes.length;
        this.mode = this.modes[this.modeIndex];
        
        if (this.mode === 'chase') {
            this.activeCamera = this.chaseCamera;
        } else if (this.mode === 'fpv') {
            this.activeCamera = this.fpvCamera;
        } else if (this.mode === 'trackside') {
            this.activeCamera = this.tracksideCamera;
        } else if (this.mode === 'broadcast') {
            this.activeCamera = this.broadcastCamera;
        } else if (this.mode === 'orbit') {
            this.activeCamera = this.orbitCamera;
        } else if (this.mode === 'selfie') {
            this.activeCamera = this.selfieCamera;
        }
    }
    
    update(dronePos, droneQuat, dt) {
        this.time += dt;
        
        if (Array.isArray(droneQuat) || droneQuat instanceof Float32Array) {
            this._droneQuat.set(droneQuat[1], droneQuat[2], droneQuat[3], droneQuat[0]);
        } else {
            this._droneQuat.copy(droneQuat);
        }
        
        if (Array.isArray(dronePos) || dronePos instanceof Float32Array) {
            this._dronePos.set(dronePos[0], dronePos[1], dronePos[2]);
        } else {
            this._dronePos.copy(dronePos);
        }
        
        if (this.mode === 'fpv') {
            this.updateFPV(dt);
            return;
        }

        if (this.mode === 'trackside' || this.mode === 'selfie') {
            // Static or hard-mounted cameras need no dynamic positional updates here
            return;
        }
        
        if (this.mode === 'broadcast') {
            // Broadcast dynamically tracks the drone but stays fixed in space
            this.broadcastCamera.lookAt(this._dronePos);
            return;
        }

        if (this.mode === 'orbit') {
            // Smooth cinematic rotation around the drone
            const radius = 42.0; // 25 put the camera inside the LiDAR cloud
            const height = 14.0;
            const speed = 0.5;
            const cx = this._dronePos.x + Math.cos(this.time * speed) * radius;
            const cz = this._dronePos.z + Math.sin(this.time * speed) * radius;
            
            // Lerp the orbit camera for super smooth tracking
            this._targetPos.set(cx, this._dronePos.y + height, cz);
            this.orbitCamera.position.lerp(this._targetPos, 1 - Math.exp(-dt / 0.35));
            this.orbitCamera.lookAt(this._dronePos);
            return;
        }

        if (this.mode === 'chase') {
            // Extract Euler angles to stabilize Pitch (X) and Roll (Z), keeping ONLY Yaw (Y) dynamic
            const euler = new THREE.Euler().setFromQuaternion(this._droneQuat, 'YXZ');
            euler.x = 0; // Keep camera perfectly level (no pitching up/down)
            euler.z = 0; // Keep camera perfectly level (no banking/rolling)
            const stabilizedQuat = new THREE.Quaternion().setFromEuler(euler);
            
            // Compute target position: 20m behind (+Z) and 6m above (+Y) the giant drone
            this._targetPos.set(0, 6.0, 20.0);
            this._targetPos.applyQuaternion(stabilizedQuat);
            this._targetPos.add(this._dronePos);
            
            // Lerp camera position toward target for smooth follow
            this.chaseCamera.position.lerp(this._targetPos, 1 - Math.exp(-dt / 0.22));
            
            // Look slightly ahead of the drone to maintain a consistent preset viewing angle
            this._lookTarget.set(0, 0, -5);
            this._lookTarget.applyQuaternion(stabilizedQuat);
            this._lookTarget.add(this._dronePos);
            
            this.chaseCamera.lookAt(this._lookTarget);
        }
    }
    
    // Every camera needs the new aspect, not just the active one, otherwise switching
    // cameras after a resize shows a stretched view. `aspect` is the viewport's, which
    // is narrower than the window because of the two side panels.
    updateFPV(dt) {
        // Camera sits at the airframe's camera position, which does move with the drone.
        this._tmpVec.copy(this._fpvOffset).applyQuaternion(this._droneQuat).add(this._dronePos);
        this.fpvCamera.position.copy(this._tmpVec);

        if (this.fpvStabilized) {
            // Decompose in YXZ so yaw is independent of the other two, then scale back
            // pitch and roll toward level.
            this._euler.setFromQuaternion(this._droneQuat, 'YXZ');
            this._euler.x = this._euler.x * this.fpvPitchRetain - this.fpvUpTilt;
            this._euler.z *= this.fpvRollRetain;
            this._fpvTarget.setFromEuler(this._euler);
        } else {
            this._euler.set(-this.fpvUpTilt, 0, 0, 'YXZ');
            this._fpvTarget.copy(this._droneQuat).multiply(new THREE.Quaternion().setFromEuler(this._euler));
        }

        if (!this._fpvInit) {
            this.fpvCamera.quaternion.copy(this._fpvTarget);
            this._fpvInit = true;
            return;
        }

        // Frame-rate independent smoothing: a fixed slerp factor per frame would make the
        // damping depend on how fast the machine renders.
        const k = 1 - Math.exp(-dt / Math.max(1e-4, this.fpvTau));
        this.fpvCamera.quaternion.slerp(this._fpvTarget, k);
    }

    resize(aspect = window.innerWidth / window.innerHeight) {
        for (const cam of [
            this.fpvCamera, this.chaseCamera, this.tracksideCamera,
            this.broadcastCamera, this.orbitCamera, this.selfieCamera
        ]) {
            cam.aspect = aspect;
            cam.updateProjectionMatrix();
        }
    }
    
    getCamera() {
        return this.activeCamera;
    }
}
