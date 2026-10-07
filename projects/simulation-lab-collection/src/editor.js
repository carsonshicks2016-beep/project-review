import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { setGateState } from './scene.js';

export class TrackEditor {
  constructor(scene, camera, rendererDom, gateMeshes) {
    this.scene = scene;
    this.camera = camera;
    this.domElement = rendererDom;
    this.gateMeshes = gateMeshes;
    this.enabled = false;

    // Callbacks
    this.onGateMoved = null;
    this.onGateSelected = null;

    // Orbit Controls
    this.orbit = new OrbitControls(this.camera, this.domElement);
    this.orbit.enabled = false;

    // Transform Controls
    this.transform = new TransformControls(this.camera, this.domElement);
    this.transform.enabled = false;
    this.transform.visible = false;
    this.transform.addEventListener('dragging-changed', (event) => {
      this.orbit.enabled = !event.value && this.enabled;
    });
    this.transform.addEventListener('change', () => {
      if (this.selectedGateMesh && this.onGateMoved) {
        this.onGateMoved(this.selectedGateMesh);
      }
    });
    this.scene.add(this.transform);

    // Raycaster
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();
    this.selectedGateMesh = null;

    // Event binding
    this.onPointerDown = this.onPointerDown.bind(this);
  }

  enable() {
    this.enabled = true;
    this.orbit.enabled = true;
    this.domElement.addEventListener('pointerdown', this.onPointerDown);
    if (this.selectedGateMesh) {
      this.transform.enabled = true;
      this.transform.visible = true;
    }
  }

  disable() {
    this.enabled = false;
    this.orbit.enabled = false;
    this.transform.enabled = false;
    this.transform.visible = false;
    this.domElement.removeEventListener('pointerdown', this.onPointerDown);
  }

  setMode(mode) {
    // mode: 'translate', 'rotate', 'scale'
    this.transform.setMode(mode);
  }

  selectGate(mesh) {
    this.selectedGateMesh = mesh;
    if (mesh) {
      this.transform.attach(mesh);
      this.transform.enabled = true;
      this.transform.visible = true;
      
      // Magenta selection: distinct from every race state, and from the cyan chrome.
      setGateState(mesh, 'selected');
    } else {
      this.transform.detach();
      this.transform.enabled = false;
      this.transform.visible = false;
    }

    if (this.onGateSelected) {
      this.onGateSelected(mesh);
    }
  }

  deselectAll() {
    // Reset colors
    this.gateMeshes.forEach(mesh => {
      setGateState(mesh, 'upcoming');
    });
    this.selectGate(null);
  }

  onPointerDown(event) {
    if (!this.enabled) return;
    
    // Ignore clicks if clicking on the transform control gizmo itself
    if (this.transform.axis !== null) return; 

    // Convert mouse position to normalized device coordinates
    const rect = this.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);

    // Intersect against all torus meshes (children[0] of the gate groups)
    const targets = this.gateMeshes.map(g => g.children[0]);
    const intersects = this.raycaster.intersectObjects(targets);

    if (intersects.length > 0) {
      this.deselectAll();
      // Get the parent group
      const selectedGroup = intersects[0].object.parent;
      this.selectGate(selectedGroup);
    } else {
      this.deselectAll();
    }
  }
}
