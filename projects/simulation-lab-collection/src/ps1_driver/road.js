// 2D Faux-3D Track & Road Geometry Engine

export const ROAD_CONSTANTS = {
  SEGMENT_LENGTH: 200,
  ROAD_WIDTH: 2000,
  RUMBLE_LENGTH: 3,
  LANES: 3,
  DRAW_DISTANCE: 260,
  CAMERA_HEIGHT: 1000,
  FIELD_OF_VIEW: 100, // degrees
  CAMERA_DEPTH: null // calculated from FOV
};
ROAD_CONSTANTS.CAMERA_DEPTH = 1 / Math.tan((ROAD_CONSTANTS.FIELD_OF_VIEW / 2) * Math.PI / 180);

export const TRACK_THEMES = {
  touge: {
    id: 'touge',
    name: 'Akina Touge',
    sub: 'Downhill Mountain Pass',
    sky: { top: '#1c132b', mid: '#612b48', bot: '#c95e54' },
    fog: '#612b48',
    roadDark: '#23262f',
    roadLight: '#2c303c',
    grassDark: '#1c361e',
    grassLight: '#234426',
    rumbleDark: '#a31c1c',
    rumbleLight: '#dfdfdf',
    laneColor: '#e0ded0',
    propSet: ['pine_tree', 'guardrail_l', 'guardrail_r', 'rock_cliff', 'sign_turn']
  },
  yokohama: {
    id: 'yokohama',
    name: 'Neo Yokohama',
    sub: 'Wangan Midnight Expressway',
    sky: { top: '#050711', mid: '#0d1527', bot: '#182b4a' },
    fog: '#0c1628',
    roadDark: '#12141c',
    roadLight: '#181b26',
    grassDark: '#08090d',
    grassLight: '#0d0e14',
    rumbleDark: '#00c3ff',
    rumbleLight: '#ff0077',
    laneColor: '#ffea00',
    propSet: ['street_lamp', 'tunnel_arch', 'neon_sign_1', 'neon_sign_2', 'cyber_billboard']
  },
  coast: {
    id: 'coast',
    name: 'Pacific Highway',
    sub: 'Coastal Sunset Speed Run',
    sky: { top: '#1e3868', mid: '#b2574e', bot: '#f79f53' },
    fog: '#b2574e',
    roadDark: '#363942',
    roadLight: '#41454f',
    grassDark: '#2b5066',
    grassLight: '#325e78',
    rumbleDark: '#ff9d00',
    rumbleLight: '#ffffff',
    laneColor: '#ffffff',
    propSet: ['palm_tree', 'ocean_rock', 'beach_sign', 'billboard_ridge']
  }
};

export class RoadTrack {
  constructor(themeId = 'touge') {
    this.theme = TRACK_THEMES[themeId] || TRACK_THEMES.touge;
    this.segments = [];
    this.totalLength = 0;
    this.checkpoints = [];
    this.buildTrack();
  }

  setTheme(themeId) {
    if (TRACK_THEMES[themeId]) {
      this.theme = TRACK_THEMES[themeId];
      this.refreshSegmentColors();
      this.populateScenery();
    }
  }

  refreshSegmentColors() {
    for (let i = 0; i < this.segments.length; i++) {
      const seg = this.segments[i];
      const isAlt = Math.floor(i / ROAD_CONSTANTS.RUMBLE_LENGTH) % 2 === 0;
      seg.colors = {
        road: isAlt ? this.theme.roadLight : this.theme.roadDark,
        grass: isAlt ? this.theme.grassLight : this.theme.grassDark,
        rumble: isAlt ? this.theme.rumbleLight : this.theme.rumbleDark,
        lane: isAlt ? this.theme.laneColor : null
      };
    }
  }

  buildTrack() {
    this.segments = [];
    this.checkpoints = [];

    // Construct circuit segments with varying curves and hills
    this.addStraight(50);
    this.addCurve(40, 2.5, 200);   // Right curve with uphill
    this.addHill(40, -400);        // Crest & drop
    this.addCurve(60, -3.0, -200); // Sharp left sweeper downhill
    this.addStraight(40);
    this.addCheckpoint();          // Checkpoint 1

    this.addCurve(50, 4.0, 300);   // Tight hairpin right
    this.addHill(30, 400);
    this.addCurve(50, -3.5, 0);    // Counter curve
    this.addStraight(60, true);    // Straight tunnel section
    this.addCurve(45, 2.0, -100);
    this.addCheckpoint();          // Checkpoint 2

    this.addHill(50, 600);         // Big mountain ascent
    this.addCurve(70, -4.5, -400); // Sudden blind crest into hairpin left
    this.addStraight(40);
    this.addCurve(50, 3.5, 0);
    this.addHill(60, -600);        // High speed downhill dive
    this.addStraight(60);
    this.addCheckpoint();          // Checkpoint 3 / Finish loop

    this.totalLength = this.segments.length * ROAD_CONSTANTS.SEGMENT_LENGTH;
    this.refreshSegmentColors();
    this.populateScenery();
  }

  addSegment(curve, y, isTunnel = false) {
    const n = this.segments.length;
    const lastY = n > 0 ? this.segments[n - 1].p2.world.y : 0;
    
    this.segments.push({
      index: n,
      p1: {
        world: { x: 0, y: lastY, z: n * ROAD_CONSTANTS.SEGMENT_LENGTH },
        camera: { x: 0, y: 0, z: 0 },
        screen: { x: 0, y: 0, w: 0, scale: 0 }
      },
      p2: {
        world: { x: 0, y: y, z: (n + 1) * ROAD_CONSTANTS.SEGMENT_LENGTH },
        camera: { x: 0, y: 0, z: 0 },
        screen: { x: 0, y: 0, w: 0, scale: 0 }
      },
      curve: curve,
      sprites: [],
      cars: [],
      isTunnel: isTunnel,
      isCheckpoint: false
    });
  }

  addStraight(num, isTunnel = false) {
    const lastY = this.segments.length > 0 ? this.segments[this.segments.length - 1].p2.world.y : 0;
    for (let i = 0; i < num; i++) {
      this.addSegment(0, lastY, isTunnel);
    }
  }

  addCurve(num, curve, endYDelta = 0) {
    const lastY = this.segments.length > 0 ? this.segments[this.segments.length - 1].p2.world.y : 0;
    for (let i = 0; i < num; i++) {
      // Ease curve in and out with sine
      const factor = Math.sin((i / num) * Math.PI);
      const y = lastY + (endYDelta * (i / num));
      this.addSegment(curve * factor, y);
    }
  }

  addHill(num, heightDelta) {
    const lastY = this.segments.length > 0 ? this.segments[this.segments.length - 1].p2.world.y : 0;
    for (let i = 0; i < num; i++) {
      const ease = (1 - Math.cos((i / num) * Math.PI)) / 2;
      const y = lastY + heightDelta * ease;
      this.addSegment(0, y);
    }
  }

  addCheckpoint() {
    if (this.segments.length > 0) {
      const seg = this.segments[this.segments.length - 1];
      seg.isCheckpoint = true;
      this.checkpoints.push(seg.index);
    }
  }

  populateScenery() {
    const props = this.theme.propSet;
    for (let i = 0; i < this.segments.length; i++) {
      const seg = this.segments[i];
      seg.sprites = [];
      
      // Don't place roadside trees inside tunnels
      if (seg.isTunnel) {
        if (i % 8 === 0) {
          seg.sprites.push({ type: 'tunnel_arch', offset: 0 });
        }
        continue;
      }

      if (i >= 2) {
        // Left & right roadside scenery
        if (i % 4 === 0) {
          const propLeft = props[Math.floor(Math.random() * props.length)];
          const isWide = propLeft.includes('billboard') || propLeft.includes('sign');
          seg.sprites.push({ type: propLeft, offset: -(isWide ? 1.75 + Math.random() * 0.4 : 1.35 + Math.random() * 0.4) });
        }

        if (i % 5 === 0) {
          const propRight = props[Math.floor(Math.random() * props.length)];
          const isWide = propRight.includes('billboard') || propRight.includes('sign');
          seg.sprites.push({ type: propRight, offset: (isWide ? 1.75 + Math.random() * 0.4 : 1.35 + Math.random() * 0.4) });
        }
      }

      // Checkpoint arch banner
      if (seg.isCheckpoint) {
        seg.sprites.push({ type: 'checkpoint_arch', offset: 0 });
      }
    }
  }

  findSegment(z) {
    return this.segments[Math.floor(z / ROAD_CONSTANTS.SEGMENT_LENGTH) % this.segments.length];
  }
}
