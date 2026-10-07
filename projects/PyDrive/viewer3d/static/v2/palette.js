// v2 single source of truth: every tunable color/intensity lives here, per mood.
// Day/night (and future dusk) are data, not code — materials.js lerps between
// these configs. Palette anchors sampled from the two inspiration frames
// (see VIEWER3D_V2_PLAN.md §1).

export const V2 = {
  fov: 60,
  moodFade: 1.5, // seconds for the day/night cross-fade
};

export const MOODS = {
  night: {
    skyTop: 0x0b1220,
    skyHorizon: 0x2a3a50,
    skyBottom: 0x10161f,
    skyMix: 0,               // sky shader: 0 = night extras, 1 = day extras
    exposure: 1.0,

    // NOTE: built-in materials in this three build encode linear->sRGB at
    // output but read hex as linear, so dark values must be authored ~gamma
    // darker than they should appear. 0x070a10 renders as deep blue-gray.
    fog: 0x070a10,
    fogDensity: 0.0026,

    hemiSky: 0x33415a,
    hemiGround: 0x10151c,
    hemiIntensity: 0.72,
    sunColor: 0xbfc8de,        // moonlight
    sunIntensity: 0.70,
    sunPos: [-120, 140, 60],   // offset from the car, not absolute
    ambient: 0x2a3040,
    ambientIntensity: 0.45,

    terrainTint: 0x97a2ba,   // multiplies day-authored facet colors
    shoulder: 0x4a4337,
    road: 0x8d97a8,          // tints the asphalt texture
    laneLine: 0xd9b13b,
    edgeLine: 0xa9ada7,

    guardrail: 0x59626a,
    lampPole: 0x23282e,
    lampHead: 0xffc46a,
    lampGlowOpacity: 0.9,
    lampPoolOpacity: 0.5,
    lampIntensity: 14,
    delineator: 0xb9bdc0,
    chevron: 0xd0c068,       // retroreflective at night
    wall: 0x4a5260,          // masonry under moonlight
    wallSnow: 0x55606e,

    pineTint: 0x3d4654,      // silhouette pines, faintly moonlit
    boulder: 0x262c34,

    mist: 0x6e7f93,
    mistOpacity: 0.36,

    mountainTint: 0x0a0d16,  // near-silhouette: barely lighter than the sky
    ridgeMid: 0x202c42,
    ridgeFar: 0x26344e,      // closest to the night sky horizon

    carTint: 0xb4bccc,       // dims the voxel body's baked colors
    tailLight: 0xff2230,
    headLight: 0xffe9b8,
    headPoolOpacity: 0.34,
    tailGlowOpacity: 0.80,
    contactShadow: 0.50,
    tailEmissive: 1.6,       // glb hero car: taillights glow at night
    headEmissive: 2.2,       // headlight lenses glow at night
    headlightPower: 320,     // SpotLight intensity — real light on the road
    wheelCap: 0x9aa0a8,
  },

  day: {
    skyTop: 0x4a7fc4,
    skyHorizon: 0xbdd2e2,
    skyBottom: 0x9db4c6,
    skyMix: 1,
    exposure: 1.06,

    fog: 0xb9cad8,
    fogDensity: 0.0012,

    hemiSky: 0xcfe2f2,
    hemiGround: 0x6a7a62,
    hemiIntensity: 0.85,
    sunColor: 0xfff2d8,
    sunIntensity: 1.85,
    sunPos: [110, 160, -70],
    ambient: 0x9aa8b5,
    ambientIntensity: 0.35,

    terrainTint: 0xffffff,
    shoulder: 0x6a5f4a,
    road: 0xf0f1f2,
    laneLine: 0xe8c43c,
    edgeLine: 0xe8eae4,

    guardrail: 0xc6cdd1,
    lampPole: 0x787f85,
    lampHead: 0xa6adb2,      // unlit housing by day
    lampGlowOpacity: 0.0,
    lampPoolOpacity: 0.0,
    lampIntensity: 0,
    delineator: 0xffffff,
    chevron: 0xffffff,
    wall: 0xffffff,
    wallSnow: 0xf2f7fa,

    pineTint: 0xffffff,
    boulder: 0x76776f,

    mist: 0xdde7ee,
    mistOpacity: 0.16,

    mountainTint: 0xffffff,
    ridgeMid: 0x8fa6bd,
    ridgeFar: 0xa7bdd1,      // lightest — nearly sky

    carTint: 0xffffff,
    tailLight: 0xb01018,     // lit but not glowing by day
    headLight: 0x8e9498,     // off: gray housing
    headPoolOpacity: 0.0,
    tailGlowOpacity: 0.22,
    contactShadow: 0.36,
    tailEmissive: 0.25,      // subtle by day
    headEmissive: 0.12,      // off (gray lenses) by day
    headlightPower: 0.0,     // headlights off in daylight
    wheelCap: 0xc9ccd0,
  },
};
