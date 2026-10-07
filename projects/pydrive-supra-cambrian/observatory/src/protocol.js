export const PROTOCOL_VERSION = "fable-observatory-v1";
export const DEFAULT_CLASSIFICATION = "UNVERIFIED PLAYBACK";

export function socketUrl(path, locationLike = globalThis.location) {
  if (!locationLike) throw new Error("A browser location is required to build a WebSocket URL");
  if (/^wss?:\/\//i.test(path)) return path;
  if (/^https?:\/\//i.test(path)) return path.replace(/^http/i, "ws");
  const protocol = locationLike.protocol === "https:" ? "wss:" : "ws:";
  const clean = path.startsWith("/") ? path : `/${path}`;
  return `${protocol}//${locationLike.host}${clean}`;
}

export function sessionIdFromLocation(locationLike = globalThis.location) {
  if (!locationLike) return "";
  const query = new URLSearchParams(locationLike.search || "");
  const fromQuery = query.get("session") || query.get("id");
  if (fromQuery) return sanitizeSessionId(fromQuery);
  const match = String(locationLike.pathname || "").match(/\/observatory\/([^/]+)\/?$/);
  return match ? sanitizeSessionId(match[1]) : "";
}

export function vehicleFromIdentity(raw = {}) {
  const car = String(normalizeIdentity(raw).car || "").trim();
  if (!car || car === "UNRESOLVED" || !/^[a-z0-9_-]+$/.test(car)) {
    throw new Error("The server did not provide a safe registered vehicle identity");
  }
  return car;
}

export function sanitizeSessionId(value) {
  return String(value || "").replace(/[^a-zA-Z0-9_.-]/g, "").slice(0, 128);
}

export function normalizeIdentity(raw = {}) {
  const envelope = raw.identity || raw;
  const identity = (envelope.checkpoint && typeof envelope.checkpoint === "object")
    ? envelope.checkpoint
    : envelope;
  const checkpoint = identity.filename || identity.checkpoint_id || identity.checkpoint_name || "NO CHECKPOINT";
  const hash = identity.policy_sha256 || identity.policy_hash || "HASH UNAVAILABLE";
  return {
    car: identity.car || identity.vehicle_id || envelope.car || "UNRESOLVED",
    checkpoint: String(checkpoint),
    policyHash: String(hash),
    classification: String(identity.classification || envelope.classification || DEFAULT_CLASSIFICATION),
    stage: identity.stage || identity.fable_stage || envelope.stage || "—",
    drivetrain: identity.drivetrain || identity.drivetrain_version || envelope.drivetrain_version || "—",
    edition: identity.edition || envelope.edition || "—",
    compatibility: identity.compatibility_class || envelope.compatibility_class || "—",
    observationLayout: identity.observation_layout || envelope.observation_layout || "—",
    warnings: Array.from(identity.warnings || envelope.warnings || [], (warning) => String(warning)),
    trackHash: envelope.track?.hash || identity.track_hash || "",
    protocol: envelope.protocol || identity.protocol || PROTOCOL_VERSION,
  };
}

function number(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function tuple(value, length, fallback = 0) {
  const source = Array.isArray(value) || ArrayBuffer.isView(value) ? value : [];
  return Array.from({ length }, (_, index) => number(source[index], fallback));
}

export function normalizeFrame(raw = {}) {
  if (raw.type === "frame" && raw.frame) raw = raw.frame;
  const vehicle = raw.vehicle || raw.veh || {};
  const pose = vehicle.pose || {};
  const dynamics = vehicle.dynamics || {};
  const powertrain = vehicle.powertrain || {};
  const geometry = vehicle.geometry || {};
  const wheels = Array.isArray(vehicle.wheels) ? vehicle.wheels : [];
  const telemetry = raw.telemetry || vehicle.telemetry || {};
  const action = raw.action || raw.controls || {};
  const brain = raw.brain || {};
  const replay = raw.replay || raw.playback || {};
  const sensor = raw.sensor_geometry || {};
  const observations = raw.observations || {};
  const fable = raw.fable || {};
  const track = raw.track || {};
  const position = tuple(
    vehicle.position || vehicle.pos || [pose.x ?? vehicle.x, pose.y ?? vehicle.y, pose.z ?? vehicle.z],
    3,
  );
  const roadZ = Number(track.road_z_m);
  const airborne = Boolean(dynamics.airborne);
  // Grounded vehicle z is kinematic in the simulator. The vehicle pose is
  // sampled at the end of the final physics substep while track telemetry is
  // sampled at the resulting x/y, leaving a harmless one-substep height phase
  // offset. In 2D it was invisible; in the 3D viewer it looks like suspension
  // hopping. Align only the rendered grounded pose to the authoritative road
  // sample. Ballistic/airborne z remains the simulator's exact value.
  if (!airborne && Number.isFinite(roadZ)) position[2] = roadZ;
  const progress = number(raw.progress ?? telemetry.progress ?? raw.track?.progress, 0);
  const outputMeans = tuple(brain.output_means, 3);
  const beamOrigin = tuple(sensor.beam_origin, 3);
  const beamPoints = Array.isArray(sensor.beam_points) ? sensor.beam_points : [];
  const sensorRays = beamPoints.map((end) => ({
    start: beamOrigin,
    end: [number(end?.[0]), number(end?.[1]), end?.length > 2 ? number(end[2]) : beamOrigin[2]],
  }));
  const predictedPath = Array.isArray(brain.predicted_path)
    ? brain.predicted_path.map((point) => [number(point.x), number(point.y), number(point.z)])
    : [];
  return {
    type: "frame",
    seq: number(raw.sequence ?? raw.seq, 0),
    simTime: number(raw.sim_time ?? raw.sim_time_s ?? raw.simTime ?? raw.t, 0),
    identity: normalizeIdentity(raw),
    vehicle: {
      position,
      yaw: number(pose.yaw ?? vehicle.yaw),
      pitch: number(pose.pitch ?? vehicle.pitch),
      roll: number(pose.roll ?? vehicle.roll),
      airborne,
      roadZ: Number.isFinite(roadZ) ? roadZ : position[2],
      steer: number(vehicle.steer ?? vehicle.steer_angle ?? telemetry.steer),
      wheelRotation: wheels.length ? wheels.map((wheel) => number(wheel.rotation_rad)).slice(0, 4) : tuple(vehicle.wheel_rotation ?? vehicle.wheelRotation, 4),
      wheelSteer: wheels.length ? wheels.map((wheel) => number(wheel.steer_rad)).slice(0, 4) : [0, 0, 0, 0],
      wheelTravel: tuple(vehicle.suspension_travel ?? vehicle.wheelTravel, 4),
      wheels: wheels.map((wheel, index) => ({
        id: String(wheel.id || ["FL", "FR", "RL", "RR"][index] || index),
        loadN: number(wheel.load_n),
        grip: number(wheel.grip),
        slipRatio: number(wheel.slip_ratio),
        slipAngle: number(wheel.slip_angle_rad),
        contact: wheel.contact !== false,
      })),
      collisionLength: number(geometry.collision_length_m),
      collisionWidth: number(geometry.collision_width_m),
      footprint: Array.isArray(geometry.footprint_world_xy)
        ? geometry.footprint_world_xy.map((point) => tuple(point, 2))
        : [],
    },
    telemetry: {
      speedMps: number(dynamics.speed_mps ?? telemetry.speed_mps ?? telemetry.speed ?? vehicle.speed),
      rpm: number(powertrain.rpm ?? telemetry.rpm ?? vehicle.rpm),
      gear: number(powertrain.gear ?? telemetry.gear ?? vehicle.gear),
      throttle: number(action.throttle ?? telemetry.throttle),
      brake: number(action.brake ?? telemetry.brake),
      steer: number(action.steer ?? telemetry.steer),
      long: number(action.long ?? action.longitudinal),
      gearOffset: number(action.gear_offset ?? action.gearOffset),
      paceRatio: number(fable.pace_ratio ?? telemetry.pace_ratio ?? telemetry.paceRatio),
      progress,
      lap: number(raw.episode ?? telemetry.lap ?? raw.lap),
      lapTime: number(raw.episode_time ?? telemetry.lap_time_s ?? telemetry.lapTime ?? raw.lap_time_s),
      delta: Number.isFinite(Number(telemetry.delta_s ?? raw.delta_s)) ? Number(telemetry.delta_s ?? raw.delta_s) : null,
      offTrack: Boolean(track.off_track ?? telemetry.off_track ?? raw.off_track),
      mguPower: Number.isFinite(Number(powertrain.mgu_power_w ?? telemetry.mgu_power ?? telemetry.mguPower)) ? Number(powertrain.mgu_power_w ?? telemetry.mgu_power ?? telemetry.mguPower) : null,
      hybridSoc: Number.isFinite(Number(powertrain.hybrid_soc ?? telemetry.hybrid_soc ?? telemetry.hybridSoc)) ? Number(powertrain.hybrid_soc ?? telemetry.hybrid_soc ?? telemetry.hybridSoc) : null,
      criticValue: Number.isFinite(Number(brain.critic_value)) ? Number(brain.critic_value) : null,
      boost: number(powertrain.boost),
      activeAero: Boolean(powertrain.active_aero_engaged),
      activeAeroLowDrag: number(powertrain.active_aero_low_drag),
      fableValid: fable.valid !== false,
      footprintValid: fable.footprint_valid !== false,
    },
    brain: {
      observations: Array.from(observations.normalized_vector || brain.observations || brain.observation || [], (entry) => number(entry)),
      hiddenLayers: Array.isArray(brain.hidden_layers)
        ? brain.hidden_layers.slice(0, 2).map((layer) => Array.from(layer || [], (entry) => number(entry)))
        : [],
      actions: outputMeans,
      policyStd: tuple(brain.policy_std, 3),
      observationGroups: Object.entries(observations.groups || {}).map(([key, group]) => ({
        key,
        name: String(group?.name || key),
        samples: Array.from(group?.values || []).slice(0, 4).map((entry) => ({
          label: String(entry.label || entry.index || "feature"),
          value: number(entry.policy_value ?? entry.value),
        })),
      })),
      sensitivity: brain.sensitivity || null,
      rays: Array.isArray(brain.rays) ? brain.rays : sensorRays,
      pacePoints: Array.isArray(brain.pace_points || brain.pacePoints) ? (brain.pace_points || brain.pacePoints) : [],
      predictedPath: predictedPath.length ? predictedPath : (Array.isArray(brain.predictedPath) ? brain.predictedPath : []),
      note: brain.note || brain.sensitivity?.label || "",
    },
    replay: {
      cursor: number(replay.cursor, 1),
      duration: number(replay.duration_s ?? replay.duration, 0),
      live: replay.live !== false && !replay.scrubbing,
      paused: Boolean(replay.paused ?? raw.paused),
      speed: number(replay.speed ?? raw.speed, 1),
      scrubbing: Boolean(replay.scrubbing),
    },
  };
}

export function encodeControl(command, value = undefined) {
  const payload = { type: command };
  if (command === "speed") payload.speed = number(value, 1);
  if (command === "scrub") payload.seconds_ago = Math.max(0, number(value));
  if (command === "listener" && value && typeof value === "object") payload.listener = value;
  return JSON.stringify(payload);
}

export function classifyMessage(raw) {
  if (raw instanceof ArrayBuffer || ArrayBuffer.isView(raw)) return { type: "binary", value: raw };
  const parsed = typeof raw === "string" ? JSON.parse(raw) : raw;
  if (!parsed || typeof parsed !== "object") throw new Error("Observatory message must be an object");
  if ((parsed.type === "frame" && parsed.frame) || parsed.vehicle || parsed.veh) return { type: "frame", value: normalizeFrame(parsed) };
  if (parsed.protocol === PROTOCOL_VERSION || ["hello", "identity", "ready"].includes(parsed.type)) {
    return { type: "identity", value: normalizeIdentity(parsed.hello || parsed.identity || parsed) };
  }
  return { type: parsed.type || "event", value: parsed };
}
