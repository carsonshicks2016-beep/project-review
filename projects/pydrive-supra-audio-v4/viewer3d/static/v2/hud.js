// DOM wiring for the HUD. Pure view: app.js owns all state.
const $ = (s) => document.querySelector(s);

export function createHud() {
  const el = {
    conn: $("#conn"),
    trackName: $("#track-name"),
    cameraMode: $("#camera-mode"),
    fps: $("#fps"),
    air: $("#air"),
    brain: $("#brain"),
    speed: $("#speed"),
    rpm: $("#rpm"),
    gear: $("#gear"),
    progress: $("#progress"),
    throttle: $("#throttle"),
    brake: $("#brake"),
    steer: $("#steer"),
    car: $("#car"),
    track: $("#track"),
    restart: $("#restart"),
    daynight: $("#daynight"),
    toast: $("#toast"),
  };

  function setConn(label, cls = "") {
    el.conn.textContent = label;
    el.conn.className = `pill ${cls}`;
  }

  function toast(msg) {
    el.toast.textContent = msg;
    el.toast.classList.add("show");
    clearTimeout(el.toast._t);
    el.toast._t = setTimeout(() => el.toast.classList.remove("show"), 2200);
  }

  function update(msg) {
    el.speed.textContent = Math.round(msg.vehicle.speedKmh);
    el.rpm.textContent = Math.round(msg.vehicle.rpm);
    el.gear.textContent = msg.vehicle.gear;
    el.progress.textContent = `${Math.round(msg.track.progress * 100)}%`;
    el.throttle.style.width = `${Math.round(msg.controls.throttle * 100)}%`;
    el.brake.style.width = `${Math.round(msg.controls.brake * 100)}%`;
    const steer = Math.max(-1, Math.min(1, msg.controls.steer));
    el.steer.style.width = `${Math.abs(steer) * 50}%`;
    el.steer.style.marginLeft = steer >= 0 ? "50%" : `${50 - Math.abs(steer) * 50}%`;
    el.trackName.className = msg.track.offTrack ? "pill warn" : "pill";

    // brain pill: who's driving + how trained (refreshes on live hot-reload)
    if (msg.agent) {
      el.brain.style.display = "";
      el.brain.textContent =
        `🧠 ${msg.agent.name} · ${msg.agent.mode} · ${msg.agent.updates}u`;
    } else {
      el.brain.style.display = "none";
    }

    // AIR pill: live airtime while flying; landing-g flash on touchdown
    const v = msg.vehicle;
    if (v.airborne) {
      clearTimeout(el.air._t);
      el.air._t = 0;
      el.air.style.display = "";
      el.air.textContent = `AIR ${(v.airTime || 0).toFixed(1)}s`;
      el.air._wasAir = true;
    } else if (el.air._wasAir) {
      el.air._wasAir = false;
      if ((v.landingG || 0) > 0.5) {
        el.air.textContent = `▼ ${v.landingG.toFixed(1)}g`;
        el.air._t = setTimeout(() => {
          el.air.style.display = "none";
        }, 1400);
      } else {
        el.air.style.display = "none";
      }
    }
  }

  function setFps(v) {
    el.fps.textContent = `${v} fps`;
  }

  return { el, setConn, toast, update, setFps };
}
