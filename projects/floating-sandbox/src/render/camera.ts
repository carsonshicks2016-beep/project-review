import type { TankBounds, Vec2 } from '../sim/types';

/** World-space camera: (x, y) is the view center; zoom is screen pixels per world unit. */
export class Camera {
  x = 0;
  y = 0;
  zoom = 20;

  /**
   * Convert viewport client coordinates (e.g. `event.clientX/Y`) to world space.
   * Also accepts canvas-local CSS pixels when the canvas is origin-aligned.
   */
  screenToWorld(sx: number, sy: number, canvas: HTMLCanvasElement): Vec2 {
    const rect = canvas.getBoundingClientRect();
    const localX = sx - rect.left;
    const localY = sy - rect.top;
    const cssW = rect.width || canvas.clientWidth || 1;
    const cssH = rect.height || canvas.clientHeight || 1;
    return {
      x: this.x + (localX - cssW * 0.5) / this.zoom,
      y: this.y + (localY - cssH * 0.5) / this.zoom,
    };
  }

  /** Convert world space to canvas-local CSS pixels. */
  worldToScreen(wx: number, wy: number, canvas: HTMLCanvasElement): Vec2 {
    const cssW = canvas.clientWidth || canvas.getBoundingClientRect().width || 1;
    const cssH = canvas.clientHeight || canvas.getBoundingClientRect().height || 1;
    return {
      x: (wx - this.x) * this.zoom + cssW * 0.5,
      y: (wy - this.y) * this.zoom + cssH * 0.5,
    };
  }

  /** Pan by screen-pixel delta (positive dx moves view right / world left). */
  pan(dx: number, dy: number): void {
    this.x -= dx / this.zoom;
    this.y -= dy / this.zoom;
  }

  /**
   * Zoom keeping the world point under the cursor fixed.
   * `sx/sy` are viewport client coordinates (same as `screenToWorld`).
   */
  zoomAt(sx: number, sy: number, factor: number, canvas: HTMLCanvasElement): void {
    const before = this.screenToWorld(sx, sy, canvas);
    this.zoom = Math.min(400, Math.max(2, this.zoom * factor));
    const after = this.screenToWorld(sx, sy, canvas);
    this.x += before.x - after.x;
    this.y += before.y - after.y;
  }

  /**
   * Fit tank bounds in view.
   * `padding` ≥ 1 is CSS pixels; otherwise a fraction of the smaller side (e.g. 0.08).
   */
  fitBounds(bounds: TankBounds, canvas: HTMLCanvasElement, padding = 40): void {
    const worldW = Math.max(1e-6, bounds.x1 - bounds.x0);
    const worldH = Math.max(1e-6, bounds.y1 - bounds.y0);
    const cssW = canvas.clientWidth || canvas.width;
    const cssH = canvas.clientHeight || canvas.height;
    const padPx = padding >= 1 ? padding : Math.min(cssW, cssH) * padding;
    const viewW = Math.max(1, cssW - padPx * 2);
    const viewH = Math.max(1, cssH - padPx * 2);
    this.zoom = Math.min(viewW / worldW, viewH / worldH);
    this.x = (bounds.x0 + bounds.x1) * 0.5;
    this.y = (bounds.y0 + bounds.y1) * 0.5;
  }
}

export type DetachCameraControls = () => void;

/**
 * Middle-mouse or Alt+drag to pan; wheel to zoom at cursor.
 * `isPanModifier` should return true when a tool wants left-drag to pan (e.g. space held).
 */
export function attachCameraControls(
  canvas: HTMLCanvasElement,
  camera: Camera,
  isPanModifier: () => boolean = () => false,
): DetachCameraControls {
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const onPointerDown = (e: PointerEvent) => {
    const pan =
      e.button === 1 ||
      (e.button === 0 && (e.altKey || isPanModifier()));
    if (!pan) return;
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  const onPointerMove = (e: PointerEvent) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    camera.pan(dx, dy);
  };

  const onPointerUp = (e: PointerEvent) => {
    if (!dragging) return;
    dragging = false;
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
      /* already released */
    }
  };

  const onWheel = (e: WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    camera.zoomAt(e.clientX, e.clientY, factor, canvas);
  };

  const onContextMenu = (e: Event) => e.preventDefault();

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointercancel', onPointerUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('contextmenu', onContextMenu);

  return () => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', onPointerUp);
    canvas.removeEventListener('pointercancel', onPointerUp);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('contextmenu', onContextMenu);
  };
}
