import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repoRoot = path.resolve(root, "..");
const publicAssets = path.join(root, "public", "assets", "observatory");

async function json(relative) {
  return JSON.parse(await readFile(path.join(publicAssets, relative), "utf8"));
}

async function digest(relative) {
  return createHash("sha256").update(await readFile(path.join(publicAssets, relative))).digest("hex");
}

test("world manifest separates exact truth from visual-only dressing", async () => {
  const manifest = await json("world/world.manifest.json");
  const data = await json("world/world_data.json");
  assert.equal(manifest.schema, "supra-observatory-world-v1");
  assert.equal(manifest.track.sample_count, 6944);
  assert.ok(manifest.track.truth_road_vertex_max_error_m <= 0.02);
  assert.equal(data.track.centerline.length, 6944);
  assert.equal(data.track.heading.length, 6944);
  assert.equal(data.track.width_m.length, 6944);
  assert.equal(data.track.bank_rad.length, 6944);
  assert.equal(manifest.layers.visual.simulation_authority, "none");
  assert.equal(manifest.curbs.visual_only, true);
  assert.equal(manifest.curbs.collision_authority, false);
  assert.equal(manifest.curbs.sample_count, 6944);
  assert.equal(manifest.curbs.nodes.length, 4);
  assert.deepEqual(manifest.layers.truth.nodes, ["truth_road", "truth_shoulder_left", "truth_shoulder_right"]);
  for (const record of Object.values(manifest.files)) {
    assert.equal(await digest(`world/${record.path}`), record.sha256);
  }
  assert.equal(manifest.terrain.network_required, false);
  assert.equal(manifest.terrain.collision_authority, false);
  assert.equal(manifest.terrain.surface_stack.terrain_cutout_from_road_edge_m, 18);
  assert.equal(manifest.terrain.surface_stack.seam_skirt_depth_m, 15);
  assert.equal(manifest.terrain.surface_stack.containment_shell_depth_m, 14);
  assert.equal(manifest.terrain.surface_stack.containment_shell_padding_m, 1800);
  assert.deepEqual(manifest.terrain.surface_stack.lod_transition, {
    near_m: 820, far_m: 980, mode: "hysteretic_single_surface",
  });
  assert.equal(data.terrain.surface_stack.terrain_cutout_from_road_edge_m, 18);
  assert.equal(data.terrain.surface_stack.shoulder_width_m, 2.75);
  assert.equal(data.terrain.surface_stack.containment_shell_depth_m, 14);
  assert.ok(manifest.layers.visual.nodes.includes("terrain containment shell"));
  assert.ok(data.trackside.reflectors.length > 100);
  assert.ok(data.trackside.guardrail_posts.length > 500);
  assert.ok(data.trackside.drainage_markers.length > 500);
  assert.ok(data.trackside.marshal_stations.length >= 10);
  assert.ok(data.forest.some((entry) => entry.tier === "hero"));
  assert.ok(data.forest.some((entry) => entry.tier === "mid"));
  assert.ok(data.forest.some((entry) => entry.tier === "horizon"));
  assert.ok(data.forest.every((entry) => Number.isInteger(entry.species) && entry.species >= 0 && entry.species <= 2));
  for (const record of [...data.forest.slice(0, 40), ...data.trackside.reflectors.slice(0, 40)]) {
    assert.equal(record.position.length, 3);
    assert.equal(record.normal.length, 3);
    assert.ok(record.position.every(Number.isFinite));
    assert.ok(record.normal.every(Number.isFinite));
  }
  assert.ok(manifest.terrain.coordinate_alignment_max_error_m <= 0.15);
  assert.equal(manifest.terrain.inputs.length, 21);
  assert.deepEqual(manifest.terrain.levels.map((level) => [level.level, level.sample_stride_m, level.files.length]), [
    [0, 25, 21], [1, 100, 21],
  ]);
  for (const source of manifest.terrain.inputs) {
    const payload = await readFile(path.join(repoRoot, source.path));
    assert.equal(createHash("sha256").update(payload).digest("hex"), source.sha256);
  }
  for (const [relative, source] of Object.entries(manifest.sources)) {
    const payload = await readFile(path.join(repoRoot, relative));
    assert.equal(createHash("sha256").update(payload).digest("hex"), source.sha256);
  }
  for (const level of manifest.terrain.levels) {
    for (const record of level.files) {
      assert.equal(await digest(`world/${record.path}`), record.sha256);
      assert.ok(record.triangle_count > 0);
    }
  }
});

test("clean-room Mazda package stays under geometry and texture budgets", async () => {
  const manifest = await json("vehicles/mazda787b/asset_manifest.json");
  const provenance = await json("vehicles/mazda787b/provenance.json");
  assert.equal(manifest.vehicle_id, "mazda787b");
  assert.equal(manifest.asset_id, "mazda787b_renown_cleanroom_v2");
  assert.equal(manifest.classification, "project-authored-visual-reconstruction");
  assert.equal(manifest.runtime_detail_kit, false);
  assert.ok(manifest.budgets.triangle_count >= 24000);
  assert.ok(manifest.budgets.triangle_count <= 80000);
  assert.ok(Math.max(...manifest.budgets.texture_dimensions_px) <= 2048);
  assert.deepEqual(manifest.budgets.texture_dimensions_px, [2048, 2048]);
  assert.deepEqual(manifest.named_nodes.wheels, ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]);
  assert.equal(manifest.dimensions_m.length, 4.782);
  assert.equal(manifest.dimensions_m.width, 1.994);
  assert.equal(manifest.dimensions_m.height, 1.003);
  assert.equal(manifest.dimensions_m.wheelbase, 2.662);
  assert.equal(manifest.visual_ground_contact.schema, "supra-observatory-ground-contact-v1");
  assert.ok(manifest.visual_ground_contact.ground_anchor_offset_m >= 0);
  assert.ok(manifest.visual_ground_contact.contact_tolerance_m <= 0.02);
  assert.deepEqual(Object.keys(manifest.visual_ground_contact.wheel_nodes).sort(), ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]);
  assert.equal(provenance.exclusions.prior_viewer_geometry_or_texture_used, false);
  assert.equal(provenance.exclusions.official_or_third_party_media_embedded, false);
  for (const record of Object.values(manifest.files)) {
    assert.equal(await digest(`vehicles/mazda787b/${record.path}`), record.sha256);
  }
});

test("919 browser package points only at the project-authored source model", async () => {
  const manifest = await json("vehicles/porsche_919evo/asset_manifest.json");
  const source = await json("vehicles/porsche_919evo/source_asset_manifest.json");
  assert.equal(manifest.vehicle_id, "porsche_919evo");
  assert.equal(manifest.asset_id, "porsche_919evo_cleanroom_v2");
  assert.equal(manifest.classification, "project-authored-visual-reconstruction");
  assert.equal(manifest.runtime_detail_kit, false);
  assert.equal(manifest.source_package, "assets/vehicles/porsche_919evo");
  assert.equal(await digest(`vehicles/porsche_919evo/${manifest.entry_glb}`), manifest.files.glb.sha256);
  assert.equal(await digest(`vehicles/porsche_919evo/${manifest.files.texture.path}`), manifest.files.texture.sha256);
  assert.ok(manifest.budgets.triangle_count >= 24000);
  assert.ok(manifest.budgets.triangle_count <= 80000);
  assert.equal(manifest.dimensions_m.length, 5.078);
  assert.equal(manifest.dimensions_m.base_body_length, 4.65);
  assert.equal(manifest.dimensions_m.width, 1.9);
  assert.equal(manifest.dimensions_m.height, 1.05);
  assert.equal(manifest.dimensions_m.wheelbase, 2.957);
  assert.equal(source.faithful_geometry_eligible, false);
  assert.equal(source.provenance.official_media_embedded, false);
  assert.equal(source.provenance.prior_observatory_geometry_used, false);
  assert.equal(manifest.visual_ground_contact.schema, "supra-observatory-ground-contact-v1");
  assert.ok(manifest.visual_ground_contact.ground_anchor_offset_m >= 0);
  assert.ok(manifest.visual_ground_contact.contact_tolerance_m <= 0.02);
  assert.deepEqual(Object.keys(manifest.visual_ground_contact.wheel_nodes).sort(), ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"]);
});
