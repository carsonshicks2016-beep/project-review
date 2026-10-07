import math
from pathlib import Path

import bmesh
import bpy


FRAME_START = 1
FRAME_END = 240
FPS = 24
RESOLUTION = (1920, 1080)
OUTPUT_NAME = "frutiger_aero_loop.mp4"


def loop_phase(frame: int, phase: float = 0.0) -> float:
    span = max(1, FRAME_END - FRAME_START)
    return ((frame - FRAME_START) / span) * (math.tau) + phase


def smooth_mesh(obj):
    if obj.type != "MESH":
        return
    for poly in obj.data.polygons:
        poly.use_smooth = True


def set_socket(node, names, value):
    if isinstance(names, str):
        names = [names]
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            socket.default_value = value
            return socket
    return None


def add_material(obj, material):
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)


def move_to_collection(obj, collection):
    if collection not in obj.users_collection:
        collection.objects.link(obj)
    for user_collection in list(obj.users_collection):
        if user_collection != collection:
            user_collection.objects.unlink(obj)


def iter_action_fcurves(action):
    if action is None:
        return

    if hasattr(action, "fcurves"):
        for fcurve in action.fcurves:
            yield fcurve
        return

    for layer in getattr(action, "layers", []):
        for strip in getattr(layer, "strips", []):
            for channelbag in getattr(strip, "channelbags", []):
                for fcurve in getattr(channelbag, "fcurves", []):
                    yield fcurve


def keyframe_array(target, data_path, index, frame_values, interpolation="BEZIER"):
    for frame, value in frame_values:
        vector = getattr(target, data_path)
        vector[index] = value
        target.keyframe_insert(data_path=data_path, index=index, frame=frame)

    action = target.animation_data.action if target.animation_data else None
    if not action:
        return

    for fcurve in iter_action_fcurves(action):
        if fcurve.data_path == data_path and fcurve.array_index == index:
            for key in fcurve.keyframe_points:
                key.interpolation = interpolation


def keyframe_scalar(target, data_path, frame_values, interpolation="BEZIER"):
    for frame, value in frame_values:
        setattr(target, data_path, value)
        target.keyframe_insert(data_path=data_path, frame=frame)

    action = target.animation_data.action if target.animation_data else None
    if not action:
        return

    for fcurve in iter_action_fcurves(action):
        if fcurve.data_path == data_path:
            for key in fcurve.keyframe_points:
                key.interpolation = interpolation


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    data_blocks = (
        bpy.data.meshes,
        bpy.data.materials,
        bpy.data.textures,
        bpy.data.images,
        bpy.data.curves,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.actions,
        bpy.data.collections,
    )
    for block_list in data_blocks:
        for block in list(block_list):
            if block.users == 0:
                block_list.remove(block)


def make_collection(name):
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def configure_scene():
    scene = bpy.context.scene
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END
    scene.render.fps = FPS
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    scene.render.resolution_percentage = 100
    scene.render.use_file_extension = True

    scene.cycles.samples = 256
    scene.cycles.preview_samples = 64
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.max_bounces = 10
    scene.cycles.transparent_max_bounces = 10
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.blur_glossy = 0.35
    scene.cycles.sample_clamp_indirect = 2.0

    try:
        scene.cycles.device = "GPU"
    except Exception:
        pass

    view_layer = bpy.context.view_layer
    if hasattr(view_layer, "cycles") and hasattr(view_layer.cycles, "use_denoising"):
        view_layer.cycles.use_denoising = True

    if hasattr(scene.view_settings, "view_transform"):
        transforms = [
            item.identifier
            for item in bpy.context.scene.view_settings.bl_rna.properties["view_transform"].enum_items
        ]
        scene.view_settings.view_transform = "AgX" if "AgX" in transforms else "Filmic"

    if hasattr(scene.view_settings, "look"):
        available_looks = [
            item.identifier
            for item in bpy.context.scene.view_settings.bl_rna.properties["look"].enum_items
        ]
        if "Medium High Contrast" in available_looks:
            scene.view_settings.look = "Medium High Contrast"
        elif "High Contrast" in available_looks:
            scene.view_settings.look = "High Contrast"

    output_base = None
    if bpy.data.filepath:
        output_base = Path(bpy.data.filepath).with_name("frutiger_aero_loop")

    try:
        if output_base:
            scene.render.filepath = str(output_base.with_suffix(".mp4"))
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "HIGH"
        scene.render.ffmpeg.ffmpeg_preset = "GOOD"
        scene.render.ffmpeg.audio_codec = "NONE"
    except TypeError:
        if output_base:
            output_base.parent.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(output_base) + "_"
        scene.render.image_settings.file_format = "PNG"

    world = bpy.data.worlds.new("FrutigerWorld")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    bg = nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.94, 0.985, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 0.08

    output = nodes.new("ShaderNodeOutputWorld")
    links.new(bg.outputs["Background"], output.inputs["Surface"])
    scene.world = world

    scene.render.use_compositing = False
    if hasattr(scene, "node_tree") and scene.node_tree is not None:
        scene.use_nodes = True
        comp_nodes = scene.node_tree.nodes
        comp_links = scene.node_tree.links
        comp_nodes.clear()

        render_layers = comp_nodes.new("CompositorNodeRLayers")
        glare = comp_nodes.new("CompositorNodeGlare")
        glare.glare_type = "FOG_GLOW"
        glare.threshold = 0.55
        glare.size = 7
        glare.quality = "HIGH"

        hue_sat = comp_nodes.new("CompositorNodeHueSat")
        hue_sat.color_saturation = 1.06
        hue_sat.color_value = 1.02

        lens = comp_nodes.new("CompositorNodeLensdist")
        lens.use_fit = True
        lens.inputs["Distort"].default_value = 0.01
        lens.inputs["Dispersion"].default_value = 0.012

        composite = comp_nodes.new("CompositorNodeComposite")
        viewer = comp_nodes.new("CompositorNodeViewer")

        render_layers.location = (-500, 0)
        glare.location = (-260, 0)
        hue_sat.location = (0, 0)
        lens.location = (220, 0)
        composite.location = (460, 70)
        viewer.location = (460, -90)

        comp_links.new(render_layers.outputs["Image"], glare.inputs["Image"])
        comp_links.new(glare.outputs["Image"], hue_sat.inputs["Image"])
        comp_links.new(hue_sat.outputs["Image"], lens.inputs["Image"])
        comp_links.new(lens.outputs["Image"], composite.inputs["Image"])
        comp_links.new(lens.outputs["Image"], viewer.inputs["Image"])
        scene.render.use_compositing = True


def build_stage_material():
    material = bpy.data.materials.new("PearlStage")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    bump = nodes.new("ShaderNodeBump")
    mix = nodes.new("ShaderNodeMixRGB")

    noise.inputs["Scale"].default_value = 18.0
    noise.inputs["Detail"].default_value = 8.0
    noise.inputs["Roughness"].default_value = 0.55

    mix.blend_type = "SOFT_LIGHT"
    mix.inputs["Fac"].default_value = 0.22
    mix.inputs["Color1"].default_value = (0.96, 0.985, 0.965, 1.0)
    mix.inputs["Color2"].default_value = (0.88, 0.96, 0.9, 1.0)

    set_socket(bsdf, "Base Color", (0.94, 0.985, 0.95, 1.0))
    set_socket(bsdf, "Roughness", 0.18)
    set_socket(bsdf, ["Coat Weight", "Clearcoat"], 0.65)
    set_socket(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.08)

    bump.inputs["Strength"].default_value = 0.02
    bump.inputs["Distance"].default_value = 0.12

    links.new(noise.outputs["Fac"], mix.inputs["Fac"])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_glass_material(name, base_color, absorption_color, density, roughness, ior, edge_glow=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    fresnel = nodes.new("ShaderNodeFresnel")
    ramp = nodes.new("ShaderNodeValToRGB")
    emission = nodes.new("ShaderNodeEmission")
    mix_shader = nodes.new("ShaderNodeMixShader")
    absorption = nodes.new("ShaderNodeVolumeAbsorption")
    noise = nodes.new("ShaderNodeTexNoise")
    bump = nodes.new("ShaderNodeBump")

    set_socket(bsdf, "Base Color", base_color)
    set_socket(bsdf, ["Transmission Weight", "Transmission"], 1.0)
    set_socket(bsdf, "Roughness", roughness)
    set_socket(bsdf, "IOR", ior)
    set_socket(bsdf, ["Coat Weight", "Clearcoat"], 0.2)
    set_socket(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.03)

    fresnel.inputs["IOR"].default_value = 1.08
    ramp.color_ramp.elements[0].position = 0.5
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    emission.inputs["Color"].default_value = (
        min(base_color[0] + 0.08, 1.0),
        min(base_color[1] + 0.1, 1.0),
        min(base_color[2] + 0.12, 1.0),
        1.0,
    )
    emission.inputs["Strength"].default_value = edge_glow

    absorption.inputs["Color"].default_value = absorption_color
    absorption.inputs["Density"].default_value = density

    noise.inputs["Scale"].default_value = 11.0
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.38
    bump.inputs["Strength"].default_value = 0.016

    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(fresnel.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix_shader.inputs["Fac"])
    links.new(bsdf.outputs["BSDF"], mix_shader.inputs[1])
    links.new(emission.outputs["Emission"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
    links.new(absorption.outputs["Volume"], output.inputs["Volume"])
    return material


def build_water_material():
    material = bpy.data.materials.new("LivingWater")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    noise = nodes.new("ShaderNodeTexNoise")
    wave = nodes.new("ShaderNodeTexWave")
    wave_b = nodes.new("ShaderNodeTexWave")
    mix_rgb = nodes.new("ShaderNodeMixRGB")
    bump = nodes.new("ShaderNodeBump")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    fresnel = nodes.new("ShaderNodeFresnel")
    ramp = nodes.new("ShaderNodeValToRGB")
    mix_color = nodes.new("ShaderNodeMixRGB")

    wave.wave_type = "RINGS"
    wave.rings_direction = "SPHERICAL"
    wave.inputs["Scale"].default_value = 1.3
    wave.inputs["Distortion"].default_value = 2.1
    wave.inputs["Detail"].default_value = 4.0
    wave.inputs["Detail Scale"].default_value = 1.4

    wave_b.wave_type = "BANDS"
    wave_b.bands_direction = "DIAGONAL"
    wave_b.inputs["Scale"].default_value = 6.0
    wave_b.inputs["Distortion"].default_value = 3.0
    wave_b.inputs["Detail"].default_value = 5.0

    noise.inputs["Scale"].default_value = 9.5
    noise.inputs["Detail"].default_value = 12.0
    noise.inputs["Roughness"].default_value = 0.52

    mix_rgb.blend_type = "SCREEN"
    mix_rgb.inputs["Fac"].default_value = 0.45

    set_socket(bsdf, "Base Color", (0.77, 0.95, 0.92, 1.0))
    set_socket(bsdf, ["Transmission Weight", "Transmission"], 1.0)
    set_socket(bsdf, "Roughness", 0.018)
    set_socket(bsdf, "IOR", 1.333)
    set_socket(bsdf, ["Coat Weight", "Clearcoat"], 0.2)

    fresnel.inputs["IOR"].default_value = 1.1
    ramp.color_ramp.elements[0].position = 0.42
    ramp.color_ramp.elements[1].position = 0.96

    mix_color.blend_type = "SCREEN"
    mix_color.inputs["Fac"].default_value = 0.22
    mix_color.inputs["Color1"].default_value = (0.71, 0.92, 0.9, 1.0)
    mix_color.inputs["Color2"].default_value = (0.95, 1.0, 0.99, 1.0)

    bump.inputs["Strength"].default_value = 0.09
    bump.inputs["Distance"].default_value = 0.14

    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave_b.inputs["Vector"])
    links.new(wave.outputs["Color"], mix_rgb.inputs["Color1"])
    links.new(wave_b.outputs["Color"], mix_rgb.inputs["Color2"])
    links.new(mix_rgb.outputs["Color"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(fresnel.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix_color.inputs["Fac"])
    links.new(mix_color.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    rot_socket = mapping.inputs["Rotation"]
    loc_socket = mapping.inputs["Location"]
    for frame in (FRAME_START, 80, 160, FRAME_END):
        rot_socket.default_value[2] = 0.12 * math.sin(loop_phase(frame, 0.25))
        rot_socket.keyframe_insert(data_path="default_value", index=2, frame=frame)
        loc_socket.default_value[0] = 0.12 * math.sin(loop_phase(frame, 0.65))
        loc_socket.default_value[1] = 0.14 * math.cos(loop_phase(frame, 1.1))
        loc_socket.keyframe_insert(data_path="default_value", index=0, frame=frame)
        loc_socket.keyframe_insert(data_path="default_value", index=1, frame=frame)

    return material


def build_leaf_material():
    material = bpy.data.materials.new("AeroLeaf")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeValToRGB")
    noise = nodes.new("ShaderNodeTexNoise")
    mix_color = nodes.new("ShaderNodeMixRGB")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    translucent = nodes.new("ShaderNodeBsdfTranslucent")
    mix_shader = nodes.new("ShaderNodeMixShader")
    bump = nodes.new("ShaderNodeBump")

    ramp.color_ramp.elements[0].position = 0.1
    ramp.color_ramp.elements[0].color = (0.08, 0.32, 0.16, 1.0)
    ramp.color_ramp.elements[1].position = 0.88
    ramp.color_ramp.elements[1].color = (0.76, 0.95, 0.42, 1.0)
    ramp.color_ramp.elements.new(0.48)
    ramp.color_ramp.elements[1].color = (0.24, 0.62, 0.18, 1.0)

    noise.inputs["Scale"].default_value = 11.0
    noise.inputs["Detail"].default_value = 6.0
    noise.inputs["Roughness"].default_value = 0.45

    mix_color.blend_type = "OVERLAY"
    mix_color.inputs["Fac"].default_value = 0.18
    mix_color.inputs["Color2"].default_value = (0.95, 1.0, 0.86, 1.0)

    set_socket(bsdf, "Roughness", 0.32)
    set_socket(bsdf, ["Subsurface Weight", "Subsurface"], 0.08)
    set_socket(bsdf, "Subsurface Radius", (0.8, 0.35, 0.18))
    set_socket(bsdf, ["Transmission Weight", "Transmission"], 0.08)
    set_socket(bsdf, ["Coat Weight", "Clearcoat"], 0.25)

    translucent.inputs["Color"].default_value = (0.5, 0.84, 0.35, 1.0)
    bump.inputs["Strength"].default_value = 0.045
    bump.inputs["Distance"].default_value = 0.08
    mix_shader.inputs["Fac"].default_value = 0.16

    links.new(tex_coord.outputs["Generated"], separate.inputs["Vector"])
    links.new(separate.outputs["Y"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix_color.inputs["Color1"])
    links.new(noise.outputs["Fac"], mix_color.inputs["Fac"])
    links.new(mix_color.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], mix_shader.inputs[1])
    links.new(translucent.outputs["BSDF"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
    return material


def build_moss_material():
    material = bpy.data.materials.new("MossIsland")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    noise = nodes.new("ShaderNodeTexNoise")
    coarse_noise = nodes.new("ShaderNodeTexNoise")
    ramp = nodes.new("ShaderNodeValToRGB")
    mix_color = nodes.new("ShaderNodeMixRGB")
    bump = nodes.new("ShaderNodeBump")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")

    noise.inputs["Scale"].default_value = 5.0
    noise.inputs["Detail"].default_value = 9.0
    noise.inputs["Roughness"].default_value = 0.68

    coarse_noise.inputs["Scale"].default_value = 18.0
    coarse_noise.inputs["Detail"].default_value = 3.0
    coarse_noise.inputs["Roughness"].default_value = 0.72

    ramp.color_ramp.elements[0].position = 0.22
    ramp.color_ramp.elements[0].color = (0.05, 0.18, 0.09, 1.0)
    ramp.color_ramp.elements[1].position = 0.95
    ramp.color_ramp.elements[1].color = (0.42, 0.58, 0.24, 1.0)

    mix_color.blend_type = "MULTIPLY"
    mix_color.inputs["Fac"].default_value = 0.35

    set_socket(bsdf, "Roughness", 0.92)
    set_socket(bsdf, ["Subsurface Weight", "Subsurface"], 0.03)

    bump.inputs["Strength"].default_value = 0.18
    bump.inputs["Distance"].default_value = 0.2

    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix_color.inputs["Color1"])
    links.new(coarse_noise.outputs["Fac"], mix_color.inputs["Color2"])
    links.new(mix_color.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(coarse_noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_backdrop_material():
    material = bpy.data.materials.new("SkyGlow")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeValToRGB")
    noise = nodes.new("ShaderNodeTexNoise")
    cloud_ramp = nodes.new("ShaderNodeValToRGB")
    mix_color = nodes.new("ShaderNodeMixRGB")
    emission = nodes.new("ShaderNodeEmission")

    noise.inputs["Scale"].default_value = 3.4
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.44

    ramp.color_ramp.elements[0].position = 0.1
    ramp.color_ramp.elements[0].color = (0.95, 0.98, 0.9, 1.0)
    ramp.color_ramp.elements[1].position = 0.86
    ramp.color_ramp.elements[1].color = (0.54, 0.83, 0.91, 1.0)
    ramp.color_ramp.elements.new(0.45)
    ramp.color_ramp.elements[1].color = (0.7, 0.93, 0.87, 1.0)

    cloud_ramp.color_ramp.elements[0].position = 0.45
    cloud_ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    cloud_ramp.color_ramp.elements[1].position = 0.7
    cloud_ramp.color_ramp.elements[1].color = (0.35, 0.35, 0.35, 1.0)

    mix_color.blend_type = "SCREEN"
    mix_color.inputs["Fac"].default_value = 0.38
    emission.inputs["Strength"].default_value = 0.95

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], separate.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(separate.outputs["Z"], ramp.inputs["Fac"])
    links.new(noise.outputs["Fac"], cloud_ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix_color.inputs["Color1"])
    links.new(cloud_ramp.outputs["Color"], mix_color.inputs["Color2"])
    links.new(mix_color.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def build_caustic_material():
    material = bpy.data.materials.new("CausticGlow")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    tex_coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    wave = nodes.new("ShaderNodeTexWave")
    noise = nodes.new("ShaderNodeTexNoise")
    mix_rgb = nodes.new("ShaderNodeMixRGB")
    ramp = nodes.new("ShaderNodeValToRGB")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    emission = nodes.new("ShaderNodeEmission")
    mix_shader = nodes.new("ShaderNodeMixShader")

    wave.wave_type = "RINGS"
    wave.rings_direction = "SPHERICAL"
    wave.inputs["Scale"].default_value = 7.5
    wave.inputs["Distortion"].default_value = 5.5
    wave.inputs["Detail"].default_value = 5.0

    noise.inputs["Scale"].default_value = 15.0
    noise.inputs["Detail"].default_value = 5.0

    mix_rgb.blend_type = "MULTIPLY"
    mix_rgb.inputs["Fac"].default_value = 0.5

    ramp.color_ramp.elements[0].position = 0.52
    ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
    ramp.color_ramp.elements[1].position = 0.78
    ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)

    transparent.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Color"].default_value = (0.82, 1.0, 0.97, 1.0)
    emission.inputs["Strength"].default_value = 5.5

    links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], wave.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(wave.outputs["Color"], mix_rgb.inputs["Color1"])
    links.new(noise.outputs["Color"], mix_rgb.inputs["Color2"])
    links.new(mix_rgb.outputs["Color"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix_shader.inputs["Fac"])
    links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
    links.new(emission.outputs["Emission"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])

    rot_socket = mapping.inputs["Rotation"]
    for frame in (FRAME_START, 60, 120, 180, FRAME_END):
        rot_socket.default_value[2] = loop_phase(frame, 0.0)
        rot_socket.keyframe_insert(data_path="default_value", index=2, frame=frame)
    return material


def create_stage(collection, material):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=5.8, depth=0.45, location=(0.0, 0.0, -0.9))
    stage = bpy.context.active_object
    move_to_collection(stage, collection)
    smooth_mesh(stage)

    bevel = stage.modifiers.new("Bevel", "BEVEL")
    bevel.width = 0.14
    bevel.segments = 8
    subsurf = stage.modifiers.new("Subdivision", "SUBSURF")
    subsurf.levels = 2
    subsurf.render_levels = 2
    stage.scale = (1.0, 1.0, 0.36)
    add_material(stage, material)
    return stage


def create_backdrop(collection, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=22.0, location=(0.0, 0.0, 3.5))
    backdrop = bpy.context.active_object
    move_to_collection(backdrop, collection)
    smooth_mesh(backdrop)
    add_material(backdrop, material)

    keyframe_array(
        backdrop,
        "rotation_euler",
        2,
        [
            (FRAME_START, math.radians(-6.0)),
            (120, math.radians(4.0)),
            (FRAME_END, math.radians(-6.0)),
        ],
    )
    return backdrop


def create_bowl(collection, material):
    mesh = bpy.data.meshes.new("GlassBowl")
    bowl = bpy.data.objects.new("GlassBowl", mesh)
    collection.objects.link(bowl)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=96, v_segments=48, radius=4.5)
    for vert in bm.verts:
        vert.co.z *= 0.24
    top_verts = [vert for vert in bm.verts if vert.co.z > 0.18]
    bmesh.ops.delete(bm, geom=top_verts, context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

    bowl.location = (0.0, 0.0, -0.18)
    smooth_mesh(bowl)

    solidify = bowl.modifiers.new("Solidify", "SOLIDIFY")
    solidify.thickness = 0.11
    solidify.offset = -1.0
    solidify.use_even_offset = True

    bevel = bowl.modifiers.new("RimBevel", "BEVEL")
    bevel.width = 0.03
    bevel.segments = 5

    subsurf = bowl.modifiers.new("Subdivision", "SUBSURF")
    subsurf.levels = 2
    subsurf.render_levels = 3

    add_material(bowl, material)
    return bowl


def create_caustic_disk(collection, material):
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=3.55, fill_type="NGON", location=(0.0, 0.0, -0.87))
    disk = bpy.context.active_object
    move_to_collection(disk, collection)
    smooth_mesh(disk)
    add_material(disk, material)
    return disk


def create_moss_island(collection, material):
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1.2, location=(0.0, 0.0, -0.57))
    island = bpy.context.active_object
    move_to_collection(island, collection)
    island.scale = (1.0, 1.0, 0.34)
    smooth_mesh(island)
    add_material(island, material)

    for idx, angle in enumerate((0.4, 2.2, 4.4)):
        bpy.ops.mesh.primitive_ico_sphere_add(
            subdivisions=2,
            radius=0.28 + idx * 0.06,
            location=(math.cos(angle) * 1.15, math.sin(angle) * 0.95, -0.78 + idx * 0.03),
        )
        pebble = bpy.context.active_object
        move_to_collection(pebble, collection)
        pebble.scale = (1.1, 0.9, 0.4)
        smooth_mesh(pebble)
        add_material(pebble, material)

    return island


def create_water_surface(collection, material):
    bpy.ops.mesh.primitive_circle_add(vertices=96, radius=3.9, fill_type="NGON", location=(0.0, 0.0, 0.02))
    water = bpy.context.active_object
    move_to_collection(water, collection)

    mesh = water.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=5, use_grid_fill=True)
    bm.to_mesh(mesh)
    bm.free()
    smooth_mesh(water)

    water.shape_key_add(name="Basis", from_mix=False)
    broad = water.shape_key_add(name="BroadRipple", from_mix=False)
    petal = water.shape_key_add(name="PetalRipple", from_mix=False)

    for block in (broad, petal):
        block.slider_min = 0.0
        block.slider_max = 1.0

    for index, vert in enumerate(broad.data):
        base = water.data.vertices[index].co
        radius = math.hypot(base.x, base.y) / 3.9
        falloff = max(0.0, 1.0 - radius ** 1.5)
        wave = math.sin(radius * math.pi * 2.6)
        vert.co.z += wave * 0.08 * falloff

    for index, vert in enumerate(petal.data):
        base = water.data.vertices[index].co
        radius = math.hypot(base.x, base.y) / 3.9
        angle = math.atan2(base.y, base.x)
        falloff = max(0.0, 1.0 - radius ** 1.8)
        pattern = math.cos(angle * 4.0) * math.sin(radius * math.pi * 1.5 + 0.35)
        vert.co.z += pattern * 0.05 * falloff

    subsurf = water.modifiers.new("Subdivision", "SUBSURF")
    subsurf.levels = 2
    subsurf.render_levels = 3

    add_material(water, material)

    for frame in (FRAME_START, 60, 120, 180, FRAME_END):
        broad.value = 0.45 + 0.35 * math.sin(loop_phase(frame, 0.0))
        broad.keyframe_insert(data_path="value", frame=frame)
        petal.value = 0.45 + 0.35 * math.sin(loop_phase(frame, math.pi / 2))
        petal.keyframe_insert(data_path="value", frame=frame)

    return water


def create_center_orb(collection, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=1.08, location=(0.0, 0.0, 1.96))
    orb = bpy.context.active_object
    move_to_collection(orb, collection)
    smooth_mesh(orb)
    add_material(orb, material)

    keyframe_array(
        orb,
        "location",
        2,
        [
            (FRAME_START, 1.96),
            (60, 2.14),
            (120, 1.99),
            (180, 2.08),
            (FRAME_END, 1.96),
        ],
    )
    keyframe_array(
        orb,
        "rotation_euler",
        2,
        [
            (FRAME_START, 0.0),
            (FRAME_END, math.tau),
        ],
        interpolation="LINEAR",
    )
    return orb


def create_ribbon(collection, material):
    bpy.ops.mesh.primitive_torus_add(
        major_segments=96,
        minor_segments=24,
        major_radius=1.9,
        minor_radius=0.1,
        location=(0.0, 0.0, 1.78),
        rotation=(math.radians(70.0), math.radians(10.0), math.radians(18.0)),
    )
    ribbon = bpy.context.active_object
    move_to_collection(ribbon, collection)
    ribbon.scale = (1.02, 0.84, 0.56)
    smooth_mesh(ribbon)
    add_material(ribbon, material)

    keyframe_array(
        ribbon,
        "rotation_euler",
        2,
        [
            (FRAME_START, math.radians(18.0)),
            (FRAME_END, math.radians(378.0)),
        ],
        interpolation="LINEAR",
    )
    keyframe_array(
        ribbon,
        "rotation_euler",
        0,
        [
            (FRAME_START, math.radians(68.0)),
            (120, math.radians(74.0)),
            (FRAME_END, math.radians(68.0)),
        ],
    )
    return ribbon


def build_leaf_mesh(name):
    mesh = bpy.data.meshes.new(name)
    leaf = bpy.data.objects.new(name, mesh)

    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=14, y_segments=36, size=0.5)

    leaf_length = 2.7
    max_width = 0.72

    for vert in bm.verts:
        u = vert.co.x / 0.5
        t = (vert.co.y / 0.5 + 1.0) * 0.5
        half_width = max_width * (math.sin(t * math.pi) ** 0.82) + 0.015
        ridge = 1.0 - min(abs(u), 1.0)

        vert.co.x = u * half_width + 0.08 * math.sin(t * math.pi * 0.9) * ridge
        vert.co.y = t * leaf_length
        vert.co.z = 0.1 * math.sin(t * math.pi * 0.7) * ridge + 0.08 * ridge

    bm.to_mesh(mesh)
    bm.free()
    smooth_mesh(leaf)

    subsurf = leaf.modifiers.new("Subdivision", "SUBSURF")
    subsurf.levels = 2
    subsurf.render_levels = 3

    solidify = leaf.modifiers.new("Solidify", "SOLIDIFY")
    solidify.thickness = 0.028
    solidify.offset = 0.0
    return leaf


def create_leaf_cluster(collection, material):
    leaves = []
    count = 7
    for idx in range(count):
        leaf = build_leaf_mesh(f"Leaf_{idx + 1}")
        collection.objects.link(leaf)
        angle = idx * (math.tau / count) + (0.08 if idx % 2 else -0.06)
        radius = 1.35 + (0.35 if idx % 2 else 0.12)
        leaf.location = (
            math.cos(angle) * radius,
            math.sin(angle) * radius,
            0.18 + 0.04 * math.sin(idx * 1.3),
        )
        leaf.rotation_euler = (
            math.radians(58.0 + idx * 2.5),
            math.radians(-8.0 if idx % 2 else 10.0),
            angle + math.radians(84.0),
        )
        leaf.scale = (1.0 + 0.08 * math.sin(idx), 1.0 + 0.12 * math.cos(idx * 1.7), 1.0)
        add_material(leaf, material)

        frames = (FRAME_START, 60, 120, 180, FRAME_END)
        for frame in frames:
            phase = loop_phase(frame, idx * 0.7)
            leaf.rotation_euler[0] = math.radians(58.0 + idx * 2.5) + math.radians(6.0) * math.sin(phase)
            leaf.rotation_euler[1] = math.radians(-8.0 if idx % 2 else 10.0) + math.radians(3.0) * math.cos(phase + 0.8)
            leaf.rotation_euler[2] = angle + math.radians(84.0) + math.radians(2.5) * math.sin(phase + 1.2)
            leaf.keyframe_insert(data_path="rotation_euler", frame=frame)
        leaves.append(leaf)

    return leaves


def create_droplet_orbits(collection, material):
    settings = (
        {"radius": 2.25, "height": 1.58, "count": 6, "tilt_x": 62.0, "tilt_y": 18.0, "phase": 0.0, "direction": 1.0},
        {"radius": 2.9, "height": 1.32, "count": 5, "tilt_x": 48.0, "tilt_y": -22.0, "phase": 0.55, "direction": -1.0},
    )

    for orbit_index, config in enumerate(settings):
        orbit = bpy.data.objects.new(f"DropletOrbit_{orbit_index + 1}", None)
        orbit.empty_display_type = "SPHERE"
        orbit.location = (0.0, 0.0, config["height"])
        orbit.rotation_euler = (
            math.radians(config["tilt_x"]),
            math.radians(config["tilt_y"]),
            0.0,
        )
        collection.objects.link(orbit)

        keyframe_array(
            orbit,
            "rotation_euler",
            2,
            [
                (FRAME_START, 0.0),
                (FRAME_END, math.tau * config["direction"]),
            ],
            interpolation="LINEAR",
        )

        for idx in range(config["count"]):
            scale = 0.09 + 0.05 * ((idx + orbit_index) % 3)
            angle = config["phase"] * math.tau + idx * (math.tau / config["count"])
            z_offset = 0.18 * math.sin(idx * 1.9)

            bpy.ops.mesh.primitive_uv_sphere_add(
                segments=48,
                ring_count=24,
                radius=scale,
                location=(0.0, 0.0, 0.0),
            )
            droplet = bpy.context.active_object
            move_to_collection(droplet, collection)
            droplet.parent = orbit
            droplet.location = (
                math.cos(angle) * config["radius"],
                math.sin(angle) * config["radius"],
                z_offset,
            )
            smooth_mesh(droplet)
            add_material(droplet, material)

            for frame in (FRAME_START, 80, 160, FRAME_END):
                pulse = 1.0 + 0.08 * math.sin(loop_phase(frame, idx * 0.9))
                droplet.scale = (pulse, pulse, pulse)
                droplet.keyframe_insert(data_path="scale", frame=frame)


def create_lights(collection):
    def make_area(name, location, rotation, color, energy, size):
        light_data = bpy.data.lights.new(name, type="AREA")
        light_data.color = color
        light_data.energy = energy
        light_data.shape = "RECTANGLE"
        light_data.size = size
        light_data.size_y = size * 0.65
        light = bpy.data.objects.new(name, light_data)
        light.location = location
        light.rotation_euler = rotation
        collection.objects.link(light)
        return light

    make_area(
        "KeyLight",
        (6.2, -5.8, 6.5),
        (math.radians(56.0), 0.0, math.radians(42.0)),
        (0.86, 0.98, 0.93),
        4200,
        7.5,
    )
    make_area(
        "RimLight",
        (-6.5, 4.2, 5.8),
        (math.radians(58.0), 0.0, math.radians(-135.0)),
        (0.68, 0.95, 0.9),
        3200,
        5.5,
    )
    make_area(
        "TopWash",
        (0.0, 0.0, 9.2),
        (0.0, 0.0, 0.0),
        (0.98, 1.0, 0.98),
        1800,
        9.0,
    )

    point_data = bpy.data.lights.new("UnderGlow", type="POINT")
    point_data.color = (0.7, 1.0, 0.92)
    point_data.energy = 500
    point = bpy.data.objects.new("UnderGlow", point_data)
    point.location = (0.0, 0.0, -1.0)
    collection.objects.link(point)


def create_camera(collection):
    focus = bpy.data.objects.new("FocusTarget", None)
    focus.empty_display_type = "PLAIN_AXES"
    focus.location = (0.0, 0.0, 1.72)
    collection.objects.link(focus)

    rig = bpy.data.objects.new("CameraRig", None)
    rig.empty_display_type = "PLAIN_AXES"
    rig.location = (0.0, 0.0, 0.0)
    collection.objects.link(rig)

    camera_data = bpy.data.cameras.new("FrutigerCamera")
    camera = bpy.data.objects.new("FrutigerCamera", camera_data)
    camera.parent = rig
    camera.location = (8.0, -6.0, 4.8)
    collection.objects.link(camera)

    camera.data.lens = 58
    camera.data.dof.use_dof = True
    camera.data.dof.focus_object = focus
    camera.data.dof.aperture_fstop = 1.8

    track = camera.constraints.new(type="TRACK_TO")
    track.target = focus
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"

    scene = bpy.context.scene
    scene.camera = camera

    keyframe_array(
        rig,
        "rotation_euler",
        2,
        [
            (FRAME_START, math.radians(-8.0)),
            (120, math.radians(8.0)),
            (FRAME_END, math.radians(-8.0)),
        ],
    )
    keyframe_array(
        rig,
        "location",
        2,
        [
            (FRAME_START, 0.0),
            (80, 0.28),
            (160, 0.14),
            (FRAME_END, 0.0),
        ],
    )
    keyframe_array(
        focus,
        "location",
        2,
        [
            (FRAME_START, 1.72),
            (120, 1.84),
            (FRAME_END, 1.72),
        ],
    )


def main():
    clear_scene()
    configure_scene()
    collection = make_collection("Frutiger Aero")

    stage_material = build_stage_material()
    bowl_material = build_glass_material(
        name="AeroGlassBowl",
        base_color=(0.9, 0.99, 0.97, 1.0),
        absorption_color=(0.53, 0.9, 0.82, 1.0),
        density=0.08,
        roughness=0.03,
        ior=1.46,
        edge_glow=0.08,
    )
    orb_material = build_glass_material(
        name="AeroOrb",
        base_color=(0.82, 0.98, 0.94, 1.0),
        absorption_color=(0.45, 0.92, 0.76, 1.0),
        density=0.14,
        roughness=0.018,
        ior=1.43,
        edge_glow=0.24,
    )
    ribbon_material = build_glass_material(
        name="AeroRibbon",
        base_color=(0.63, 0.96, 0.89, 1.0),
        absorption_color=(0.35, 0.88, 0.8, 1.0),
        density=0.1,
        roughness=0.012,
        ior=1.39,
        edge_glow=0.18,
    )
    water_material = build_water_material()
    leaf_material = build_leaf_material()
    moss_material = build_moss_material()
    backdrop_material = build_backdrop_material()
    caustic_material = build_caustic_material()

    create_stage(collection, stage_material)
    create_backdrop(collection, backdrop_material)
    create_bowl(collection, bowl_material)
    create_caustic_disk(collection, caustic_material)
    create_moss_island(collection, moss_material)
    create_water_surface(collection, water_material)
    create_leaf_cluster(collection, leaf_material)
    create_center_orb(collection, orb_material)
    create_ribbon(collection, ribbon_material)
    create_droplet_orbits(collection, orb_material)
    create_lights(collection)
    create_camera(collection)

    bpy.context.scene.frame_set(FRAME_START)


if __name__ == "__main__":
    main()
