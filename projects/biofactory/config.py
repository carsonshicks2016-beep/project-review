from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowConfig:
    width: int = 1280
    height: int = 820
    fps: int = 60
    title: str = "BioFactory: Neural Ant Logistics Simulator"


@dataclass(frozen=True)
class WorldConfig:
    width: int = 180
    height: int = 120
    seed: int = 1337
    nest_x: float = 90.0
    nest_y: float = 60.0
    storage_radius: float = 7.5
    leaf_patch_count: int = 24
    leaf_patch_min_radius: int = 4
    leaf_patch_max_radius: int = 11
    leaf_patch_amount: float = 10.0


@dataclass(frozen=True)
class AntConfig:
    count: int = 500
    base_speed: float = 0.46
    max_turn: float = 0.42
    pickup_amount: float = 1.0
    inventory_capacity: float = 1.0
    resource_scent_radius: int = 4
    pheromone_sensor_distance: float = 5.0
    spawn_radius: float = 6.0
    energy_drain: float = 0.0018
    storage_recharge: float = 0.035


@dataclass(frozen=True)
class ColonyConfig:
    desired_leaf_storage: float = 1800.0
    desired_seed_storage: float = 160.0
    desired_food_storage: float = 420.0
    desired_water_storage: float = 620.0
    desired_protein_storage: float = 320.0
    desired_wood_fiber_storage: float = 180.0
    desired_mineral_storage: float = 140.0
    desired_soil_storage: float = 220.0
    desired_resin_storage: float = 120.0
    desired_dead_insect_storage: float = 150.0
    desired_dead_ant_storage: float = 70.0
    desired_waste_storage: float = 180.0
    desired_fungus_storage: float = 260.0
    desired_fungus_substrate_storage: float = 0.0
    desired_nutrient_paste_storage: float = 0.0
    desired_protein_paste_storage: float = 180.0
    desired_larvae_food_storage: float = 0.0
    desired_reinforced_soil_storage: float = 0.0
    desired_structural_resin_storage: float = 0.0
    desired_compost_storage: float = 160.0
    desired_fertilizer_storage: float = 0.0
    desired_soldier_feed_storage: float = 0.0
    leaf_processing_rate: float = 0.075
    food_consumption_per_ant: float = 0.00004
    water_consumption_per_ant: float = 0.000028
    protein_consumption_per_ant: float = 0.000014
    waste_generation_per_ant: float = 0.000018
    base_demand_emission: float = 10.0
    queen_signal_radius: float = 75.0
    processor_base_work_per_tick: float = 0.45
    processor_worker_bonus_per_ant: float = 0.08
    processor_max_worker_bonus: float = 0.45
    processor_worker_radius: float = 6.0
    processor_input_capacity_multiplier: float = 4.0
    processor_output_capacity_multiplier: float = 6.0
    fungus_farm_base_growth_per_tick: float = 0.035
    fungus_farm_worker_bonus_per_ant: float = 0.006
    fungus_farm_max_worker_bonus: float = 0.045
    fungus_farm_worker_radius: float = 6.0
    fungus_leaf_per_fungus: float = 0.60
    fungus_water_per_fungus: float = 0.22
    fungus_compost_per_fungus: float = 0.08
    fungus_compost_growth_boost: float = 0.55
    fungus_farm_leaf_buffer: float = 18.0
    fungus_farm_water_buffer: float = 9.0
    fungus_farm_compost_buffer: float = 6.0
    fungus_farm_output_capacity: float = 20.0
    max_population: int = 650
    queen_egg_interval_ticks: int = 140
    queen_food_per_egg: float = 0.18
    queen_water_per_egg: float = 0.06
    queen_min_health_to_lay: float = 0.35
    nursery_capacity: int = 80
    nursery_food_buffer: float = 24.0
    nursery_water_buffer: float = 10.0
    egg_to_larva_ticks: int = 420
    larva_to_pupa_ticks: int = 720
    pupa_to_worker_ticks: int = 840
    nursery_base_efficiency: float = 0.55
    nursery_worker_bonus_per_ant: float = 0.035
    nursery_max_worker_bonus: float = 0.45
    nursery_worker_radius: float = 6.0
    nursery_food_per_larva_tick: float = 0.00055
    nursery_water_per_larva_tick: float = 0.00024
    nursery_pupa_consumption_multiplier: float = 0.30
    nursery_starvation_grace_ticks: int = 900
    nursery_starvation_loss_per_tick: float = 0.002


@dataclass(frozen=True)
class RenderConfig:
    cell_size: int = 6
    art_cell_size: int = 10
    default_visual_mode: str = "hybrid"
    default_render_quality: str = "balanced"
    max_glow_points: int = 7000
    max_detailed_ants: int = 1500
    close_ant_zoom: float = 1.7
    mid_ant_zoom: float = 0.85
    trail_pulse_speed: float = 0.018
    min_zoom: float = 0.45
    max_zoom: float = 4.0
    ui_width: int = 310


@dataclass(frozen=True)
class SimulationConfig:
    window: WindowConfig = WindowConfig()
    world: WorldConfig = WorldConfig()
    ants: AntConfig = AntConfig()
    colony: ColonyConfig = ColonyConfig()
    render: RenderConfig = RenderConfig()
    speed_options: tuple[int, ...] = (1, 2, 5, 20, 100)


DEFAULT_CONFIG = SimulationConfig()
