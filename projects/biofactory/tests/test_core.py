from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from biofactory.config import DEFAULT_CONFIG
from biofactory.colonies.role_inference import infer_role
from biofactory.logistics.demand import CRITICAL_URGENCY_BOOST, DemandSignal, calculate_resource_demand
from biofactory.nest.nest_grid import NestCellType
from biofactory.nest.brood import NurseryState
from biofactory.nest.chambers import Chamber
from biofactory.nest.farming import FungusFarmState
from biofactory.nest.processing import ProcessingState
from biofactory.nest.storage import StoragePolicy
from biofactory.neural.brain import Brain, BrainInput, BrainOutput, HIDDEN_SIZE, INPUT_SIZE, MEMORY_SIZE, OUTPUT_SIZE
from biofactory.pheromones.diffusion import diffuse_and_decay
from biofactory.pheromones.pheromone_config import PheromoneType
from biofactory.resources.inventory import Inventory
from biofactory.resources.logistics import (
    CARRIABLE_RESOURCES,
    DEMAND_TRACKED_RESOURCES,
    GATHERABLE_RESOURCES,
    PHASE2_PROCESSED_RESOURCES,
    RAW_RESOURCE_TYPES,
    PROCESSED_LOGISTICS_RESOURCES,
    RESOURCE_PHEROMONES,
)
from biofactory.resources.production_chains import (
    COMPOST_PROCESSING,
    LEAF_FALLBACK_PROCESSING,
    NUTRIENT_PROCESSING,
    PHASE2_PRODUCTION_CHAINS,
    PROTEIN_PROCESSING,
)
from biofactory.resources.resource_nodes import ResourceSourceKind
from biofactory.resources.resource_types import ResourceType
from biofactory.resources.resource_types import RESOURCE_SPECS
from biofactory.simulation.engine import SimulationEngine
from biofactory.simulation.terrain import (
    TERRAIN_PROPERTIES,
    TerrainType,
    construction_difficulty_map,
    energy_cost_map,
    resource_spawn_chance_map,
    visibility_map,
)
from biofactory.simulation.world import SurfaceCellType


class PheromoneTests(unittest.TestCase):
    def test_pheromone_decay_reduces_total_concentration(self) -> None:
        layer = np.zeros((7, 7), dtype=np.float32)
        layer[3, 3] = 100.0
        updated = diffuse_and_decay(layer, diffusion_rate=0.0, decay_rate=0.10, max_concentration=100.0)
        self.assertAlmostEqual(float(updated[3, 3]), 90.0, places=4)
        self.assertLess(float(np.sum(updated)), float(np.sum(layer)))


class PhaseOneWorldTests(unittest.TestCase):
    def test_world_exposes_required_phase_one_layers(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        world = engine.world
        shape = (world.height, world.width)
        required_maps = (
            world.terrain,
            world.obstacles,
            world.movement_speed,
            world.energy_cost,
            world.pheromone_decay_multiplier,
            world.pheromone_diffusion_multiplier,
            world.visibility,
            world.construction_difficulty,
            world.resource_spawn_chance,
            world.temperature,
            world.humidity,
            world.light,
            world.surface,
            world.underground,
        )
        for layer in required_maps:
            with self.subTest(layer=layer.dtype):
                self.assertEqual(layer.shape, shape)
        cx, cy = world.cell(DEFAULT_CONFIG.world.nest_x, DEFAULT_CONFIG.world.nest_y)
        self.assertEqual(SurfaceCellType(int(world.surface[cy, cx])), SurfaceCellType.NEST_ENTRANCE)
        self.assertEqual(NestCellType(int(world.underground[cy, cx])), NestCellType.ENTRANCE)
        self.assertTrue(np.any(world.underground == NestCellType.QUEEN_CHAMBER))

    def test_all_required_terrain_effect_maps_match_properties(self) -> None:
        terrain = np.array([[terrain_type for terrain_type in TerrainType]], dtype=np.uint8)
        energy = energy_cost_map(terrain)
        visibility = visibility_map(terrain)
        construction = construction_difficulty_map(terrain)
        spawn = resource_spawn_chance_map(terrain)
        for index, terrain_type in enumerate(TerrainType):
            props = TERRAIN_PROPERTIES[terrain_type]
            self.assertAlmostEqual(float(energy[0, index]), props.energy_cost, places=5)
            self.assertAlmostEqual(float(visibility[0, index]), props.visibility, places=5)
            self.assertAlmostEqual(float(construction[0, index]), props.construction_difficulty, places=5)
            self.assertAlmostEqual(float(spawn[0, index]), props.resource_spawn_chance, places=5)

    def test_engine_records_phase_one_loop_stages(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        engine.step()
        self.assertEqual(
            engine.last_loop_stages,
            (
                "weather",
                "seasons",
                "terrain_effects",
                "pheromones",
                "resource_generation",
                "production_chambers",
                "colonies",
                "predators",
                "ants",
                "combat",
                "logistics_metrics",
                "statistics",
                "events",
            ),
        )
        self.assertIn("population", engine.statistics)
        self.assertEqual(engine.statistics["tick"], 1.0)

    def test_terrain_energy_map_changes_ant_energy_cost(self) -> None:
        low_cost = _single_ant_engine()
        high_cost = _single_ant_engine()
        for engine, cost in ((low_cost, 0.50), (high_cost, 2.00)):
            ant = engine.ants[0]
            ant.x = 5.0
            ant.y = 5.0
            ant.energy = 1.0
            cx, cy = engine.world.cell(ant.x, ant.y)
            engine.world.obstacles[cy, cx] = False
            engine.world.energy_cost[cy, cx] = cost
            for layer in engine.resources.layers.values():
                layer.fill(0.0)
            ant.update(engine)
        self.assertLess(high_cost.ants[0].energy, low_cost.ants[0].energy)


class DemandTests(unittest.TestCase):
    def test_demand_ratio_tracks_shortage(self) -> None:
        high = DemandSignal(current=10.0, desired=100.0)
        low = DemandSignal(current=95.0, desired=100.0)
        self.assertGreater(high.ratio, low.ratio)
        self.assertAlmostEqual(high.ratio, 0.9)
        self.assertAlmostEqual(low.ratio, 0.05)

    def test_resource_priority_rises_as_storage_falls(self) -> None:
        low = calculate_resource_demand(ResourceType.WATER, stored=10.0, desired=100.0)
        high = calculate_resource_demand(ResourceType.WATER, stored=90.0, desired=100.0)
        self.assertGreater(low.priority, high.priority)

    def test_saturated_resource_has_near_zero_priority(self) -> None:
        saturated = calculate_resource_demand(ResourceType.LEAVES, stored=140.0, desired=100.0)
        self.assertTrue(saturated.is_saturated)
        self.assertLess(saturated.priority, 0.05)

    def test_food_shortage_boosts_leaf_priority(self) -> None:
        normal = calculate_resource_demand(ResourceType.LEAVES, stored=70.0, desired=100.0, food_shortage_ratio=0.0)
        hungry = calculate_resource_demand(ResourceType.LEAVES, stored=70.0, desired=100.0, food_shortage_ratio=1.0)
        self.assertGreater(hungry.priority, normal.priority)

    def test_critical_resource_receives_urgency_boost(self) -> None:
        critical = calculate_resource_demand(ResourceType.PROTEIN, stored=5.0, desired=100.0)
        baseline = critical.shortage_ratio**1.4
        self.assertTrue(critical.is_critical)
        self.assertGreaterEqual(critical.urgency, baseline + CRITICAL_URGENCY_BOOST)

    def test_overall_colony_demand_is_bounded(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: 0.0,
            ResourceType.WATER: 0.0,
            ResourceType.PROTEIN: 0.0,
            ResourceType.WASTE: 0.0,
            ResourceType.FOOD: 0.0,
        })
        self.assertGreaterEqual(engine.colony.demand_ratio(), 0.0)
        self.assertLessEqual(engine.colony.demand_ratio(), 1.0)


class InventoryTests(unittest.TestCase):
    def test_inventory_transfer_moves_available_amount(self) -> None:
        source = Inventory({ResourceType.LEAVES: 3.0})
        target = Inventory()
        moved = source.transfer_to(target, ResourceType.LEAVES, 5.0)
        self.assertEqual(moved, 3.0)
        self.assertEqual(source.get(ResourceType.LEAVES), 0.0)
        self.assertEqual(target.get(ResourceType.LEAVES), 3.0)


class StoragePolicyTests(unittest.TestCase):
    def test_storage_policy_capacity_helpers(self) -> None:
        policy = StoragePolicy(ResourceType.WATER, desired=10.0, capacity=15.0)
        self.assertAlmostEqual(policy.fullness(7.5), 0.5)
        self.assertAlmostEqual(policy.available_capacity(7.5), 7.5)
        self.assertTrue(policy.can_accept(14.0))
        self.assertFalse(policy.can_accept(15.0))
        self.assertGreater(policy.saturation(12.0), 0.0)


class ResourceMapTests(unittest.TestCase):
    def test_mvp_resource_layers_seed_multiple_streams(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        for resource_type in GATHERABLE_RESOURCES:
            with self.subTest(resource_type=resource_type):
                self.assertIn(resource_type, engine.resources.layers)
                self.assertGreater(engine.resources.total(resource_type), 0.0)

    def test_phase_two_resource_catalog_is_complete(self) -> None:
        expected_raw = {
            ResourceType.LEAVES,
            ResourceType.SEEDS,
            ResourceType.PROTEIN,
            ResourceType.WATER,
            ResourceType.WOOD_FIBER,
            ResourceType.MINERALS,
            ResourceType.SOIL,
            ResourceType.RESIN,
            ResourceType.DEAD_INSECTS,
            ResourceType.DEAD_ANTS,
            ResourceType.WASTE,
        }
        expected_processed = {
            ResourceType.FUNGUS_SUBSTRATE,
            ResourceType.FUNGUS,
            ResourceType.NUTRIENT_PASTE,
            ResourceType.PROTEIN_PASTE,
            ResourceType.LARVAE_FOOD,
            ResourceType.REINFORCED_SOIL,
            ResourceType.STRUCTURAL_RESIN,
            ResourceType.COMPOST,
            ResourceType.FERTILIZER,
            ResourceType.SOLDIER_FEED,
        }
        self.assertEqual(set(RAW_RESOURCE_TYPES), expected_raw)
        self.assertEqual(set(PHASE2_PROCESSED_RESOURCES), expected_processed)
        for resource_type in expected_raw | expected_processed | {ResourceType.FOOD}:
            spec = RESOURCE_SPECS[resource_type]
            self.assertGreater(spec.weight, 0.0)
            self.assertGreaterEqual(spec.decay_rate, 0.0)
            self.assertIsInstance(spec.storage_requirement, str)
            self.assertIsInstance(spec.processing_requirement, str)
            self.assertGreaterEqual(spec.scent_radius, 1)
            self.assertEqual(len(spec.color), 3)

    def test_phase_two_resource_maps_have_risk_and_sources(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        for resource_type in RAW_RESOURCE_TYPES:
            with self.subTest(resource_type=resource_type):
                self.assertIn(resource_type, engine.resources.layers)
                self.assertIn(resource_type, engine.resources.risk_layers)
                self.assertGreater(engine.resources.total(resource_type), 0.0)
                self.assertGreater(len(engine.resources.source_nodes_for(resource_type)), 0)
        source_kinds = {node.source_kind for node in engine.resources.source_nodes_for()}
        for source_kind in (
            ResourceSourceKind.LEAF_PATCH,
            ResourceSourceKind.SEED_PILE,
            ResourceSourceKind.DEAD_INSECT,
            ResourceSourceKind.WATER_POOL,
            ResourceSourceKind.MINERAL_DEPOSIT,
            ResourceSourceKind.FALLEN_FRUIT,
            ResourceSourceKind.TREE_ROOT,
            ResourceSourceKind.RIVAL_WASTE_DUMP,
            ResourceSourceKind.PREDATOR_CORPSE,
            ResourceSourceKind.SOIL_BANK,
            ResourceSourceKind.RESIN_PATCH,
        ):
            self.assertIn(source_kind, source_kinds)
        self.assertGreater(max(node.risk for node in engine.resources.source_nodes_for()), 0.5)

    def test_resource_decay_reduces_decaying_world_resources(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        before = engine.resources.total(ResourceType.DEAD_INSECTS)
        engine.resources.apply_decay(240)
        after = engine.resources.total(ResourceType.DEAD_INSECTS)
        self.assertLess(after, before)
        water_before = engine.resources.total(ResourceType.WATER)
        engine.resources.apply_decay(240)
        self.assertAlmostEqual(engine.resources.total(ResourceType.WATER), water_before)

    def test_phase_two_production_chain_metadata_covers_core_outputs(self) -> None:
        produced = set()
        consumed = set()
        for recipe in PHASE2_PRODUCTION_CHAINS:
            self.assertGreater(recipe.work, 0.0)
            produced.update(recipe.outputs)
            consumed.update(recipe.inputs)
        for resource_type in (
            ResourceType.FUNGUS_SUBSTRATE,
            ResourceType.FUNGUS,
            ResourceType.NUTRIENT_PASTE,
            ResourceType.LARVAE_FOOD,
            ResourceType.PROTEIN_PASTE,
            ResourceType.COMPOST,
            ResourceType.FERTILIZER,
            ResourceType.SOLDIER_FEED,
            ResourceType.REINFORCED_SOIL,
            ResourceType.STRUCTURAL_RESIN,
        ):
            self.assertIn(resource_type, produced)
        for resource_type in (
            ResourceType.LEAVES,
            ResourceType.WATER,
            ResourceType.PROTEIN,
            ResourceType.WASTE,
            ResourceType.DEAD_INSECTS,
            ResourceType.DEAD_ANTS,
            ResourceType.WOOD_FIBER,
            ResourceType.MINERALS,
            ResourceType.SOIL,
            ResourceType.RESIN,
        ):
            self.assertIn(resource_type, consumed)


class PheromoneLayerTests(unittest.TestCase):
    def test_resource_pheromone_layers_are_active(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        for pheromone_type in RESOURCE_PHEROMONES.values():
            with self.subTest(pheromone_type=pheromone_type):
                self.assertIn(pheromone_type, engine.pheromones.layers)
        self.assertIn(PheromoneType.DEMAND, engine.pheromones.layers)
        self.assertIn(PheromoneType.TRAFFIC, engine.pheromones.layers)


class PhaseThreeAntAgentTests(unittest.TestCase):
    def test_tiny_brain_matches_phase_three_contract(self) -> None:
        engine = _single_ant_engine()
        brain = Brain.seeded_worker(engine.rng)
        self.assertEqual(INPUT_SIZE, 64)
        self.assertEqual(OUTPUT_SIZE, 28)
        self.assertEqual(HIDDEN_SIZE, 24)
        self.assertEqual(brain.w1.shape, (INPUT_SIZE, HIDDEN_SIZE))
        self.assertEqual(brain.w2.shape, (HIDDEN_SIZE, HIDDEN_SIZE))
        self.assertEqual(brain.w3.shape, (HIDDEN_SIZE, OUTPUT_SIZE))
        outputs = brain.forward(np.linspace(0.0, 1.0, INPUT_SIZE, dtype=np.float32))
        self.assertEqual(outputs.shape, (OUTPUT_SIZE,))
        for output in (BrainOutput.TURN, BrainOutput.MOVE_SPEED, BrainOutput.MEMORY_0):
            self.assertGreaterEqual(float(outputs[output]), -1.0)
            self.assertLessEqual(float(outputs[output]), 1.0)
        for output in (BrainOutput.PICKUP, BrainOutput.DROP, BrainOutput.LAY_DANGER, BrainOutput.TEND_FUNGUS):
            self.assertGreaterEqual(float(outputs[output]), 0.0)
            self.assertLessEqual(float(outputs[output]), 1.0)

    def test_ant_exposes_phase_three_traits_and_sensory_snapshots(self) -> None:
        engine = _single_ant_engine()
        ant = engine.ants[0]
        engine.step()
        self.assertEqual(ant.memory.shape, (MEMORY_SIZE,))
        self.assertEqual(ant.last_inputs.shape, (INPUT_SIZE,))
        self.assertEqual(ant.last_outputs.shape, (OUTPUT_SIZE,))
        self.assertGreater(ant.inventory_capacity, 0.0)
        self.assertGreater(ant.speed, 0.0)
        self.assertGreater(ant.strength, 0.0)
        self.assertGreaterEqual(ant.last_inputs[BrainInput.QUEEN_SIGNAL], 0.0)
        self.assertGreaterEqual(ant.last_inputs[BrainInput.ALLY_DENSITY], 0.0)
        self.assertIn("resources_delivered", ant.lifetime_stats)

    def test_phase_three_pheromone_outputs_write_signal_layers(self) -> None:
        engine = _single_ant_engine()
        ant = engine.ants[0]
        outputs = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        outputs[BrainOutput.LAY_DANGER] = 1.0
        outputs[BrainOutput.LAY_RECRUITMENT] = 1.0
        outputs[BrainOutput.LAY_EMERGENCY] = 1.0
        outputs[BrainOutput.LAY_DEMAND] = 0.90
        outputs[BrainOutput.LAY_TRAFFIC] = 0.75
        ant._lay_pheromones(engine, outputs, ResourceType.LEAVES)
        self.assertGreater(engine.pheromones.sample(PheromoneType.DANGER, ant.x, ant.y), 0.0)
        self.assertGreater(engine.pheromones.sample(PheromoneType.RECRUITMENT, ant.x, ant.y), 0.0)
        self.assertGreater(engine.pheromones.sample(PheromoneType.EMERGENCY, ant.x, ant.y), 0.0)
        self.assertGreater(engine.pheromones.sample(PheromoneType.DEMAND, ant.x, ant.y), 0.0)
        self.assertGreater(engine.pheromones.sample(PheromoneType.TRAFFIC, ant.x, ant.y), 0.0)
        self.assertGreater(ant.danger_pheromone_laid, 0.0)
        self.assertGreater(ant.recruitment_pheromone_laid, 0.0)

    def test_phase_three_local_actions_update_lifetime_stats(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=2),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        ant = engine.ants[0]
        other = engine.ants[1]
        farm = _chamber_by_type(engine, "fungus_farm")
        ant.x = farm.x
        ant.y = farm.y
        other.x = farm.x
        other.y = farm.y
        farm.health = 0.85
        engine.spatial_hash.clear()
        engine.ant_density.fill(0)
        for member in engine.ants:
            engine.spatial_hash.insert(member.ant_id, member.x, member.y)
            cx, cy = engine.world.cell(member.x, member.y)
            engine.ant_density[cy, cx] += 1
        outputs = np.ones(OUTPUT_SIZE, dtype=np.float32)
        ant._apply_local_actions(engine, outputs)
        self.assertGreater(ant.digging_ticks, 0)
        self.assertGreater(ant.building_ticks, 0)
        self.assertGreater(ant.attack_intent_ticks, 0)
        self.assertGreater(ant.defend_ticks, 0)
        self.assertGreater(ant.grooming_ticks, 0)
        self.assertGreater(ant.allies_helped, 0.0)
        self.assertGreater(ant.fungus_tended, 0.0)
        self.assertGreater(farm.health, 0.85)

    def test_memory_outputs_update_ant_memory_vector(self) -> None:
        engine = _single_ant_engine()
        ant = engine.ants[0]
        outputs = np.zeros(OUTPUT_SIZE, dtype=np.float32)
        outputs[BrainOutput.MEMORY_0] = 1.0
        outputs[BrainOutput.MEMORY_1] = -1.0
        ant._update_memory(outputs)
        self.assertGreater(ant.memory[0], 0.0)
        self.assertLess(ant.memory[1], 0.0)

    def test_role_inference_covers_phase_three_roles(self) -> None:
        engine = _single_ant_engine()
        ant = engine.ants[0]
        ant.age = 200
        ant.fungus_tended = 2.0
        self.assertEqual(infer_role(ant), "Farmer")

        defender = _single_ant_engine().ants[0]
        defender.age = 200
        defender.defend_ticks = 90
        self.assertEqual(infer_role(defender), "Defender")

        courier = _single_ant_engine().ants[0]
        courier.age = 200
        courier.deliveries = 12
        courier.target_chamber_id = 1
        courier.distance_from_nest_max = 10
        self.assertEqual(infer_role(courier), "Courier")


class ChamberStorageTests(unittest.TestCase):
    def test_chambers_accept_only_configured_resource(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        leaf_chamber = engine.colony.storage_chambers_for(ResourceType.LEAVES)[0]
        self.assertTrue(leaf_chamber.accepts(ResourceType.LEAVES))
        self.assertFalse(leaf_chamber.accepts(ResourceType.WATER))

    def test_chamber_fullness_and_capacity_are_bounded(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        water_chamber = engine.colony.storage_chambers_for(ResourceType.WATER)[0]
        water_chamber.inventory.amounts.clear()
        capacity = water_chamber.capacity_for(ResourceType.WATER)
        accepted = water_chamber.add_resource(ResourceType.WATER, capacity * 2.0)
        self.assertAlmostEqual(accepted, capacity)
        self.assertAlmostEqual(water_chamber.fullness(ResourceType.WATER), 1.0)

    def test_colony_aggregate_storage_matches_chambers(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: 55.0,
            ResourceType.WATER: 22.0,
            ResourceType.PROTEIN: 9.0,
            ResourceType.WASTE: 4.0,
            ResourceType.FOOD: 12.0,
        })
        chamber_total = sum(chamber.inventory.get(ResourceType.WATER) for chamber in engine.colony.chambers)
        self.assertAlmostEqual(engine.colony.inventory.get(ResourceType.WATER), chamber_total)
        self.assertAlmostEqual(engine.colony.stored_amount(ResourceType.LEAVES), 55.0)

    def test_demand_falls_when_typed_chamber_fills(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        _set_colony_storage(engine, {ResourceType.WATER: 0.0})
        empty_priority = engine.colony.resource_priority(ResourceType.WATER)
        _set_colony_storage(engine, {ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER)})
        full_priority = engine.colony.resource_priority(ResourceType.WATER)
        self.assertGreater(empty_priority, full_priority)

    def test_saturated_leaf_storage_does_not_suppress_protein_demand(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        leaf_capacity = engine.colony.storage_capacity(ResourceType.LEAVES)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: leaf_capacity,
            ResourceType.PROTEIN: 0.0,
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
            ResourceType.WASTE: engine.colony.desired_storage_for(ResourceType.WASTE),
            ResourceType.FOOD: engine.colony.desired_storage_for(ResourceType.FOOD),
            ResourceType.FUNGUS: engine.colony.desired_storage_for(ResourceType.FUNGUS),
            ResourceType.PROTEIN_PASTE: engine.colony.desired_storage_for(ResourceType.PROTEIN_PASTE),
            ResourceType.COMPOST: engine.colony.desired_storage_for(ResourceType.COMPOST),
        })
        _fill_nursery_buffers(engine)
        self.assertLess(engine.colony.resource_priority(ResourceType.LEAVES), 0.05)
        self.assertGreater(engine.colony.resource_priority(ResourceType.PROTEIN), 0.5)


class ProcessingTests(unittest.TestCase):
    def test_nutrient_recipe_consumes_inputs_and_produces_food(self) -> None:
        chamber = _processor_chamber(NUTRIENT_PROCESSING)
        chamber.add_resource(ResourceType.FUNGUS, 1.0)
        chamber.add_resource(ResourceType.WATER, 0.30)
        completed = chamber.processing_state.advance(chamber, NUTRIENT_PROCESSING.work)
        self.assertTrue(completed)
        self.assertAlmostEqual(chamber.inventory.get(ResourceType.FUNGUS), 0.0)
        self.assertAlmostEqual(chamber.inventory.get(ResourceType.WATER), 0.0)
        self.assertAlmostEqual(chamber.inventory.get(ResourceType.FOOD), 1.05)
        self.assertEqual(chamber.processing_state.batches_completed, 1)

    def test_leaf_fallback_recipe_is_weaker_than_fungus_nutrient_recipe(self) -> None:
        chamber = _processor_chamber(LEAF_FALLBACK_PROCESSING)
        chamber.add_resource(ResourceType.LEAVES, 2.6)
        chamber.add_resource(ResourceType.WATER, 0.80)
        completed = chamber.processing_state.advance(chamber, LEAF_FALLBACK_PROCESSING.work)
        self.assertTrue(completed)
        self.assertAlmostEqual(chamber.inventory.get(ResourceType.FOOD), 0.35)
        self.assertLess(LEAF_FALLBACK_PROCESSING.outputs[ResourceType.FOOD], NUTRIENT_PROCESSING.outputs[ResourceType.FOOD])

    def test_protein_and_compost_recipes_produce_processed_outputs(self) -> None:
        protein_chamber = _processor_chamber(PROTEIN_PROCESSING)
        protein_chamber.add_resource(ResourceType.PROTEIN, 1.4)
        protein_chamber.add_resource(ResourceType.WATER, 0.25)
        self.assertTrue(protein_chamber.processing_state.advance(protein_chamber, PROTEIN_PROCESSING.work))
        self.assertAlmostEqual(protein_chamber.inventory.get(ResourceType.PROTEIN_PASTE), 1.0)

        compost_chamber = _processor_chamber(COMPOST_PROCESSING)
        compost_chamber.add_resource(ResourceType.WASTE, 1.0)
        compost_chamber.add_resource(ResourceType.LEAVES, 0.25)
        self.assertTrue(compost_chamber.processing_state.advance(compost_chamber, COMPOST_PROCESSING.work))
        self.assertAlmostEqual(compost_chamber.inventory.get(ResourceType.COMPOST), 0.9)

    def test_recipe_blocks_when_input_missing_or_output_full(self) -> None:
        missing = _processor_chamber(NUTRIENT_PROCESSING)
        missing.add_resource(ResourceType.FUNGUS, 1.0)
        self.assertFalse(missing.processing_state.advance(missing, NUTRIENT_PROCESSING.work))
        self.assertEqual(missing.processing_state.blocked_reason, "missing_inputs")

        full = _processor_chamber(PROTEIN_PROCESSING)
        full.add_resource(ResourceType.PROTEIN, 1.4)
        full.add_resource(ResourceType.WATER, 0.25)
        full.add_resource(ResourceType.PROTEIN_PASTE, full.capacity_for(ResourceType.PROTEIN_PASTE))
        self.assertFalse(full.processing_state.advance(full, PROTEIN_PROCESSING.work))
        self.assertEqual(full.processing_state.blocked_reason, "output_full")

    def test_light_tending_increases_progress_rate(self) -> None:
        untended = _processor_chamber(NUTRIENT_PROCESSING)
        tended = _processor_chamber(NUTRIENT_PROCESSING)
        for chamber in (untended, tended):
            chamber.add_resource(ResourceType.FUNGUS, 1.0)
            chamber.add_resource(ResourceType.WATER, 0.30)
        untended.process_tick(worker_count=0, config=DEFAULT_CONFIG.colony)
        tended.process_tick(worker_count=4, config=DEFAULT_CONFIG.colony)
        self.assertGreater(tended.processing_state.progress, untended.processing_state.progress)

    def test_processor_chamber_accepts_inputs_and_releases_only_outputs(self) -> None:
        chamber = _processor_chamber(NUTRIENT_PROCESSING)
        self.assertTrue(chamber.accepts(ResourceType.FUNGUS))
        self.assertFalse(chamber.accepts(ResourceType.LEAVES))
        self.assertFalse(chamber.accepts(ResourceType.FOOD))
        self.assertFalse(chamber.can_release_output(ResourceType.FOOD))
        chamber.add_resource(ResourceType.FOOD, 0.5)
        self.assertTrue(chamber.can_release_output(ResourceType.FOOD))

    def test_fungus_resource_is_internal_logistics_resource(self) -> None:
        self.assertIn(ResourceType.FUNGUS, PROCESSED_LOGISTICS_RESOURCES)
        self.assertIn(ResourceType.FUNGUS, CARRIABLE_RESOURCES)
        self.assertIn(ResourceType.FUNGUS, DEMAND_TRACKED_RESOURCES)
        self.assertNotIn(ResourceType.FUNGUS, GATHERABLE_RESOURCES)

    def test_fungus_farm_grows_with_leaves_and_water(self) -> None:
        chamber = _fungus_farm_chamber()
        chamber.add_resource(ResourceType.LEAVES, 6.0)
        chamber.add_resource(ResourceType.WATER, 3.0)
        harvested = 0.0
        for tick in range(40):
            harvested += chamber.farm_tick(worker_count=0, config=DEFAULT_CONFIG.colony, tick=tick)
        self.assertGreater(chamber.fungus_farm_state.biomass + harvested, 0.0)
        self.assertGreaterEqual(chamber.inventory.get(ResourceType.FUNGUS), harvested)

    def test_compost_boosts_fungus_growth_and_is_consumed(self) -> None:
        plain = _fungus_farm_chamber()
        boosted = _fungus_farm_chamber()
        for chamber in (plain, boosted):
            chamber.add_resource(ResourceType.LEAVES, 6.0)
            chamber.add_resource(ResourceType.WATER, 3.0)
        boosted.add_resource(ResourceType.COMPOST, 2.0)
        plain.farm_tick(worker_count=0, config=DEFAULT_CONFIG.colony, tick=1)
        boosted.farm_tick(worker_count=0, config=DEFAULT_CONFIG.colony, tick=1)
        self.assertGreater(boosted.fungus_farm_state.last_efficiency, plain.fungus_farm_state.last_efficiency)
        self.assertLess(boosted.inventory.get(ResourceType.COMPOST), 2.0)

    def test_fungus_farm_blocks_without_required_inputs_or_output_room(self) -> None:
        missing = _fungus_farm_chamber()
        missing.add_resource(ResourceType.LEAVES, 4.0)
        self.assertEqual(missing.farm_tick(worker_count=0, config=DEFAULT_CONFIG.colony, tick=1), 0.0)
        self.assertEqual(missing.fungus_farm_state.blocked_reason, "missing_inputs")

        full = _fungus_farm_chamber()
        full.add_resource(ResourceType.LEAVES, 4.0)
        full.add_resource(ResourceType.WATER, 2.0)
        full.add_resource(ResourceType.FUNGUS, full.capacity_for(ResourceType.FUNGUS))
        self.assertEqual(full.farm_tick(worker_count=0, config=DEFAULT_CONFIG.colony, tick=1), 0.0)
        self.assertEqual(full.fungus_farm_state.blocked_reason, "output_full")

    def test_fungus_farm_releases_only_fungus(self) -> None:
        chamber = _fungus_farm_chamber()
        chamber.add_resource(ResourceType.LEAVES, 2.0)
        chamber.add_resource(ResourceType.FUNGUS, 1.0)
        self.assertTrue(chamber.can_release_output(ResourceType.FUNGUS))
        self.assertFalse(chamber.can_release_output(ResourceType.LEAVES))

    def test_colony_aggregate_inventory_includes_processor_buffers(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        nutrient = _chamber_by_type(engine, "nutrient_processor")
        nutrient.add_resource(ResourceType.FOOD, 2.0)
        engine.colony.refresh_aggregate_inventory()
        self.assertGreaterEqual(engine.colony.inventory.get(ResourceType.FOOD), 2.0)


class NurseryTests(unittest.TestCase):
    def test_queen_lays_eggs_only_with_resources_health_and_capacity(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            colony=replace(DEFAULT_CONFIG.colony, queen_egg_interval_ticks=1, max_population=10),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        _set_colony_storage(engine, {
            ResourceType.FOOD: 5.0,
            ResourceType.WATER: 5.0,
            ResourceType.LEAVES: engine.colony.desired_storage_for(ResourceType.LEAVES),
        })
        engine.step()
        self.assertEqual(engine.colony.eggs, 1)

        no_health = SimulationEngine(config)
        no_health.colony.queen_health = 0.1
        _set_colony_storage(no_health, {ResourceType.FOOD: 5.0, ResourceType.WATER: 5.0})
        no_health.step()
        self.assertEqual(no_health.colony.eggs, 0)

        no_food = SimulationEngine(config)
        _set_colony_storage(no_food, {ResourceType.FOOD: 0.0, ResourceType.WATER: 5.0})
        no_food.step()
        self.assertEqual(no_food.colony.eggs, 0)

    def test_population_cap_blocks_new_eggs(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            colony=replace(DEFAULT_CONFIG.colony, queen_egg_interval_ticks=1, max_population=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        _set_colony_storage(engine, {ResourceType.FOOD: 5.0, ResourceType.WATER: 5.0})
        engine.step()
        self.assertEqual(engine.colony.eggs, 0)

    def test_brood_advances_to_worker_birth(self) -> None:
        config = replace(
            DEFAULT_CONFIG.colony,
            egg_to_larva_ticks=1,
            larva_to_pupa_ticks=1,
            pupa_to_worker_ticks=1,
            nursery_base_efficiency=1.0,
        )
        chamber = _nursery_chamber()
        chamber.brood_state.eggs = 1
        chamber.add_resource(ResourceType.FOOD, 5.0)
        chamber.add_resource(ResourceType.WATER, 5.0)
        births = chamber.nursery_tick(worker_count=0, config=config, tick=1)
        self.assertEqual(births, 1)
        self.assertEqual(chamber.brood_state.workers_born, 1)
        self.assertEqual(chamber.brood_state.total_brood, 0)

    def test_nursery_tending_increases_efficiency(self) -> None:
        untended = _nursery_chamber()
        tended = _nursery_chamber()
        for chamber in (untended, tended):
            chamber.brood_state.eggs = 2
            chamber.add_resource(ResourceType.FOOD, 5.0)
            chamber.add_resource(ResourceType.WATER, 5.0)
        untended.nursery_tick(worker_count=0, config=DEFAULT_CONFIG.colony, tick=1)
        tended.nursery_tick(worker_count=8, config=DEFAULT_CONFIG.colony, tick=1)
        self.assertGreater(tended.brood_state.last_efficiency, untended.brood_state.last_efficiency)

    def test_missing_food_or_water_blocks_nursery_development(self) -> None:
        missing_food = _nursery_chamber()
        missing_food.brood_state.larvae = 1
        missing_food.add_resource(ResourceType.WATER, 2.0)
        missing_food.nursery_tick(worker_count=0, config=DEFAULT_CONFIG.colony, tick=1)
        self.assertEqual(missing_food.brood_state.blocked_reason, "missing_food")

        missing_water = _nursery_chamber()
        missing_water.brood_state.larvae = 1
        missing_water.add_resource(ResourceType.FOOD, 2.0)
        missing_water.nursery_tick(worker_count=0, config=DEFAULT_CONFIG.colony, tick=1)
        self.assertEqual(missing_water.brood_state.blocked_reason, "missing_water")

    def test_starvation_grace_delays_loss_then_reduces_larvae_and_generates_waste(self) -> None:
        config = replace(DEFAULT_CONFIG.colony, nursery_starvation_grace_ticks=1, nursery_starvation_loss_per_tick=1.0)
        chamber = _nursery_chamber()
        chamber.brood_state.larvae = 2
        chamber.nursery_tick(worker_count=0, config=config, tick=1)
        self.assertEqual(chamber.brood_state.larvae, 2)
        chamber.nursery_tick(worker_count=0, config=config, tick=2)
        self.assertLess(chamber.brood_state.larvae, 2)
        self.assertGreater(chamber.inventory.get(ResourceType.WASTE), 0.0)

    def test_nursery_accepts_food_and_water_but_is_not_storage(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        nursery = _chamber_by_type(engine, "nursery")
        self.assertTrue(nursery.accepts(ResourceType.FOOD))
        self.assertTrue(nursery.accepts(ResourceType.WATER))
        self.assertFalse(nursery.accepts(ResourceType.LEAVES))
        self.assertTrue(nursery.is_nursery)
        self.assertFalse(nursery.is_storage)

    def test_colony_brood_counters_match_nursery_state(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        nursery = _chamber_by_type(engine, "nursery")
        nursery.brood_state.eggs = 2
        nursery.brood_state.larvae = 3
        nursery.brood_state.pupae = 4
        engine.colony._refresh_brood_counts()
        self.assertEqual(engine.colony.eggs, 2)
        self.assertEqual(engine.colony.larvae, 3)
        self.assertEqual(engine.colony.pupae, 4)


class AntResourceLoopTests(unittest.TestCase):
    def test_ant_targets_critical_water_over_saturated_leaves(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: engine.colony.desired_storage_for(ResourceType.LEAVES) * 1.25,
            ResourceType.WATER: 0.0,
            ResourceType.PROTEIN: engine.colony.desired_storage_for(ResourceType.PROTEIN),
            ResourceType.WASTE: engine.colony.desired_storage_for(ResourceType.WASTE),
            ResourceType.FOOD: engine.colony.desired_storage_for(ResourceType.FOOD),
        })
        ant = engine.ants[0]
        self.assertEqual(ant._target_resource(engine, engine.config.ants), ResourceType.WATER)

    def test_pickup_prefers_scarce_protein_over_saturated_leaves(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        for layer in engine.resources.layers.values():
            layer.fill(0.0)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: engine.colony.desired_storage_for(ResourceType.LEAVES) * 1.30,
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
            ResourceType.PROTEIN: 0.0,
            ResourceType.WASTE: engine.colony.desired_storage_for(ResourceType.WASTE),
            ResourceType.FOOD: engine.colony.desired_storage_for(ResourceType.FOOD),
        })
        ant = engine.ants[0]
        ant.x = engine.colony.nest_x + engine.colony.storage_radius + 2.0
        ant.y = engine.colony.nest_y
        engine.resources.add_at(ResourceType.LEAVES, ant.x, ant.y, 8.0)
        engine.resources.add_at(ResourceType.PROTEIN, ant.x, ant.y, 1.0)
        self.assertEqual(ant._best_pickup_resource(engine), ResourceType.PROTEIN)

    def test_source_pheromone_laying_stops_for_saturated_resource(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        for layer in engine.resources.layers.values():
            layer.fill(0.0)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: engine.colony.desired_storage_for(ResourceType.LEAVES) * 1.30,
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
            ResourceType.PROTEIN: engine.colony.desired_storage_for(ResourceType.PROTEIN),
            ResourceType.WASTE: engine.colony.desired_storage_for(ResourceType.WASTE),
            ResourceType.FOOD: engine.colony.desired_storage_for(ResourceType.FOOD),
            ResourceType.FUNGUS: engine.colony.desired_storage_for(ResourceType.FUNGUS),
            ResourceType.PROTEIN_PASTE: engine.colony.desired_storage_for(ResourceType.PROTEIN_PASTE),
            ResourceType.COMPOST: engine.colony.desired_storage_for(ResourceType.COMPOST),
        })
        _fill_nursery_buffers(engine)
        ant = engine.ants[0]
        ant.x = engine.colony.nest_x + engine.colony.storage_radius + 2.0
        ant.y = engine.colony.nest_y
        engine.resources.add_at(ResourceType.LEAVES, ant.x, ant.y, 8.0)
        before = engine.pheromones.sample(PheromoneType.FOOD, ant.x, ant.y)
        ant._lay_pheromones(engine, np.zeros(OUTPUT_SIZE, dtype=np.float32), ResourceType.LEAVES)
        after = engine.pheromones.sample(PheromoneType.FOOD, ant.x, ant.y)
        self.assertEqual(after, before)

    def test_ant_pickup_and_drop_increases_storage_for_each_raw_resource(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        for resource_type in GATHERABLE_RESOURCES:
            with self.subTest(resource_type=resource_type):
                engine = SimulationEngine(config)
                for layer in engine.resources.layers.values():
                    layer.fill(0.0)
                ant = engine.ants[0]
                ant.x = engine.colony.nest_x + engine.colony.storage_radius + 2.0
                ant.y = engine.colony.nest_y
                ant.target_resource = resource_type
                engine.resources.add_at(resource_type, ant.x, ant.y, 4.0)

                before = engine.colony.inventory.get(resource_type)
                ant._try_pickup(engine, pickup_signal=1.0, target_resource=resource_type)
                self.assertTrue(ant.has_cargo)
                self.assertEqual(ant.inventory_type, resource_type)

                chamber = engine.colony.best_drop_chamber(resource_type, ant.x, ant.y)
                self.assertIsNotNone(chamber)
                ant.x = chamber.x
                ant.y = chamber.y
                ant._try_drop(engine, drop_signal=1.0)
                after = engine.colony.inventory.get(resource_type)

                self.assertFalse(ant.has_cargo)
                self.assertGreater(after, before)

    def test_ant_carrying_water_targets_water_reservoir(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.WATER
        ant.inventory_amount = 1.0
        target = ant._drop_target(engine)
        chamber = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(chamber)
        self.assertTrue(chamber.accepts(ResourceType.WATER))
        self.assertEqual(target, (chamber.x, chamber.y))

    def test_ant_near_food_storage_picks_up_food_for_nursery(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        _set_colony_storage(engine, {
            ResourceType.FOOD: 20.0,
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
        })
        nursery = _chamber_by_type(engine, "nursery")
        nursery.brood_state.larvae = 3
        engine.colony._refresh_brood_counts()
        engine.colony._refresh_demand_cache()
        food_storage = engine.colony.storage_chambers_for(ResourceType.FOOD)[0]
        ant = engine.ants[0]
        ant.x = food_storage.x
        ant.y = food_storage.y
        self.assertTrue(ant._try_chamber_pickup(engine))
        self.assertEqual(ant.inventory_type, ResourceType.FOOD)
        target = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(target)
        self.assertEqual(target.chamber_type, "nursery")

    def test_ant_carrying_food_targets_hungry_nursery(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        nursery = _chamber_by_type(engine, "nursery")
        nursery.brood_state.larvae = 3
        engine.colony._refresh_brood_counts()
        engine.colony._refresh_demand_cache()
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.FOOD
        ant.inventory_amount = 1.0
        target = ant._drop_target(engine)
        chamber = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(chamber)
        self.assertEqual(chamber.chamber_type, "nursery")
        self.assertEqual(ant.cargo_destination_kind, "nursery")
        self.assertEqual(target, (chamber.x, chamber.y))

    def test_ant_near_water_storage_picks_up_water_for_nursery(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        _set_colony_storage(engine, {
            ResourceType.FOOD: engine.colony.desired_storage_for(ResourceType.FOOD),
            ResourceType.WATER: 20.0,
        })
        nursery = _chamber_by_type(engine, "nursery")
        nursery.brood_state.larvae = 3
        nursery.add_resource(ResourceType.FOOD, nursery.capacity_for(ResourceType.FOOD))
        for chamber in engine.colony.input_consumer_chambers():
            if chamber is not nursery and chamber.accepts(ResourceType.WATER):
                chamber.add_resource(ResourceType.WATER, chamber.capacity_for(ResourceType.WATER))
        engine.colony._refresh_brood_counts()
        engine.colony._refresh_demand_cache()
        water_storage = engine.colony.storage_chambers_for(ResourceType.WATER)[0]
        ant = engine.ants[0]
        ant.x = water_storage.x
        ant.y = water_storage.y
        self.assertTrue(ant._try_chamber_pickup(engine))
        self.assertEqual(ant.inventory_type, ResourceType.WATER)
        target = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(target)
        self.assertEqual(target.chamber_type, "nursery")

    def test_ant_drops_only_in_accepting_chamber(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.WATER
        ant.inventory_amount = 1.0
        wrong_chamber = engine.colony.storage_chambers_for(ResourceType.LEAVES)[0]
        ant.x = wrong_chamber.x
        ant.y = wrong_chamber.y
        ant._try_drop(engine, drop_signal=1.0)
        self.assertTrue(ant.has_cargo)

        water_chamber = engine.colony.storage_chambers_for(ResourceType.WATER)[0]
        ant.x = water_chamber.x
        ant.y = water_chamber.y
        ant._try_drop(engine, drop_signal=1.0)
        self.assertFalse(ant.has_cargo)

    def test_overflow_fallback_when_typed_chamber_full(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        water_capacity = engine.colony.storage_capacity(ResourceType.WATER)
        _set_colony_storage(engine, {ResourceType.WATER: water_capacity})
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.WATER
        ant.inventory_amount = 1.0
        ant.x = engine.colony.nest_x
        ant.y = engine.colony.nest_y
        before = engine.colony.overflow_inventory.get(ResourceType.WATER)
        ant._try_drop(engine, drop_signal=1.0)
        after = engine.colony.overflow_inventory.get(ResourceType.WATER)
        self.assertFalse(ant.has_cargo)
        self.assertGreater(after, before)

    def test_ant_near_leaf_storage_picks_up_for_hungry_processor(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: 25.0,
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
            ResourceType.PROTEIN: engine.colony.desired_storage_for(ResourceType.PROTEIN),
            ResourceType.WASTE: engine.colony.desired_storage_for(ResourceType.WASTE),
            ResourceType.FOOD: engine.colony.desired_storage_for(ResourceType.FOOD) * 0.25,
        })
        leaf_storage = engine.colony.storage_chambers_for(ResourceType.LEAVES)[0]
        ant = engine.ants[0]
        ant.x = leaf_storage.x
        ant.y = leaf_storage.y
        self.assertTrue(ant._try_chamber_pickup(engine))
        self.assertEqual(ant.inventory_type, ResourceType.LEAVES)
        target = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(target)
        self.assertEqual(target.chamber_type, "fungus_farm")
        self.assertEqual(ant.cargo_destination_kind, "farm")

    def test_ant_carrying_leaves_targets_hungry_processor(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: engine.colony.desired_storage_for(ResourceType.LEAVES),
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
            ResourceType.PROTEIN: engine.colony.desired_storage_for(ResourceType.PROTEIN),
            ResourceType.WASTE: engine.colony.desired_storage_for(ResourceType.WASTE),
            ResourceType.FOOD: 0.0,
        })
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.LEAVES
        ant.inventory_amount = 1.0
        target = ant._drop_target(engine)
        chamber = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(chamber)
        self.assertEqual(chamber.chamber_type, "fungus_farm")
        self.assertEqual(ant.cargo_destination_kind, "farm")
        self.assertEqual(target, (chamber.x, chamber.y))

    def test_ant_near_processor_output_picks_up_food_for_storage(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        nutrient = _chamber_by_type(engine, "nutrient_processor")
        nutrient.add_resource(ResourceType.FOOD, 2.0)
        ant = engine.ants[0]
        ant.x = nutrient.x
        ant.y = nutrient.y
        self.assertTrue(ant._try_chamber_pickup(engine))
        self.assertEqual(ant.inventory_type, ResourceType.FOOD)
        self.assertEqual(ant.cargo_source, "processor")
        target = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(target)
        self.assertEqual(target.chamber_type, "food_storage")

    def test_ant_carrying_protein_paste_targets_paste_storage(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.PROTEIN_PASTE
        ant.inventory_amount = 1.0
        target = ant._drop_target(engine)
        chamber = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(chamber)
        self.assertEqual(chamber.chamber_type, "protein_paste_storage")
        self.assertEqual(target, (chamber.x, chamber.y))

    def test_ant_near_fungus_farm_output_picks_up_fungus_for_storage(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        farm = _chamber_by_type(engine, "fungus_farm")
        farm.add_resource(ResourceType.FUNGUS, 2.0)
        ant = engine.ants[0]
        ant.x = farm.x
        ant.y = farm.y
        self.assertTrue(ant._try_chamber_pickup(engine))
        self.assertEqual(ant.inventory_type, ResourceType.FUNGUS)
        self.assertEqual(ant.cargo_source, "farm")
        target = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(target)
        self.assertEqual(target.chamber_type, "fungus_storage")

    def test_ant_carrying_fungus_targets_fungus_storage(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.FUNGUS
        ant.inventory_amount = 1.0
        target = ant._drop_target(engine)
        chamber = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(chamber)
        self.assertEqual(chamber.chamber_type, "fungus_storage")
        self.assertEqual(target, (chamber.x, chamber.y))

    def test_ant_near_fungus_storage_picks_up_for_nutrient_processor(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: engine.colony.desired_storage_for(ResourceType.LEAVES),
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
            ResourceType.PROTEIN: engine.colony.desired_storage_for(ResourceType.PROTEIN),
            ResourceType.WASTE: engine.colony.desired_storage_for(ResourceType.WASTE),
            ResourceType.FOOD: 0.0,
            ResourceType.FUNGUS: 10.0,
        })
        fungus_storage = engine.colony.storage_chambers_for(ResourceType.FUNGUS)[0]
        ant = engine.ants[0]
        ant.x = fungus_storage.x
        ant.y = fungus_storage.y
        self.assertTrue(ant._try_chamber_pickup(engine))
        self.assertEqual(ant.inventory_type, ResourceType.FUNGUS)
        target = engine.colony.chamber_by_id(ant.target_chamber_id)
        self.assertIsNotNone(target)
        self.assertEqual(target.chamber_type, "nutrient_processor")

    def test_ant_carrying_fungus_preserves_processor_target(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        nutrient = _chamber_by_type(engine, "nutrient_processor")
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.FUNGUS
        ant.inventory_amount = 1.0
        ant.target_chamber_id = nutrient.chamber_id
        target = ant._drop_target(engine)
        self.assertEqual(target, (nutrient.x, nutrient.y))
        self.assertEqual(ant.target_chamber_id, nutrient.chamber_id)

    def test_fungus_farm_output_not_picked_up_when_storage_full(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        fungus_storage = engine.colony.storage_chambers_for(ResourceType.FUNGUS)[0]
        fungus_storage.add_resource(ResourceType.FUNGUS, fungus_storage.capacity_for(ResourceType.FUNGUS))
        farm = _chamber_by_type(engine, "fungus_farm")
        farm.add_resource(ResourceType.FUNGUS, 2.0)
        ant = engine.ants[0]
        ant.x = farm.x
        ant.y = farm.y
        self.assertFalse(ant._try_chamber_pickup(engine))
        self.assertFalse(ant.has_cargo)

    def test_processor_output_not_picked_up_when_matching_storage_full(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        paste_storage = engine.colony.storage_chambers_for(ResourceType.PROTEIN_PASTE)[0]
        paste_storage.add_resource(ResourceType.PROTEIN_PASTE, paste_storage.capacity_for(ResourceType.PROTEIN_PASTE))
        protein = _chamber_by_type(engine, "protein_processor")
        protein.add_resource(ResourceType.PROTEIN_PASTE, 2.0)
        ant = engine.ants[0]
        ant.x = protein.x
        ant.y = protein.y
        self.assertFalse(ant._try_chamber_pickup(engine))
        self.assertFalse(ant.has_cargo)

    def test_ant_does_not_drop_raw_input_into_full_processor_buffer(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        nutrient = _chamber_by_type(engine, "nutrient_processor")
        nutrient.add_resource(ResourceType.FUNGUS, nutrient.capacity_for(ResourceType.FUNGUS))
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.FUNGUS
        ant.inventory_amount = 1.0
        ant.x = nutrient.x
        ant.y = nutrient.y
        ant._try_drop(engine, drop_signal=1.0)
        self.assertTrue(ant.has_cargo)

    def test_ant_does_not_drop_food_into_full_nursery_buffer(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        nursery = _chamber_by_type(engine, "nursery")
        nursery.add_resource(ResourceType.FOOD, nursery.capacity_for(ResourceType.FOOD))
        before = nursery.inventory.get(ResourceType.FOOD)
        ant = engine.ants[0]
        ant.inventory_type = ResourceType.FOOD
        ant.inventory_amount = 1.0
        ant.x = nursery.x
        ant.y = nursery.y
        ant._try_drop(engine, drop_signal=1.0)
        self.assertEqual(nursery.inventory.get(ResourceType.FOOD), before)

    def test_nursery_tending_updates_ant_stats_and_role(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=1),
            world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
        )
        engine = SimulationEngine(config)
        nursery = _chamber_by_type(engine, "nursery")
        nursery.brood_state.larvae = 2
        nursery.add_resource(ResourceType.FOOD, 5.0)
        nursery.add_resource(ResourceType.WATER, 5.0)
        ant = engine.ants[0]
        ant.x = nursery.x
        ant.y = nursery.y
        engine.colony._update_reproduction(engine.ants, tick=1, events=None)
        self.assertGreater(ant.nursing_ticks, 0)
        self.assertGreater(ant.larvae_helped, 0.0)
        ant.nursing_ticks = 200
        self.assertEqual(infer_role(ant), "Nurse")


class ProcessorDemandTests(unittest.TestCase):
    def test_processor_input_demand_raises_raw_priority(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        _set_colony_storage(engine, {
            ResourceType.LEAVES: engine.colony.desired_storage_for(ResourceType.LEAVES),
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
            ResourceType.PROTEIN: engine.colony.desired_storage_for(ResourceType.PROTEIN),
            ResourceType.WASTE: engine.colony.desired_storage_for(ResourceType.WASTE),
            ResourceType.FOOD: 0.0,
        })
        self.assertGreater(engine.colony.processor_input_demand(ResourceType.LEAVES), 0.0)
        self.assertGreater(engine.colony.resource_priority(ResourceType.LEAVES), 0.05)

    def test_output_full_processor_stops_requesting_inputs(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        nutrient = _chamber_by_type(engine, "nutrient_processor")
        nutrient.add_resource(ResourceType.FOOD, nutrient.capacity_for(ResourceType.FOOD))
        engine.colony.refresh_aggregate_inventory()
        engine.colony._refresh_demand_cache()
        self.assertEqual(nutrient.input_demand(ResourceType.FUNGUS), 0.0)

    def test_processed_storage_saturation_reduces_output_pressure(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        paste_storage = engine.colony.storage_chambers_for(ResourceType.PROTEIN_PASTE)[0]
        paste_storage.add_resource(ResourceType.PROTEIN_PASTE, paste_storage.capacity_for(ResourceType.PROTEIN_PASTE))
        engine.colony.refresh_aggregate_inventory()
        engine.colony._refresh_demand_cache()
        self.assertEqual(engine.colony.processor_output_pressure(ResourceType.PROTEIN_PASTE), 0.0)

    def test_fungus_shortage_raises_fungus_priority(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        _set_colony_storage(engine, {
            ResourceType.FOOD: 0.0,
            ResourceType.FUNGUS: 0.0,
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
        })
        self.assertGreater(engine.colony.resource_priority(ResourceType.FUNGUS), 0.4)

    def test_fungus_saturation_stops_farm_input_demand(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        _set_colony_storage(engine, {
            ResourceType.FOOD: engine.colony.desired_storage_for(ResourceType.FOOD),
            ResourceType.FUNGUS: engine.colony.desired_storage_for(ResourceType.FUNGUS),
            ResourceType.LEAVES: engine.colony.desired_storage_for(ResourceType.LEAVES),
            ResourceType.WATER: engine.colony.desired_storage_for(ResourceType.WATER),
            ResourceType.COMPOST: engine.colony.desired_storage_for(ResourceType.COMPOST),
        })
        _fill_nursery_buffers(engine)
        farm = _chamber_by_type(engine, "fungus_farm")
        self.assertEqual(engine.colony.processor_output_demand(ResourceType.FUNGUS), 0.0)
        self.assertEqual(engine.colony.processor_input_demand(ResourceType.LEAVES), 0.0)

    def test_compost_saturation_lowers_optional_farm_demand_but_does_not_block_growth(self) -> None:
        engine = SimulationEngine(DEFAULT_CONFIG)
        compost_capacity = engine.colony.storage_capacity(ResourceType.COMPOST)
        _set_colony_storage(engine, {
            ResourceType.COMPOST: compost_capacity,
            ResourceType.FUNGUS: 0.0,
            ResourceType.FOOD: 0.0,
        })
        self.assertLess(engine.colony.resource_priority(ResourceType.COMPOST), 0.05)
        farm = _chamber_by_type(engine, "fungus_farm")
        farm.add_resource(ResourceType.LEAVES, 4.0)
        farm.add_resource(ResourceType.WATER, 2.0)
        farm.farm_tick(worker_count=0, config=DEFAULT_CONFIG.colony, tick=1)
        self.assertGreater(farm.fungus_farm_state.biomass, 0.0)


def _set_colony_storage(engine: SimulationEngine, amounts: dict[ResourceType, float]) -> None:
    for chamber in engine.colony.chambers:
        chamber.inventory.amounts.clear()
    engine.colony.overflow_inventory.amounts.clear()
    for resource_type, amount in amounts.items():
        engine.colony.add_to_storage(resource_type, amount, tick=-1)
    engine.colony.refresh_aggregate_inventory()
    engine.colony._refresh_demand_cache()


def _fill_nursery_buffers(engine: SimulationEngine) -> None:
    for chamber in engine.colony.nursery_chambers():
        chamber.add_resource(ResourceType.FOOD, chamber.capacity_for(ResourceType.FOOD))
        chamber.add_resource(ResourceType.WATER, chamber.capacity_for(ResourceType.WATER))
    engine.colony.refresh_aggregate_inventory()
    engine.colony._refresh_demand_cache()


def _single_ant_engine() -> SimulationEngine:
    config = replace(
        DEFAULT_CONFIG,
        ants=replace(DEFAULT_CONFIG.ants, count=1),
        world=replace(DEFAULT_CONFIG.world, width=60, height=40, nest_x=30.0, nest_y=20.0, leaf_patch_count=0),
    )
    return SimulationEngine(config)


def _processor_chamber(recipe) -> Chamber:
    capacity: dict[ResourceType, float] = {}
    desired: dict[ResourceType, float] = {}
    for resource_type, amount in recipe.inputs.items():
        desired[resource_type] = amount * 2.0
        capacity[resource_type] = amount * DEFAULT_CONFIG.colony.processor_input_capacity_multiplier
    for resource_type, amount in recipe.outputs.items():
        capacity[resource_type] = amount * DEFAULT_CONFIG.colony.processor_output_capacity_multiplier
    return Chamber(
        chamber_id=999,
        chamber_type=recipe.recipe_id,
        x=0.0,
        y=0.0,
        radius=3.0,
        accepted_inputs=set(recipe.inputs),
        outputs=set(recipe.outputs),
        desired_inventory_by_resource=desired,
        capacity_by_resource=capacity,
        processing_state=ProcessingState(recipe),
    )


def _nursery_chamber() -> Chamber:
    config = DEFAULT_CONFIG.colony
    return Chamber(
        chamber_id=997,
        chamber_type="nursery",
        x=0.0,
        y=0.0,
        radius=3.0,
        accepted_inputs={ResourceType.FOOD, ResourceType.WATER},
        desired_inventory_by_resource={
            ResourceType.FOOD: config.nursery_food_buffer,
            ResourceType.WATER: config.nursery_water_buffer,
        },
        capacity_by_resource={
            ResourceType.FOOD: config.nursery_food_buffer * 1.35,
            ResourceType.WATER: config.nursery_water_buffer * 1.35,
            ResourceType.WASTE: max(4.0, config.nursery_capacity * 0.18),
        },
        brood_state=NurseryState(),
    )


def _fungus_farm_chamber() -> Chamber:
    config = DEFAULT_CONFIG.colony
    return Chamber(
        chamber_id=998,
        chamber_type="fungus_farm",
        x=0.0,
        y=0.0,
        radius=3.0,
        accepted_inputs={ResourceType.LEAVES, ResourceType.WATER, ResourceType.COMPOST},
        outputs={ResourceType.FUNGUS},
        desired_inventory_by_resource={
            ResourceType.LEAVES: config.fungus_farm_leaf_buffer,
            ResourceType.WATER: config.fungus_farm_water_buffer,
            ResourceType.COMPOST: config.fungus_farm_compost_buffer,
        },
        capacity_by_resource={
            ResourceType.LEAVES: config.fungus_farm_leaf_buffer * 1.35,
            ResourceType.WATER: config.fungus_farm_water_buffer * 1.35,
            ResourceType.COMPOST: config.fungus_farm_compost_buffer * 1.35,
            ResourceType.FUNGUS: config.fungus_farm_output_capacity,
        },
        fungus_farm_state=FungusFarmState(),
    )


def _chamber_by_type(engine: SimulationEngine, chamber_type: str) -> Chamber:
    for chamber in engine.colony.chambers:
        if chamber.chamber_type == chamber_type:
            return chamber
    raise AssertionError(f"Missing chamber {chamber_type}")


if __name__ == "__main__":
    unittest.main()
