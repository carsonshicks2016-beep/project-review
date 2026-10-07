from __future__ import annotations

from dataclasses import dataclass

from biofactory.resources.resource_types import ResourceType


@dataclass(frozen=True)
class ProductionRecipe:
    recipe_id: str
    name: str
    inputs: dict[ResourceType, float]
    outputs: dict[ResourceType, float]
    work: float


MVP_LEAF_TO_FOOD = ProductionRecipe(
    recipe_id="mvp_leaf_to_food",
    name="Simplified leaf food conversion",
    inputs={ResourceType.LEAVES: 1.0},
    outputs={ResourceType.FOOD: 0.35},
    work=1.0,
)

NUTRIENT_PROCESSING = ProductionRecipe(
    recipe_id="nutrient_processing",
    name="Nutrient processor",
    inputs={
        ResourceType.FUNGUS: 1.0,
        ResourceType.WATER: 0.30,
    },
    outputs={ResourceType.FOOD: 1.05},
    work=70.0,
)

LEAF_FALLBACK_PROCESSING = ProductionRecipe(
    recipe_id="leaf_fallback_processing",
    name="Leaf fallback processor",
    inputs={
        ResourceType.LEAVES: 2.6,
        ResourceType.WATER: 0.80,
    },
    outputs={ResourceType.FOOD: 0.35},
    work=160.0,
)

PROTEIN_PROCESSING = ProductionRecipe(
    recipe_id="protein_processing",
    name="Protein processor",
    inputs={
        ResourceType.PROTEIN: 1.4,
        ResourceType.WATER: 0.25,
    },
    outputs={ResourceType.PROTEIN_PASTE: 1.0},
    work=100.0,
)

COMPOST_PROCESSING = ProductionRecipe(
    recipe_id="compost_processing",
    name="Compost processor",
    inputs={
        ResourceType.WASTE: 1.0,
        ResourceType.LEAVES: 0.25,
    },
    outputs={ResourceType.COMPOST: 0.9},
    work=130.0,
)

FUNGUS_SUBSTRATE_PROCESSING = ProductionRecipe(
    recipe_id="fungus_substrate_processing",
    name="Fungus substrate mixing",
    inputs={
        ResourceType.LEAVES: 1.8,
        ResourceType.WATER: 0.35,
    },
    outputs={ResourceType.FUNGUS_SUBSTRATE: 1.0},
    work=65.0,
)

FUNGUS_FARM_CULTIVATION = ProductionRecipe(
    recipe_id="fungus_farm_cultivation",
    name="Fungus farm cultivation",
    inputs={
        ResourceType.FUNGUS_SUBSTRATE: 1.0,
        ResourceType.WATER: 0.25,
        ResourceType.COMPOST: 0.10,
    },
    outputs={ResourceType.FUNGUS: 1.15},
    work=140.0,
)

FUNGUS_TO_NUTRIENT_PASTE = ProductionRecipe(
    recipe_id="fungus_to_nutrient_paste",
    name="Fungus nutrient paste",
    inputs={
        ResourceType.FUNGUS: 1.0,
        ResourceType.WATER: 0.25,
    },
    outputs={ResourceType.NUTRIENT_PASTE: 1.0},
    work=70.0,
)

LARVAE_FOOD_PROCESSING = ProductionRecipe(
    recipe_id="larvae_food_processing",
    name="Larvae food mixing",
    inputs={
        ResourceType.NUTRIENT_PASTE: 1.0,
        ResourceType.WATER: 0.18,
    },
    outputs={ResourceType.LARVAE_FOOD: 1.0},
    work=60.0,
)

PROTEIN_RECOVERY = ProductionRecipe(
    recipe_id="protein_recovery",
    name="Corpse protein recovery",
    inputs={
        ResourceType.DEAD_INSECTS: 1.0,
        ResourceType.DEAD_ANTS: 0.5,
        ResourceType.WATER: 0.15,
    },
    outputs={ResourceType.PROTEIN_PASTE: 1.1},
    work=95.0,
)

FERTILIZER_PROCESSING = ProductionRecipe(
    recipe_id="fertilizer_processing",
    name="Compost fertilizer curing",
    inputs={
        ResourceType.COMPOST: 1.0,
        ResourceType.WATER: 0.10,
    },
    outputs={ResourceType.FERTILIZER: 0.85},
    work=90.0,
)

SOLDIER_FEED_PROCESSING = ProductionRecipe(
    recipe_id="soldier_feed_processing",
    name="Soldier feed mixing",
    inputs={
        ResourceType.PROTEIN_PASTE: 1.0,
        ResourceType.SEEDS: 0.35,
    },
    outputs={ResourceType.SOLDIER_FEED: 1.0},
    work=85.0,
)

REINFORCED_SOIL_PROCESSING = ProductionRecipe(
    recipe_id="reinforced_soil_processing",
    name="Reinforced soil packing",
    inputs={
        ResourceType.SOIL: 1.6,
        ResourceType.MINERALS: 0.45,
        ResourceType.RESIN: 0.20,
    },
    outputs={ResourceType.REINFORCED_SOIL: 1.0},
    work=110.0,
)

STRUCTURAL_RESIN_PROCESSING = ProductionRecipe(
    recipe_id="structural_resin_processing",
    name="Structural resin bonding",
    inputs={
        ResourceType.WOOD_FIBER: 1.1,
        ResourceType.RESIN: 0.75,
    },
    outputs={ResourceType.STRUCTURAL_RESIN: 1.0},
    work=95.0,
)

MVP_PROCESSING_RECIPES: tuple[ProductionRecipe, ...] = (
    NUTRIENT_PROCESSING,
    LEAF_FALLBACK_PROCESSING,
    PROTEIN_PROCESSING,
    COMPOST_PROCESSING,
)

PHASE2_PRODUCTION_CHAINS: tuple[ProductionRecipe, ...] = (
    FUNGUS_SUBSTRATE_PROCESSING,
    FUNGUS_FARM_CULTIVATION,
    NUTRIENT_PROCESSING,
    FUNGUS_TO_NUTRIENT_PASTE,
    LARVAE_FOOD_PROCESSING,
    PROTEIN_PROCESSING,
    PROTEIN_RECOVERY,
    SOLDIER_FEED_PROCESSING,
    COMPOST_PROCESSING,
    FERTILIZER_PROCESSING,
    REINFORCED_SOIL_PROCESSING,
    STRUCTURAL_RESIN_PROCESSING,
)
