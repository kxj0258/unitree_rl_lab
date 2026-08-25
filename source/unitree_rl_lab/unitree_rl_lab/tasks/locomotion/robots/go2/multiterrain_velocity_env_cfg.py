"""Challenging mixed-terrain velocity task for the Unitree Go2."""

import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RobotEnvCfg


# Keep the terrain mix aligned to the 20 curriculum columns so that every
# proportion maps to an exact integer number of columns.
GO2_MULTITERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.10),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.01, 0.08),
            noise_step=0.01,
            border_width=0.25,
        ),
        "pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.05, 0.45),
            platform_width=2.0,
            border_width=0.25,
        ),
        "inverted_pyramid_slope": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.05, 0.45),
            platform_width=2.0,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.20),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "inverted_pyramid_stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.20),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.45,
            grid_height_range=(0.04, 0.16),
            platform_width=2.0,
            holes=False,
        ),
    },
)


@configclass
class MultiTerrainRobotEnvCfg(RobotEnvCfg):
    """Go2 velocity task over a challenging seven-type terrain curriculum."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GO2_MULTITERRAIN_CFG.copy()
        self.curriculum.terrain_levels.func = mdp.multiterrain_levels_survival
        self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None
        self.scene.terrain.max_init_terrain_level = 1
        # The inherited +/-0.1 m/s range cannot cover the 4 m distance needed
        # by Isaac Lab's terrain_levels_vel curriculum to move an environment
        # up. Start MultiTerrain v2 at a moderate command range that makes the
        # existing distance-based progression reachable within a 20 s episode.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)


@configclass
class MultiTerrainRobotPlayEnvCfg(MultiTerrainRobotEnvCfg):
    """Smaller scene that retains all terrain types for playback and evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 20
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
