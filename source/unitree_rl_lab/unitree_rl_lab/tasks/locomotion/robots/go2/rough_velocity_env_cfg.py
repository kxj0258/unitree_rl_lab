import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass

from .velocity_env_cfg import RobotEnvCfg


GO2_ROUGH_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
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
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.5,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
    },
)


@configclass
class RoughRobotEnvCfg(RobotEnvCfg):
    """Go2 velocity task with terrain curriculum and unchanged policy observations."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = GO2_ROUGH_TERRAINS_CFG.copy()
        self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None
        self.scene.terrain.max_init_terrain_level = 1


@configclass
class RoughRobotPlayEnvCfg(RoughRobotEnvCfg):
    """Smaller rough-terrain scene for playback and evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 2
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
