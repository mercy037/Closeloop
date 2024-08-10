from collections import OrderedDict
from itertools import product
from typing import List, Dict, TYPE_CHECKING, Optional, Union, Tuple
from gym import spaces
import numpy as np
import pandas as pd

from config import cfg
import common
from common.utils import lamp
from common.utils import Vector

class ObservationType(object):
    def __init__(self, env: "gym_env-v0", **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle



class OccupancyGridObservation(ObservationType):

    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: List[str] = ['presence', 'vx', 'vy', 'on_road']
    GRID_SIZE: List[List[float]] = [[-5.5*5, 5.5*5], [-5.5*5, 5.5*5]]
    GRID_STEP: List[int] = [5, 5]

    def __init__(self,
                 env: "gym_env-v0",
                 features: Optional[List[str]] = None,
                 grid_size: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                 grid_step: Optional[Tuple[float, float]] = None,
                 features_range: Dict[str, List[float]] = None,
                 absolute: bool = False,
                 align_to_vehicle_axes: bool = False,
                 clip: bool = True,
                 as_image: bool = False,
                 **kwargs: dict) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param grid_size: real world size of the grid [[min_x, max_x], [min_y, max_y]]
        :param grid_step: steps between two cells of the grid [step_x, step_y]
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: use absolute or relative coordinates
        :param align_to_vehicle_axes: if True, the grid axes are aligned with vehicle axes. Else, they are aligned
               with world axes.
        :param clip: clip the observation in [-1, 1]
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        self.grid_step = np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        grid_shape = np.asarray(np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step), dtype=np.int)
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute
        self.align_to_vehicle_axes = align_to_vehicle_axes
        self.clip = clip
        self.as_image = as_image

    def space(self) -> spaces.Space:
        if self.as_image:
            return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        else:
            return spaces.Box(shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2*float(cfg.GYM_ENV.MAX_SPEED), 2*float(cfg.GYM_ENV.MAX_SPEED)],
                "vy": [-2*float(cfg.GYM_ENV.MAX_SPEED), 2*float(cfg.GYM_ENV.MAX_SPEED)]
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = common.utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Initialize empty data
            self.grid.fill(np.nan)

            # Get nearby traffic data
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles])
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                if feature in df.columns:  # A vehicle feature
                    for _, vehicle in df.iterrows():
                        x, y = vehicle["x"], vehicle["y"]
                        # Recover unnormalized coordinates for cell index
                        if "x" in self.features_range:
                            x = common.utils.lmap(x, [-1, 1], [self.features_range["x"][0], self.features_range["x"][1]])
                        if "y" in self.features_range:
                            y = common.utils.lmap(y, [-1, 1], [self.features_range["y"][0], self.features_range["y"][1]])
                        cell = self.pos_to_index((x, y), relative=not self.absolute)
                        if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                            self.grid[layer, cell[1], cell[0]] = vehicle[feature]
                elif feature == "on_road":
                    self.fill_road_layer_by_lanes(layer)

            obs = self.grid

            if self.clip:
                obs = np.clip(obs, -1, 1)

            if self.as_image:
                obs = ((np.clip(obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)

            obs = np.nan_to_num(obs).astype(self.space().dtype)

            return obs

    def pos_to_index(self, position: Vector, relative: bool = False) -> Tuple[int, int]:
        """
        Convert a world position to a grid cell index

        If align_to_vehicle_axes the cells are in the vehicle's frame, otherwise in the world frame.

        :param position: a world position
        :param relative: whether the position is already relative to the observer's position
        :return: the pair (i,j) of the cell index
        """
        if not relative:
            position -= self.observer_vehicle.position
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(self.observer_vehicle.heading)
            position = np.array([[c, s], [-s, c]]) @ position
        return int(np.floor((position[0] - self.grid_size[0, 0]) / self.grid_step[0])),\
               int(np.floor((position[1] - self.grid_size[1, 0]) / self.grid_step[1]))

    def index_to_pos(self, index: Tuple[int, int]) -> np.ndarray:

        position = np.array([
            (index[1] + 0.5) * self.grid_step[0] + self.grid_size[0, 0],
            (index[0] + 0.5) * self.grid_step[1] + self.grid_size[1, 0]
        ])
        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(-self.observer_vehicle.heading)
            position = np.array([[c, s], [-s, c]]) @ position

        position += self.observer_vehicle.position
        return position

    def fill_road_layer_by_lanes(self, layer_index: int, lane_perception_distance: float = 100) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        Here, we iterate over lanes and regularly placed waypoints on these lanes to fill the corresponding cells.
        This approach is faster if the grid is large and the road network is small.

        :param layer_index: index of the layer in the grid
        :param lane_perception_distance: lanes are rendered +/- this distance from vehicle location
        """
        lane_waypoints_spacing = np.amin(self.grid_step)
        road = self.env.road

        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for lane in road.network.graph[_from][_to]:
                    origin, _ = lane.local_coordinates(self.observer_vehicle.position)
                    waypoints = np.arange(origin - lane_perception_distance,
                                            origin + lane_perception_distance,
                                            lane_waypoints_spacing).clip(0, lane.length)
                    for waypoint in waypoints:
                        cell = self.pos_to_index(lane.position(waypoint, 0))
                        if 0 <= cell[1] < self.grid.shape[-2] and 0 <= cell[0] < self.grid.shape[-1]:
                            self.grid[layer_index, cell[1], cell[0]] = 1

    def fill_road_layer_by_cell(self, layer_index) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        In this implementation, we iterate the grid cells and check whether the corresponding world position
        at the center of the cell is onroad/offroad. This approach is faster if the grid is small and the road network large.
        """
        road = self.env.road
        for i, j in product(range(self.grid.shape[-2]), range(self.grid.shape[-1])):
            for _from in road.network.graph.keys():
                for _to in road.network.graph[_from].keys():
                    for lane in road.network.graph[_from][_to]:
                        if lane.on_lane(self.index_to_pos((i, j))):
                            self.grid[layer_index, i, j] = 1


def observation_factory(env: "gym_env-v0", config: dict) -> ObservationType:
    if config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
