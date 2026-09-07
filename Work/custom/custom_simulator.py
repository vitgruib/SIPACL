"""Simulator interface for MetaDrive (based on scenarios/controllers/simulator.py)."""

try:
    from metadrive.component.traffic_participants.pedestrian import Pedestrian
    from metadrive.component.vehicle.vehicle_type import DefaultVehicle
except ImportError as e:
    raise ModuleNotFoundError(
        "Metadrive is required. Please install the 'metadrive-simulator' package (and sumolib) or use scenic[metadrive]."
    ) from e

import logging
import math
import time

from scenic.core.simulators import InvalidScenarioError, SimulationCreationError
from scenic.domains.driving.actions import *
from scenic.domains.driving.controllers import (
    PIDLateralController,
    PIDLongitudinalController,
)
from scenic.domains.driving.simulators import DrivingSimulation, DrivingSimulator
import scenic.simulators.metadrive.utils as utils
import numpy as np

# Observation size expected by MetaDriveEnv (DEFAULT_OBS_SHAPE); keep in sync for vector env batching.
# Layout (all entries normalized to [0, 1]):
#   [0:10]  lane_line_detector cloud points (num_lasers=10; 1.0 = no lane-line/sidewalk hit within `distance`)
#   [10:14] nearest-vehicle features: forward_offset, lateral_offset, distance, present
#   [14:19] ego state: speed, heading_sin, heading_cos, steering, last throttle_brake
# If lane_line_detector's num_lasers changes, OBS_SIZE (and DEFAULT_OBS_SHAPE in custom_gym.py) must change too.
OBS_SIZE = 19

# Normalization range (meters) for the nearest-vehicle relative-position features.
NEAREST_CAR_RANGE = 50.0


class CustomMetaDriveSimulator(DrivingSimulator):
    """Implementation of `Simulator` for MetaDrive (controllers/simulator.py style)."""

    def __init__(
        self,
        timestep=0.1,
        render=False,
        render3D=False,
        sumo_map=None,
        real_time=False,
        max_steps=1000,
    ):
        super().__init__()
        self.render = render
        self.render3D = render3D if render else False
        self.scenario_number = 0
        self.timestep = timestep
        self.sumo_map = sumo_map
        self.real_time = real_time
        self.max_steps = max_steps
        self.scenic_offset, self.sumo_map_boundary = utils.getMapParameters(self.sumo_map)
        self.film_size = utils.calculateFilmSize(self.sumo_map_boundary, scaling=5) if (self.render and not self.render3D) else None

        decision_repeat = math.ceil(self.timestep / 0.02)
        physics_world_step_size = self.timestep / decision_repeat
        self.vehicle_config = {
            "spawn_position_heading": [(0.0, 0.0), 0.0],
            "lane_line_detector": dict(num_lasers=10, distance=20),
        }
        self.client = utils.DriveEnv(
            dict(
                decision_repeat=decision_repeat,
                physics_world_step_size=physics_world_step_size,
                use_render=self.render3D,
                vehicle_config=self.vehicle_config,
                use_mesh_terrain=self.render3D,
                log_level=logging.CRITICAL,
            )
        )
        self.client.config["sumo_map"] = self.sumo_map

    def createSimulation(self, scene, *, timestep, **kwargs):
        self.scenario_number += 1
        obj = scene.objects[0]
        converted_position = utils.scenicToMetaDrivePosition(obj.position, self.scenic_offset)
        converted_heading = utils.scenicToMetaDriveHeading(obj.heading)

        self.client.config["vehicle_config"]["spawn_position_heading"] = [
            converted_position,
            converted_heading,
        ]
        self.client.config["vehicle_config"]["spawn_velocity"] = [obj.velocity.x, obj.velocity.y]
        self.client.config["vehicle_config"]["lane_line_detector"] = dict(num_lasers=10, distance=20)

        return CustomMetaDriveSimulation(
            scene,
            render=self.render,
            render3D=self.render3D,
            scenario_number=self.scenario_number,
            timestep=self.timestep,
            sumo_map=self.sumo_map,
            real_time=self.real_time,
            scenic_offset=self.scenic_offset,
            sumo_map_boundary=self.sumo_map_boundary,
            film_size=self.film_size,
            client=self.client,
            max_steps=self.max_steps,
            **kwargs,
        )

    def destroy(self):
        """Tear down the MetaDrive client so the engine singleton is freed (allows creating new envs)."""
        if self.client and getattr(self.client, "close", None):
            self.client.close()
        super().destroy()


class CustomMetaDriveSimulation(DrivingSimulation):
    def __init__(
        self,
        scene,
        render,
        render3D,
        scenario_number,
        timestep,
        sumo_map,
        real_time,
        scenic_offset,
        sumo_map_boundary,
        film_size,
        client,
        max_steps=1000,
        **kwargs,
    ):
        if len(scene.objects) == 0:
            raise InvalidScenarioError(
                "Metadrive requires you to define at least one Scenic object."
            )
        if not scene.objects[0].isCar:
            raise InvalidScenarioError(
                "The first object must be a car to serve as the ego vehicle in Metadrive."
            )

        self.render = render
        self.render3D = render3D
        self.scenario_number = scenario_number
        self.defined_ego = False
        self.timestep = timestep
        self.sumo_map = sumo_map
        self.real_time = real_time
        self.scenic_offset = scenic_offset
        self.sumo_map_boundary = sumo_map_boundary
        self.film_size = film_size
        self.client = client
        self.max_steps = max_steps
        self.steps_taken = 0
        self.result = None  # set when terminal so gym's done() is True

        step_result = self.client.reset()
        self.info = step_result[1] if isinstance(step_result, tuple) and len(step_result) > 1 else {}
        self.reward = 0.0
        self.actions = [0.0, 0.0]  # list so gym can assign [steer, throttle_brake]
        super().__init__(scene, timestep=timestep, **kwargs)
        # client.reset() ran before actors existed (DummyObservation, unusable anyway);
        # build the real observation now that createObjectInSimulator has run and
        # scene.objects[0].metaDriveActor is set.
        self.observation = self._build_observation()

    def createObjectInSimulator(self, obj):
        converted_position = utils.scenicToMetaDrivePosition(
            obj.position, self.scenic_offset
        )
        converted_heading = utils.scenicToMetaDriveHeading(obj.heading)

        if not self.defined_ego:
            metadrive_objects = self.client.engine.get_objects()
            obj.metaDriveActor = list(metadrive_objects.values())[0]
            self.defined_ego = True
            return

        if obj.isVehicle:
            metaDriveActor = self.client.engine.agent_manager.spawn_object(
                DefaultVehicle,
                vehicle_config=dict(spawn_velocity=[0, 0], random_color=True),
                position=converted_position,
                heading=converted_heading,
            )
            obj.metaDriveActor = metaDriveActor
            return

        if obj.isPedestrian:
            metaDriveActor = self.client.engine.agent_manager.spawn_object(
                Pedestrian,
                position=converted_position,
                heading_theta=converted_heading,
            )
            obj.metaDriveActor = metaDriveActor
            return

        raise SimulationCreationError(
            f"Unsupported object type: {type(obj)} for object {obj}."
        )

    def executeActions(self, allActions):
        super().executeActions(allActions)
        for obj in self.scene.objects[1:]:
            if obj.isVehicle:
                action = obj._collect_action()
                obj.metaDriveActor.before_step(action)
                obj._reset_control()
            else:
                if obj._walking_direction is None:
                    obj._walking_direction = utils.scenicToMetaDriveHeading(obj.heading)
                if obj._walking_speed is None:
                    obj._walking_speed = obj.speed
                direction = [
                    math.cos(obj._walking_direction),
                    math.sin(obj._walking_direction),
                ]
                obj.metaDriveActor.set_velocity(direction, obj._walking_speed)

    def step(self):
        start_time = time.monotonic()
        ego_obj = self.scene.objects[0]

        step_out = self.client.step([self.actions[0], self.actions[1]])
        if isinstance(step_out, (list, tuple)) and len(step_out) >= 5:
            self.info = step_out[4]
        # step_out[0] is MetaDrive's DummyObservation output (always [0]) — ignored;
        # we build the real observation from the lane-line lidar + scene state below.
        self.observation = self._build_observation()
        self.reward = float(getattr(ego_obj, "reward", 0.0))
        ego_obj._reset_control()

        self.steps_taken += 1

        # Terminal detection: set result so gym's done() is True
        if self.result is None and hasattr(self.client, "vehicle"):
            s = self.client.vehicle.get_state()
            if s.get("crash_vehicle") or s.get("crash_object") or s.get("crash_building") or s.get("crash_sidewalk"):
                self.result = {"reason": "crash"}
            elif not s.get("on_lane", True):
                self.result = {"reason": "out_of_road"}

        if self.render and not self.render3D:
            self.client.render(
                mode="topdown", semantic_map=True, film_size=self.film_size, scaling=5
            )

        if self.real_time:
            elapsed = time.monotonic() - start_time
            if elapsed < self.timestep:
                time.sleep(self.timestep - elapsed)

    def _perceive_lane_lines(self, ego_actor):
        """Ray-cast the lane_line_detector lidar from the ego and return its cloud points.

        Each entry is in [0, 1]: 1.0 means no lane-line/sidewalk hit within `distance`,
        smaller values mean a closer hit. See metadrive DistanceDetector.perceive.
        """
        cfg = self.client.config["vehicle_config"]["lane_line_detector"]
        num_lasers, distance = cfg["num_lasers"], cfg["distance"]
        if num_lasers <= 0 or distance <= 0:
            return np.ones(0, dtype=np.float32)
        sensor = self.client.engine.get_sensor("lane_line_detector")
        result = sensor.perceive(
            ego_actor,
            self.client.engine.physics_world.static_world,
            num_lasers=num_lasers,
            distance=distance,
            show=False,
        )
        return np.asarray(result.cloud_points, dtype=np.float32)

    def _nearest_vehicle_features(self, ego_obj):
        """Relative position of the nearest other vehicle, in the ego's local (forward/lateral) frame.

        Returns [forward, lateral, distance, present], all normalized to [0, 1] using
        NEAREST_CAR_RANGE. When no other vehicle exists, forward/lateral default to the
        midpoint (0.5, "unknown") and distance defaults to 1.0 ("far"), with present=0.
        """
        ego_actor = ego_obj.metaDriveActor
        ex, ey = ego_actor.position[0], ego_actor.position[1]
        heading = ego_actor.heading_theta
        cos_h, sin_h = math.cos(heading), math.sin(heading)

        nearest_dist = None
        nearest_forward = 0.0
        nearest_lateral = 0.0
        for obj in self.scene.objects[1:]:
            if not obj.isVehicle:
                continue
            actor = getattr(obj, "metaDriveActor", None)
            if actor is None:
                continue
            dx = actor.position[0] - ex
            dy = actor.position[1] - ey
            dist = math.hypot(dx, dy)
            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                # Rotate the relative vector into the ego's heading-aligned frame.
                nearest_forward = dx * cos_h + dy * sin_h
                nearest_lateral = -dx * sin_h + dy * cos_h

        if nearest_dist is None:
            return np.array([0.5, 0.5, 1.0, 0.0], dtype=np.float32)

        r = NEAREST_CAR_RANGE
        forward_n = float(np.clip((nearest_forward / r + 1.0) / 2.0, 0.0, 1.0))
        lateral_n = float(np.clip((nearest_lateral / r + 1.0) / 2.0, 0.0, 1.0))
        dist_n = float(np.clip(nearest_dist / r, 0.0, 1.0))
        present = 1.0 if nearest_dist <= r else 0.0
        return np.array([forward_n, lateral_n, dist_n, present], dtype=np.float32)

    def _ego_state_features(self, ego_obj):
        """Basic ego kinematics, normalized to [0, 1]."""
        actor = ego_obj.metaDriveActor
        speed_n = float(np.clip(actor.speed_km_h / max(actor.max_speed_km_h, 1e-6), 0.0, 1.0))
        heading = actor.heading_theta
        heading_sin_n = (math.sin(heading) + 1.0) / 2.0
        heading_cos_n = (math.cos(heading) + 1.0) / 2.0
        steering_n = float(np.clip((actor.steering + 1.0) / 2.0, 0.0, 1.0))
        throttle_brake_n = float(np.clip((actor.throttle_brake + 1.0) / 2.0, 0.0, 1.0))
        return np.array(
            [speed_n, heading_sin_n, heading_cos_n, steering_n, throttle_brake_n], dtype=np.float32
        )

    def _build_observation(self):
        ego_obj = self.scene.objects[0]
        lane_lines = self._perceive_lane_lines(ego_obj.metaDriveActor)
        nearest = self._nearest_vehicle_features(ego_obj)
        ego_state = self._ego_state_features(ego_obj)
        return np.concatenate([lane_lines, nearest, ego_state]).astype(np.float32)

    def get_obs(self):
        obs = np.asarray(self.observation, dtype=np.float32).flatten()
        if obs.size >= OBS_SIZE:
            return obs[:OBS_SIZE].copy()
        out = np.zeros(OBS_SIZE, dtype=np.float32)
        out[: obs.size] = obs
        return out

    def get_truncation(self):
        if self.max_steps is None or self.max_steps < 0:
            return False
        return self.steps_taken >= self.max_steps

    def get_reward(self):
        return self.reward

    def get_info(self):
        self.info["ego_pos"] = getattr(self.scene.objects[0], "position", None)
        self.info["ego_speed"] = getattr(self.scene.objects[0], "speed", 0.0)
        return self.info

    def getProperties(self, obj, properties):
        metaDriveActor = obj.metaDriveActor
        position = utils.metadriveToScenicPosition(
            metaDriveActor.position, self.scenic_offset
        )
        velocity = Vector(*metaDriveActor.velocity, 0)
        speed = metaDriveActor.speed
        md_ang_vel = metaDriveActor.body.getAngularVelocity()
        angularVelocity = Vector(*md_ang_vel)
        angularSpeed = math.hypot(*md_ang_vel)
        converted_heading = utils.metaDriveToScenicHeading(metaDriveActor.heading_theta)
        yaw, pitch, roll = obj.parentOrientation.globalToLocalAngles(
            converted_heading, 0, 0
        )
        elevation = 0
        return dict(
            position=position,
            velocity=velocity,
            speed=speed,
            angularSpeed=angularSpeed,
            angularVelocity=angularVelocity,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            elevation=elevation,
        )

    def getLaneFollowingControllers(self, agent):
        dt = self.timestep
        if agent.isCar:
            lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
            lat_controller = PIDLateralController(K_P=0.13, K_D=0.3, K_I=0.05, dt=dt)
        else:
            lon_controller = PIDLongitudinalController(K_P=0.25, K_D=0.025, K_I=0.0, dt=dt)
            lat_controller = PIDLateralController(K_P=0.2, K_D=0.1, K_I=0.0, dt=dt)
        return lon_controller, lat_controller

    def getTurningControllers(self, agent):
        dt = self.timestep
        if agent.isCar:
            lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
            lat_controller = PIDLateralController(K_P=0.2, K_D=0.2, K_I=0.2, dt=dt)
        else:
            lon_controller = PIDLongitudinalController(K_P=0.25, K_D=0.025, K_I=0.0, dt=dt)
            lat_controller = PIDLateralController(K_P=0.4, K_D=0.1, K_I=0.0, dt=dt)
        return lon_controller, lat_controller

    def getLaneChangingControllers(self, agent):
        dt = self.timestep
        if agent.isCar:
            lon_controller = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
            lat_controller = PIDLateralController(K_P=0.2, K_D=0.2, K_I=0.02, dt=dt)
        else:
            lon_controller = PIDLongitudinalController(K_P=0.25, K_D=0.025, K_I=0.0, dt=dt)
            lat_controller = PIDLateralController(K_P=0.1, K_D=0.3, K_I=0.0, dt=dt)
        return lon_controller, lat_controller

    def destroy(self):
        if self.client and getattr(self.client, "engine", None):
            object_ids = list(self.client.engine._spawned_objects.keys())
            if object_ids:
                self.client.engine.agent_manager.clear_objects(object_ids)
        super().destroy()
