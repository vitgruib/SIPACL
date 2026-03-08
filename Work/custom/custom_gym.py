from scenic.core.simulators import Simulator, Simulation
from scenic.core.scenarios import Scenario
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional, Tuple

import glob
import io
import os
import random
import numpy as np

# Default obs/action shapes for MetaDrive (ego + nav); PPO expects Box after FlattenObservation.
DEFAULT_OBS_SHAPE = (19,)
DEFAULT_ACTION_SHAPE = (2,)

DOUBLE, RESAMPLE, NEW = 0, 1, 2


class ResetException(Exception):
    def __init__(self):
        super().__init__("Resetting")


class MetaDriveEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        scenario: Scenario,
        simulator: Simulator,
        render_mode: Optional[str] = None,
        max_steps: int = 1000,
        observation_space: Optional[spaces.Space] = None,
        action_space: Optional[spaces.Space] = None,
        record_scenic_sim_results: bool = True,
        feedback_fn: Callable = lambda x: x,
        buffer_dir: Optional[str] = None,
        replay_resample_prob: float = 0.5,
        resume_from_buffer: bool = False,
    ):
        """
        buffer_dir: Directory for scene buffer files (scene_*.bin and *_.npy).
        replay_resample_prob: Probability of resampling from buffer vs new scene.
            Use -1 to disable replay (always generate new scenes). In [0, 1] to enable.
        resume_from_buffer: If True and replay is enabled, load buffer state from buffer_dir on init.
        """
        if observation_space is None:
            observation_space = spaces.Box(low=0, high=1, shape=DEFAULT_OBS_SHAPE, dtype=np.float32)
        if action_space is None:
            action_space = spaces.Box(
                low=np.array([-1, -1], dtype=np.float32),
                high=np.array([1, 1], dtype=np.float32),
                shape=DEFAULT_ACTION_SHAPE,
                dtype=np.float32,
            )

        assert render_mode is None or render_mode in self.metadata["render_modes"]

        self.observation_space = observation_space
        self.action_space = action_space
        self.render_mode = render_mode
        self.max_steps = max_steps if max_steps == -1 else max_steps - 1
        self._step_limit = None if max_steps == -1 else self.max_steps
        self.simulator = simulator
        self.scenario = scenario
        self.simulation_results = []
        self.episode_rewards = []

        self.feedback_result = None
        self.loop = None
        self.record_scenic_sim_results = record_scenic_sim_results
        self.feedback_fn = feedback_fn

        self._buffer_dir = buffer_dir or "buffer"
        os.makedirs(self._buffer_dir, exist_ok=True)
        self.is_special_training = replay_resample_prob >= 0
        self.is_continuation = resume_from_buffer and self.is_special_training
        self.buffer_p = max(0.0, min(1.0, replay_resample_prob)) if self.is_special_training else 0.0

        if self.is_special_training and not self.is_continuation:
            for p in glob.glob(os.path.join(self._buffer_dir, "scene_*.bin")):
                try:
                    os.remove(p)
                except OSError:
                    pass
            for name in ("buffer_filenames.npy", "buffer_learning_potential.npy", "buffer_last_reward.npy", "episode_rewards.npy"):
                p = os.path.join(self._buffer_dir, name)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        def _load(path: str, default: np.ndarray) -> np.ndarray:
            try:
                return np.load(path)
            except FileNotFoundError:
                return default

        base = self._buffer_dir
        empty = np.array([])
        self.buffer_filenames = _load(f"{base}/buffer_filenames.npy", empty) if self.is_continuation else empty
        self.buffer_learning_potential = _load(f"{base}/buffer_learning_potential.npy", empty) if self.is_continuation else empty
        self.buffer_last_reward = _load(f"{base}/buffer_last_reward.npy", empty) if self.is_continuation else empty
        if self.is_continuation:
            print(f"loaded files: {self.buffer_filenames}, learning potential: {self.buffer_learning_potential}, last reward: {self.buffer_last_reward}")

        self.working_index = -1
        self.episode_mode = NEW
        self.counting_reward = 0

    def _load_scene(self, index: int):
        with io.open(f"{self._buffer_dir}/scene_{index}.bin", "rb") as f:
            return self.scenario.sceneFromBytes(f.read())

    def _save_scene(self, index: int, scene) -> None:
        with io.open(f"{self._buffer_dir}/scene_{index}.bin", "wb") as f:
            f.write(self.scenario.sceneToBytes(scene=scene))

    def _pick_scene(self) -> Tuple:
        if not self.is_special_training:
            scene, _ = self.scenario.generate(feedback=self.feedback_result)
            return scene, -1, NEW
        n = len(self.buffer_filenames)
        lp_len, lr_len = len(self.buffer_learning_potential), len(self.buffer_last_reward)
        if lp_len != lr_len:
            idx = n - 1
            return self._load_scene(idx), idx, DOUBLE
        if random.uniform(0, 1) < self.buffer_p and n > 0:
            probs = self.buffer_learning_potential / np.sum(self.buffer_learning_potential)
            idx = int(np.random.choice(n, p=probs))
            return self._load_scene(idx), idx, RESAMPLE
        scene, _ = self.scenario.generate(feedback=self.feedback_result)
        idx = n
        self._save_scene(idx, scene)
        self.buffer_filenames = np.append(self.buffer_filenames, idx)
        return scene, idx, NEW

    def _make_run_loop(self):
        while True:
            try:
                scene, self.working_index, self.episode_mode = self._pick_scene()
                self.counting_reward = 0
                step_limit = self._step_limit
                with self.simulator.simulateStepped(scene, maxSteps=step_limit) as simulation:
                    steps_taken = 0
                    done = lambda: simulation.result is not None
                    truncated = lambda: (step_limit is not None and steps_taken >= step_limit) or (step_limit is None and simulation.get_truncation())
                    observation = simulation.get_obs()
                    info = simulation.get_info()
                    actions = yield observation, info
                    simulation.actions = actions
                    while not done():
                        simulation.advance()
                        steps_taken += 1
                        observation = simulation.get_obs()
                        info = simulation.get_info()
                        reward = simulation.get_reward()
                        self.counting_reward += reward
                        term, trun = done(), truncated()
                        if term or trun:
                            self.episode_rewards.append(self.counting_reward)
                            self.logScores()
                        if term:
                            self.feedback_result = self.feedback_fn(simulation.result)
                            if self.record_scenic_sim_results:
                                self.simulation_results.append(simulation.result)
                        if term or trun:
                            if term and getattr(simulation, "result", None):
                                reason = (
                                    simulation.result.get("reason", "terminated")
                                    if isinstance(simulation.result, dict)
                                    else "terminated"
                                )
                            elif trun:
                                reason = "truncated"
                            else:
                                reason = "unknown"
                            # outcome: won (goal +10), lost (penalty -5 / crash / off_road), truncated
                            if trun:
                                outcome = "truncated"
                            elif term and reward >= 5:
                                outcome = "won"
                            elif term and reward <= -1:
                                outcome = "lost"
                            else:
                                outcome = "unknown"
                            info = {**info, "termination_reason": reason, "outcome": outcome}
                        actions = yield observation, reward, term, trun, info
                        if term or trun:
                            break
                        simulation.actions = actions
            except ResetException:
                print("reset exception caught")
                continue

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.loop is None:
            print("self loop doesnt exist, creating new one")
            self.loop = self._make_run_loop()
            observation, info = next(self.loop) # not doing self.scene.send(action) just yet
        else:
            observation, info = self.loop.throw(ResetException())


        return observation, info
        
    def step(self, action):
        assert self.loop is not None, "must call reset() before step()"
        observation, reward, terminated, truncated, info = self.loop.send(action)
        return observation, reward, terminated, truncated, info

    def render(self): # TODO figure out if this function has to be implemented here or if super() has default implementation
        """
        likely just going to be something like simulation.render() or something
        """
        # FIXME for one project only...also a bit hacky...
        # self.env.render()
        pass

    def close(self):
        if self.episode_rewards:
            path = f"{self._buffer_dir}/episode_rewards.npy"
            with io.open(path, "wb") as f:
                np.save(f, np.array(self.episode_rewards))
        self.simulator.destroy()
        
    def logScores(self):
        if not self.is_special_training:
            return
        total = self.counting_reward
        if total == 0:
            print("TOTAL REWARD is 0! suspicious!")
        i = self.working_index
        if self.episode_mode == DOUBLE:
            lp = abs(total - self.buffer_last_reward[i]) + 1e-8
            self.feedback_result = -lp
            self.buffer_learning_potential = np.append(self.buffer_learning_potential, lp)
            self.buffer_last_reward[i] = total
        elif self.episode_mode == RESAMPLE:
            if i >= len(self.buffer_last_reward):
                print(f"Warning: working index {i} out of bounds for buffer_last_reward len {len(self.buffer_last_reward)}")
            lp = abs(total - self.buffer_last_reward[i]) + 1e-8
            self.buffer_learning_potential[i] = lp
            self.buffer_last_reward[i] = total
        else:
            self.buffer_last_reward = np.append(self.buffer_last_reward, total)
        base = self._buffer_dir
        for name, arr in [
            ("buffer_filenames.npy", self.buffer_filenames),
            ("buffer_learning_potential.npy", self.buffer_learning_potential),
            ("buffer_last_reward.npy", self.buffer_last_reward),
        ]:
            with io.open(f"{base}/{name}", "wb") as f:
                np.save(f, arr)
        print("Saved buffer data to disk")