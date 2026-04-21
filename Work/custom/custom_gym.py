from scenic.core.simulators import Simulator, Simulation
from scenic.core.scenarios import Scenario
import gymnasium as gym
from gymnasium import spaces
from typing import Callable, Optional, Tuple

import csv
import glob
import io
import os
import random
import sys
import numpy as np

# Default obs/action shapes for MetaDrive (ego + nav); PPO expects Box after FlattenObservation.
DEFAULT_OBS_SHAPE = (19,)
DEFAULT_ACTION_SHAPE = (2,)

DOUBLE, RESAMPLE, NEW = 0, 1, 2

# Max buffered scenes (FIFO eviction when exceeded).
DEFAULT_BUFFER_MAX = 5000

# Fixed PLR internals (not exposed on CLI): EMA on |Δ mean return/step|; rank sample ∝ 1/rank**alpha.
DEFAULT_LP_EMA_BETA = 0.2
DEFAULT_LP_RANK_ALPHA = 1.0


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
        buffer_max: int = DEFAULT_BUFFER_MAX,
        lp_strategy: str = "pvl",
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        buffer_dir: Directory for scene buffer files (scene_*.bin and *_.npy).
        replay_resample_prob: Probability of resampling from buffer vs new scene.
            Use -1 to disable replay (always generate new scenes). In [0, 1] to enable.
        buffer_max: Maximum buffered scenes; oldest removed (FIFO) when exceeded.
        lp_strategy: Which learning-progress formula to use for PLR scoring.
            Currently supported: "l1" (|Δ mean reward|), "pvl" (positive value loss).
            Add new variants in _lp_delta().
        gamma: Discount factor (used by PVL strategy for GAE computation).
        gae_lambda: GAE lambda (used by PVL strategy for GAE computation).
        LP EMA beta and rank alpha are fixed (DEFAULT_LP_EMA_BETA, DEFAULT_LP_RANK_ALPHA), not CLI args.
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
        self.buffer_p = max(0.0, min(1.0, replay_resample_prob)) if self.is_special_training else 0.0
        self.lp_strategy = lp_strategy
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lp_ema_beta = max(1e-6, min(1.0, float(DEFAULT_LP_EMA_BETA)))
        self.buffer_max = max(1, int(buffer_max))
        self.lp_rank_alpha = max(1e-6, min(10.0, float(DEFAULT_LP_RANK_ALPHA)))

        if self.is_special_training:
            for p in glob.glob(os.path.join(self._buffer_dir, "scene_*.bin")):
                try:
                    os.remove(p)
                except OSError:
                    pass
            for name in (
                "buffer_filenames.npy",
                "buffer_learning_potential.npy",
                "buffer_last_reward.npy",
                "buffer_lp_ema.npy",
                "plr_episode_seq.npy",
                "episode_rewards.npy",
                "episode_lp_values.npy",
                "lp_episode_log.csv",
            ):
                p = os.path.join(self._buffer_dir, name)
                if os.path.isfile(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

        base = self._buffer_dir
        empty = np.array([])
        self.buffer_filenames = empty
        self.buffer_learning_potential = empty
        self.buffer_last_reward = empty
        self.buffer_lp_ema = empty
        self._plr_episode_seq = 0
        self.episode_lp_log: list[float] = []

        self._ep_rewards = []
        self._ep_values = []

        self.working_index = -1
        self.episode_mode = NEW
        self.counting_reward = 0
        self._closed = False
        # When True, next _pick_scene must resample (so a newly added scene gets replayed once).
        self._force_resample_next = False
        self._episode_steps = 0

    def _load_scene(self, index: int):
        with io.open(f"{self._buffer_dir}/scene_{index}.bin", "rb") as f:
            return self.scenario.sceneFromBytes(f.read())

    def _save_scene(self, index: int, scene) -> None:
        with io.open(f"{self._buffer_dir}/scene_{index}.bin", "wb") as f:
            f.write(self.scenario.sceneToBytes(scene=scene))

    def _evict_oldest_if_full(self) -> None:
        """Drop oldest buffered scene (FIFO) if at capacity."""
        while len(self.buffer_filenames) >= self.buffer_max:
            old_id = int(self.buffer_filenames[0])
            path = os.path.join(self._buffer_dir, f"scene_{old_id}.bin")
            try:
                if os.path.isfile(path):
                    os.remove(path)
            except OSError:
                pass
            self.buffer_filenames = np.delete(self.buffer_filenames, 0)
            self.buffer_learning_potential = np.delete(self.buffer_learning_potential, 0)
            self.buffer_last_reward = np.delete(self.buffer_last_reward, 0)
            self.buffer_lp_ema = np.delete(self.buffer_lp_ema, 0)

    # Huge LP so a newly added scene is chosen when we force-resample on the next reset.
    _NEW_SCENE_LEARNING_POTENTIAL = 1e10

    def _replay_probs_from_lp(self, lp: np.ndarray) -> np.ndarray:
        """Rank-based sampling (PER-style): P(i) ∝ 1/rank_i**alpha, rank 1 = highest score.

        `lp` is typically ``buffer_learning_potential`` (learning progress per slot).
        New scenes use huge LP so they stay rank 1 until updated.
        Ties break by buffer index (stable sort: lower index first among equal LP).
        """
        n = int(lp.size)
        if n == 0:
            return np.asarray(lp, dtype=np.float64)
        lp = np.asarray(lp, dtype=np.float64)
        order = np.argsort(-lp, kind="stable")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        w = 1.0 / np.power(ranks, self.lp_rank_alpha)
        s = float(np.sum(w))
        if not np.isfinite(s) or s <= 0:
            return np.full(n, 1.0 / n)
        return w / s

    def _pick_scene(self) -> Tuple:
        if not self.is_special_training:
            scene, _ = self.scenario.generate(feedback=self.feedback_result)
            return scene, -1, NEW
        n = len(self.buffer_filenames)
        lp_len, lr_len = len(self.buffer_learning_potential), len(self.buffer_last_reward)
        # Guarantee we resample next after adding a new scene (replay it once); huge LP picks that scene.
        lp_ema_len = len(self.buffer_lp_ema)
        buf_ok = n > 0 and n == lp_len == lr_len == lp_ema_len
        lp_scores = np.asarray(self.buffer_learning_potential, dtype=np.float64)
        if self._force_resample_next and buf_ok:
            self._force_resample_next = False
            probs = self._replay_probs_from_lp(lp_scores)
            pos = int(np.random.choice(n, p=probs))
            file_id = int(self.buffer_filenames[pos])
            return self._load_scene(file_id), pos, RESAMPLE
        if not self._force_resample_next and random.uniform(0, 1) < self.buffer_p and n > 0 and buf_ok:
            probs = self._replay_probs_from_lp(lp_scores)
            pos = int(np.random.choice(n, p=probs))
            file_id = int(self.buffer_filenames[pos])
            return self._load_scene(file_id), pos, RESAMPLE
        scene, _ = self.scenario.generate(feedback=self.feedback_result)
        self._evict_oldest_if_full()
        n = len(self.buffer_filenames)
        file_id = int(np.max(self.buffer_filenames)) + 1 if n > 0 else 0
        self._save_scene(file_id, scene)
        self.buffer_filenames = np.append(self.buffer_filenames, file_id)
        return scene, len(self.buffer_filenames) - 1, NEW

    def _make_run_loop(self):
        while True:
            try:
                scene, self.working_index, self.episode_mode = self._pick_scene()
                self.counting_reward = 0
                self._ep_rewards.clear()
                self._ep_values.clear()
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
                            self._episode_steps = max(1, steps_taken)
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

    def log_step_data(self, reward, value):
        """Called by PPO each step to feed per-step data for PVL computation."""
        self._ep_rewards.append(float(reward))
        self._ep_values.append(float(value))

    def close(self):
        if self._closed:
            return
        # Cannot safely do I/O during interpreter shutdown (__del__ with sys.meta_path is None).
        if getattr(sys, "meta_path", None) is None:
            return
        if self.episode_rewards:
            path = f"{self._buffer_dir}/episode_rewards.npy"
            with io.open(path, "wb") as f:
                np.save(f, np.array(self.episode_rewards))
        self.simulator.destroy()
        self._closed = True
        
    def _lp_delta(self, mean_r: float, prev_mean_r: float) -> float:
        """Raw LP score before EMA smoothing.  Dispatches on self.lp_strategy.

        Add new strategies here as elif branches.
        """
        if self.lp_strategy == "l1":
            return abs(mean_r - prev_mean_r)
        elif self.lp_strategy == "pvl":
            if len(self._ep_rewards) < 1 or len(self._ep_values) < 1:
                return 0.0
            n = len(self._ep_rewards)
            lastgaelam = 0.0
            advantages = [0.0] * n
            gamma, lam = self.gamma, self.gae_lambda
            for t in reversed(range(n)):
                if t == n - 1:
                    next_v, nextnonterminal = 0.0, 0.0
                else:
                    next_v = self._ep_values[t + 1]
                    nextnonterminal = 1.0
                delta = self._ep_rewards[t] + gamma * next_v * nextnonterminal - self._ep_values[t]
                advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
                advantages[t] = max(advantages[t], 0.0)
            return sum(advantages) / len(advantages)
        raise ValueError(f"Unknown lp_strategy: {self.lp_strategy!r}")

    def _compute_learning_progress(
        self,
        mean_r: float,
        prev_mean_r: float,
        old_ema: float,
    ) -> float:
        """EMA-smooth the raw LP score from _lp_delta.

        Args:
            mean_r: Mean reward per step for the episode just finished.
            prev_mean_r: Mean reward per step from the previous visit to this scene.
            old_ema: EMA-smoothed LP from previous visits.

        Returns:
            Updated LP value for this buffer slot.
        """
        raw = self._lp_delta(mean_r, prev_mean_r)
        if old_ema >= 0.5 * self._NEW_SCENE_LEARNING_POTENTIAL:
            return float(raw)
        beta = self.lp_ema_beta
        return beta * raw + (1.0 - beta) * old_ema

    def logScores(self):
        if not self.is_special_training:
            return
        total = self.counting_reward
        mean_r = total / float(self._episode_steps)
        i = self.working_index
        if self.episode_mode == RESAMPLE:
            if i < 0 or i >= len(self.buffer_last_reward):
                print(
                    f"Warning: working index {i} out of bounds for buffer_last_reward len {len(self.buffer_last_reward)}"
                )
                return
            lp_new = self._compute_learning_progress(
                mean_r, self.buffer_last_reward[i], self.buffer_lp_ema[i]
            )
            self.buffer_learning_potential[i] = lp_new
            self.buffer_lp_ema[i] = lp_new
            self.buffer_last_reward[i] = mean_r
        else:
            self.buffer_last_reward = np.append(self.buffer_last_reward, mean_r)
            self.buffer_learning_potential = np.append(
                self.buffer_learning_potential,
                self._NEW_SCENE_LEARNING_POTENTIAL,
            )
            self.buffer_lp_ema = np.append(self.buffer_lp_ema, self._NEW_SCENE_LEARNING_POTENTIAL)
            self._force_resample_next = True

        self._plr_episode_seq += 1

        # Episodic LP log (optional detail); visualize_learning_progress reads lp_per_episode from runs_results.csv.
        lp_log_path = os.path.join(self._buffer_dir, "lp_episode_log.csv")
        write_lp_row = False
        slot, fid, lp_val, mode_s = -1, -1, 0.0, ""
        if self.episode_mode == RESAMPLE and 0 <= i < len(self.buffer_filenames):
            slot = i
            fid = int(self.buffer_filenames[slot])
            lp_val = float(self.buffer_learning_potential[slot])
            mode_s = "RESAMPLE"
            write_lp_row = True
        elif self.episode_mode == NEW and len(self.buffer_filenames) > 0:
            slot = len(self.buffer_filenames) - 1
            fid = int(self.buffer_filenames[slot])
            lp_val = float(self.buffer_learning_potential[slot])
            mode_s = "NEW"
            write_lp_row = True
        if write_lp_row:
            self.episode_lp_log.append(float(lp_val))
        try:
            if write_lp_row:
                new_file = not os.path.isfile(lp_log_path)
                with io.open(lp_log_path, "a", encoding="utf-8", newline="") as lf:
                    w = csv.writer(lf)
                    if new_file:
                        w.writerow(
                            [
                                "plr_episode_seq",
                                "mode",
                                "slot",
                                "file_id",
                                "lp_value",
                                "mean_r_per_step",
                            ]
                        )
                    w.writerow(
                        [int(self._plr_episode_seq), mode_s, slot, fid, lp_val, float(mean_r)]
                    )
        except OSError:
            pass

        base = self._buffer_dir
        for name, arr in [
            ("buffer_filenames.npy", self.buffer_filenames),
            ("buffer_learning_potential.npy", self.buffer_learning_potential),
            ("buffer_last_reward.npy", self.buffer_last_reward),
            ("buffer_lp_ema.npy", self.buffer_lp_ema),
            ("plr_episode_seq.npy", np.array([self._plr_episode_seq], dtype=np.int64)),
            ("episode_lp_values.npy", np.asarray(self.episode_lp_log, dtype=np.float64)),
        ]:
            with io.open(f"{base}/{name}", "wb") as f:
                np.save(f, arr)
        print("Saved buffer data to disk")