from __future__ import annotations

from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class WAAMBaselineEnv(gym.Env):

    def __init__(self, scenario: dict[str, Any]):
        super().__init__()

        self.scenario = scenario

        # -------------------------
        # Robot 정보
        # -------------------------
        self.robots_cfg = sorted(
            scenario["robots"],
            key=lambda r: int(r["robot_id"])
        )

        self.n_robots = 3

        # -------------------------
        # Task 정보
        # -------------------------
        self.tasks_cfg = scenario["tasks"]
        self.n_tasks = len(self.tasks_cfg)

        if self.n_tasks == 0:
            raise ValueError("Task가 없습니다.")

        # -------------------------
        # Process 정보
        # -------------------------
        process = scenario["process"]

        self.travel_speed = float(
            process["travel_speed_mm_s"]
        )

        self.deposition_speed = float(
            process["deposition_speed_mm_s"]
        )

        # -------------------------
        # Normalization
        # -------------------------
        self.xyz_scale = float(
            scenario.get(
                "normalization_radius_mm",
                2000.0
            )
        )

        self.time_scale = float(
            scenario.get(
                "time_normalization_s",
                1000.0
            )
        )

        # ==================================================
        # Action
        #
        # 0 = Robot 1
        # 1 = Robot 2
        # 2 = Robot 3
        # ==================================================

        self.action_space = spaces.Discrete(3)

        # ==================================================
        # Observation
        #
        # Robot 3대:
        # x,y,z,time = 12
        #
        # 현재 Task:
        # start xyz + end xyz = 6
        #
        # progress = 1
        #
        # 총 19
        # ==================================================

        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(25,),
            dtype=np.float32
        )

        self.robot_pos = None
        self.robot_time = None

        self.task_start = None
        self.task_end = None

        self.current_task = 0
        self.assignment = None

    # ==================================================
    # 초기화
    # ==================================================

    def reset(
        self,
        *,
        seed=None,
        options=None
    ):

        super().reset(seed=seed)

        # Robot 초기 위치
        self.robot_pos = np.asarray(
            [
                r.get(
                    "home_xyz_mm",
                    r["base_xyz_mm"]
                )
                for r in self.robots_cfg
            ],
            dtype=np.float32
        )

        # Robot별 사용 가능 시간
        self.robot_time = np.zeros(
            3,
            dtype=np.float32
        )

        # Task 좌표
        self.task_start = np.asarray(
            [
                t["start_xyz_mm"]
                for t in self.tasks_cfg
            ],
            dtype=np.float32
        )

        self.task_end = np.asarray(
            [
                t["end_xyz_mm"]
                for t in self.tasks_cfg
            ],
            dtype=np.float32
        )

        # 첫 번째 Task부터 시작
        self.current_task = 0

        # Task별 담당 Robot
        self.assignment = np.full(
            self.n_tasks,
            -1,
            dtype=np.int32
        )

        return self._get_obs(), {
            "makespan": 0.0
        }

    # ==================================================
    # Makespan
    # ==================================================

    def _makespan(self):

        return float(
            np.max(self.robot_time)
        )

    # ==================================================
    # Observation
    # ==================================================

    def _get_obs(self):
        obs = []

        # 1. Robot 3대의 현재 상태
        # 각 Robot: x, y, z, 현재까지 걸린 시간
        for i in range(3):
            obs.extend(
                (self.robot_pos[i] / self.xyz_scale).tolist()
            )

            obs.append(
                float(self.robot_time[i] / self.time_scale)
            )

        # 2. Robot 3대의 상대 부하
        max_time = max(
            float(np.max(self.robot_time)),
            1.0
        )

        relative_load = self.robot_time / max_time

        obs.extend(relative_load.tolist())

        # 3. 현재 Task 정보
        if self.current_task < self.n_tasks:
            start = self.task_start[self.current_task]
            end = self.task_end[self.current_task]

        else:
            start = np.zeros(3, dtype=np.float32)
            end = np.zeros(3, dtype=np.float32)

        obs.extend(
            (start / self.xyz_scale).tolist()
        )

        obs.extend(
            (end / self.xyz_scale).tolist()
        )

        # 4. 현재 진행률
        progress = self.current_task / self.n_tasks
        obs.append(float(progress))

        # 5. 각 Robot에게 현재 Task를 줬을 때
        #    Makespan이 얼마나 증가하는지 계산
        delta_makespans = []

        if self.current_task < self.n_tasks:

            for i in range(3):

                # Robot 현재 위치 -> Task 시작점
                travel_dist = np.linalg.norm(
                    self.robot_pos[i] - start
                )

                travel_time = (
                    travel_dist / self.travel_speed
                )

                # Task 자체의 작업 거리
                deposit_dist = np.linalg.norm(
                    end[:2] - start[:2]
                )

                deposit_time = (
                    deposit_dist / self.deposition_speed
                )

                # 이 Robot이 Task를 수행했을 때 완료시간
                projected_time = (
                    self.robot_time[i]
                    + travel_time
                    + deposit_time
                )

                # 현재 Robot 시간 복사
                projected_robot_times = self.robot_time.copy()

                # 선택한 Robot의 예상 완료시간 적용
                projected_robot_times[i] = projected_time

                # 예상 Makespan
                projected_makespan = np.max(
                    projected_robot_times
                )

                # 현재 Makespan
                current_makespan = np.max(
                    self.robot_time
                )

                # Makespan 증가량
                delta = (
                    projected_makespan
                    - current_makespan
                )

                delta_makespans.append(
                    float(delta / self.time_scale)
                )

        else:
            delta_makespans = [0.0, 0.0, 0.0]

        obs.extend(delta_makespans)

        # 최종 observation
        return np.asarray(
            obs,
            dtype=np.float32
        )
    # ==================================================
    # Step
    # ==================================================

    def step(self, action):

        # PPO는 Robot만 선택
        # 현재 Task
        task_idx = self.current_task
        start = self.task_start[task_idx]
        end = self.task_end[task_idx]

        # 각 Robot이 이 Task를 맡았을 때 예상 완료시간 계산
        projected_times = []

        for i in range(3):
            travel_dist = np.linalg.norm(
                self.robot_pos[i] - start
            )

            travel_time = travel_dist / self.travel_speed

            deposit_dist = np.linalg.norm(
                end[:2] - start[:2]
            )

            deposit_time = deposit_dist / self.deposition_speed

            projected_time = (
                self.robot_time[i]
                + travel_time
                + deposit_time
            )

            projected_times.append(projected_time)

        # 예상 완료시간이 짧은 Robot 순서
        robot_ranking = np.argsort(projected_times)

        # PPO action:
        # 0 = 가장 유리한 Robot
        # 1 = 두 번째 Robot
        # 2 = 세 번째 Robot
        robot_idx = int(robot_ranking[int(action)])

        task_idx = self.current_task

        # 혹시 모든 Task가 끝났다면
        if task_idx >= self.n_tasks:

            return (
                self._get_obs(),
                0.0,
                True,
                False,
                {
                    "makespan":
                        self._makespan()
                }
            )

        # 현재 Task
        start = self.task_start[
            task_idx
        ]

        end = self.task_end[
            task_idx
        ]

        # -------------------------
        # 이동 거리
        # -------------------------

        travel_distance = float(
            np.linalg.norm(
                start
                - self.robot_pos[robot_idx]
            )
        )

        travel_time = (
            travel_distance
            / self.travel_speed
        )

        # -------------------------
        # 적층 거리
        # -------------------------

        deposition_distance = float(
            np.linalg.norm(
                end[:2]
                - start[:2]
            )
        )

        deposition_time = (
            deposition_distance
            / self.deposition_speed
        )

        # 이전 Makespan
        before = self._makespan()
        before_balance = float(np.std(self.robot_time))

        # -------------------------
        # Robot 상태 업데이트
        # -------------------------

        self.robot_time[robot_idx] += (
            travel_time
            + deposition_time
        )

        self.robot_pos[robot_idx] = end

        # 어떤 Robot이 담당했는지 저장
        self.assignment[
            task_idx
        ] = robot_idx + 1

        # 다음 Task
        self.current_task += 1

        # 작업 후 현재 makespan
        after = self._makespan()

        # 1. Makespan 증가에 대한 패널티
        makespan_penalty = after - before

        # 2. Robot 간 작업시간 불균형
        mean_time = np.mean(self.robot_time)

        if mean_time > 0:
            imbalance = np.std(self.robot_time) / mean_time
        else:
            imbalance = 0.0

        # 3. 가장 여유 있는 Robot과 선택한 Robot의 차이
        min_robot_time = np.min(self.robot_time)

        load_penalty = (
            self.robot_time[robot_idx]
            - min_robot_time
        ) / self.time_scale

        # 최종 Reward
        reward = (
            -1.0 * makespan_penalty
            -0.5 * imbalance
            -0.2 * load_penalty
        )

        # -------------------------
        # 종료 확인
        # -------------------------

        terminated = (
            self.current_task
            >= self.n_tasks
        )

        # 전부 끝냈으면 bonus
        if terminated:
            reward += 20.0

        info = {

            "robot_id":
                robot_idx + 1,

            "task_id":
                task_idx,

            "travel_time":
                travel_time,

            "deposition_time":
                deposition_time,

            "makespan":
                after,

            "completed_tasks":
                self.current_task,
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            False,
            info
        )

    # ==================================================
    # 출력
    # ==================================================

    def render(self):

        print(
            "완료 Task:",
            self.current_task,
            "/",
            self.n_tasks
        )

        print(
            "Makespan:",
            self._makespan()
        )

        print(
            "Robot별 완료시간:",
            self.robot_time.tolist()
        )