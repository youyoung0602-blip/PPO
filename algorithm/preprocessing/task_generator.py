from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
import yaml


@dataclass
class DepositionTask:
    task_id: int
    layer: int
    start_xyz_mm: np.ndarray
    end_xyz_mm: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "layer": self.layer,
            "start_xyz_mm": self.start_xyz_mm.astype(float).tolist(),
            "end_xyz_mm": self.end_xyz_mm.astype(float).tolist(),
        }


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("config.yaml을 읽지 못했습니다.")
    return config


def _load_mesh(stl_path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(stl_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError("STL에 geometry가 없습니다.")
        loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError("STL을 Trimesh로 변환하지 못했습니다.")
    return loaded


def _layer_z(config: dict[str, Any], layer: int) -> float:
    p = config["process"]
    h = float(p["layer_height_mm"])
    base = float(p["build_plane_z_mm"])
    ref = p["tcp_z_reference"]
    if ref == "center":
        return base + (layer + 0.5) * h
    if ref == "top":
        return base + (layer + 1.0) * h
    raise ValueError(f"지원하지 않는 tcp_z_reference: {ref}")


def generate_tasks(stl_path: str | Path, config_path: str | Path) -> list[DepositionTask]:
    """Baseline STL -> contour deposition tasks.

    각 layer의 STL 단면 contour를 직선 D task로 분해한다.
    이것은 PPO 파이프라인 연결용 baseline이며 solid infill은 아직 생성하지 않는다.
    """
    config = load_config(config_path)
    mesh = _load_mesh(stl_path)

    process = config["process"]
    layer_height = float(process["layer_height_mm"])
    build_z = float(process["build_plane_z_mm"])
    if layer_height <= 0:
        raise ValueError("layer_height_mm은 0보다 커야 합니다.")

    z_max = float(mesh.bounds[1, 2])
    tasks: list[DepositionTask] = []
    task_id = 0
    layer = 0

    while True:
        z = _layer_z(config, layer)
        if z >= z_max - 1e-9:
            break
        if z < build_z - 1e-9:
            layer += 1
            continue

        section = mesh.section(
            plane_origin=np.array([0.0, 0.0, z]),
            plane_normal=np.array([0.0, 0.0, 1.0]),
        )

        if section is not None:
            for path in section.discrete:
                pts = np.asarray(path, dtype=np.float32)
                if len(pts) < 2:
                    continue
                # Validator의 layer 기준 Z와 정확히 맞춘다.
                pts[:, 2] = z
                for i in range(len(pts) - 1):
                    start = pts[i].copy()
                    end = pts[i + 1].copy()
                    if float(np.linalg.norm(end[:2] - start[:2])) <= 1e-6:
                        continue
                    tasks.append(DepositionTask(task_id, layer, start, end))
                    task_id += 1
        layer += 1

    if not tasks:
        raise ValueError("STL에서 deposition task를 생성하지 못했습니다.")
    return tasks


def build_scenario(stl_path: str | Path, config_path: str | Path) -> dict[str, Any]:
    """기존 WAAMBaselineEnv가 받을 scenario dict를 만든다."""
    config = load_config(config_path)
    tasks = generate_tasks(stl_path, config_path)
    robots = []
    for r in sorted(config["robots"], key=lambda x: int(x["id"])):
        robots.append({
            "robot_id": int(r["id"]),
            "base_xyz_mm": r["base_xyz_mm"],
            "home_xyz_mm": r.get("home_xyz_mm", r["base_xyz_mm"]),
        })
    return {
        "robots": robots,
        "process": {
            "travel_speed_mm_s": float(config["process"]["travel_speed_mm_s"]),
            "deposition_speed_mm_s": float(config["process"]["deposition_speed_mm_s"]),
        },
        "tasks": [t.to_dict() for t in tasks],
        "normalization_radius_mm": max(float(r["xy_reach_radius_mm"]) for r in config["robots"]),
        "time_normalization_s": 1000.0,
        "reward": {"invalid": -10.0, "finish_bonus": 20.0},
        "max_steps": max(20, 4 * len(tasks)),
    }
if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[4]

    stl_path = project_root / "test_data" / "01" / "target.stl"
    config_path = project_root / "test_data" / "01" / "config.yaml"

    print("STL 경로:", stl_path)
    print("Config 경로:", config_path)

    print("STL 존재:", stl_path.exists())
    print("Config 존재:", config_path.exists())

    print("=== Test 01 Task Generator 시작 ===")

    tasks = generate_tasks(
        stl_path,
        config_path
    )

    print("=== Task 생성 완료 ===")
    print("총 Task 수:", len(tasks))

    for task in tasks[:10]:
        print(
            f"Task {task.task_id} | "
            f"Layer {task.layer} | "
            f"{task.start_xyz_mm} -> {task.end_xyz_mm}"
        )