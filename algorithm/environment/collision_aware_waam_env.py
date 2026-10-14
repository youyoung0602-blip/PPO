from __future__ import annotations
from typing import Any
import gymnasium as gym
from gymnasium import spaces
import numpy as np


def _seg_dist_2d(a0, a1, b0, b1):
    # Robust enough for reward shaping; official Validator remains authority.
    def point_seg(p, x, y):
        d = y - x
        q = float(np.dot(d, d))
        if q <= 1e-12:
            return float(np.linalg.norm(p - x))
        u = float(np.clip(np.dot(p - x, d) / q, 0.0, 1.0))
        return float(np.linalg.norm(p - (x + u * d)))
    # 2-D segment intersection -> zero distance.
    def orient(p, q, r): return float(np.cross(q-p, r-p))
    o1,o2,o3,o4 = orient(a0,a1,b0),orient(a0,a1,b1),orient(b0,b1,a0),orient(b0,b1,a1)
    if (o1*o2 < 0.0) and (o3*o4 < 0.0):
        return 0.0
    return min(point_seg(a0,b0,b1), point_seg(a1,b0,b1), point_seg(b0,a0,a1), point_seg(b1,a0,a1))


class CollisionAwareWAAMEnv(gym.Env):
    """Fixed-order, 3-robot PPO assignment environment with collision-aware reward.

    Action 0/1/2 means rank among robots by projected finish time, matching the
    current WAAMBaselineEnv convention. Collision terms are reward shaping only;
    final safety must be enforced by the scheduler + professor Validator.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, scenario: dict[str, Any]):
        super().__init__()
        self.scenario = scenario
        self.robots_cfg = sorted(scenario["robots"], key=lambda r: int(r["robot_id"]))
        if [int(r["robot_id"]) for r in self.robots_cfg] != [1,2,3]:
            raise ValueError("CollisionAwareWAAMEnv requires robot IDs 1,2,3")
        self.tasks_cfg = scenario["tasks"]
        self.n_tasks = len(self.tasks_cfg)
        if not self.n_tasks: raise ValueError("No tasks")
        p = scenario["process"]
        self.travel_speed = float(p["travel_speed_mm_s"])
        self.dep_speed = float(p["deposition_speed_mm_s"])
        self.xyz_scale = float(scenario.get("normalization_radius_mm", 2000.0))
        self.time_scale = float(scenario.get("time_normalization_s", 1000.0))
        self.cc = scenario.get("collision", {})
        rw = scenario.get("collision_reward", {})
        self.w_makespan = float(rw.get("makespan", 1.0))
        self.w_travel = float(rw.get("travel", 0.30))
        self.w_imbalance = float(rw.get("imbalance", 0.50))
        self.collision_penalty = float(rw.get("collision", 1000.0))
        self.near_penalty = float(rw.get("near", 10.0))
        self.near_margin_mm = float(rw.get("near_margin_mm", 50.0))
        self.finish_bonus = float(rw.get("finish_bonus", 20.0))
        self.action_space = spaces.Discrete(3)
        # robots 3*(xyz,time,reach_margin,nearest_arm_margin,nearest_tcp_margin)=21
        # task start/end=6, progress=1, projected delta makespan=3 => 31
        self.observation_space = spaces.Box(-20.0, 20.0, shape=(31,), dtype=np.float32)
        self.robot_pos = self.robot_time = None
        self.task_start = self.task_end = None
        self.assignment = None
        self.current_task = 0

    def _robot(self, i): return self.robots_cfg[i]
    def _base(self, i): return np.asarray(self._robot(i)["base_xyz_mm"], float)
    def _reach(self, i): return float(self._robot(i).get("xy_reach_radius_mm", self.xyz_scale))
    def _arm_r(self, i): return float(self._robot(i).get("arm_envelope_radius_mm", 100.0))
    def _tcp_r(self, i): return float(self._robot(i).get("tcp_radius_mm", 100.0))
    def _clearance(self): return float(self.cc.get("arm_clearance_mm", 50.0))

    def _pair_margins(self, i, pi, j, pj):
        d = _seg_dist_2d(self._base(i)[:2], pi[:2], self._base(j)[:2], pj[:2])
        arm = d - (self._arm_r(i)+self._arm_r(j)+self._clearance())
        tcp = float(np.linalg.norm(pi[:2]-pj[:2])) - (self._tcp_r(i)+self._tcp_r(j))
        return arm, tcp

    def _min_margins_for(self, i, p):
        arms=[]; tcps=[]
        for j in range(3):
            if j == i: continue
            a,t = self._pair_margins(i,p,j,self.robot_pos[j])
            arms.append(a); tcps.append(t)
        return min(arms), min(tcps)

    def _projected(self, i, start, end):
        tt=float(np.linalg.norm(start-self.robot_pos[i]))/self.travel_speed
        dt=float(np.linalg.norm(end[:2]-start[:2]))/self.dep_speed
        return float(self.robot_time[i]+tt+dt),tt,dt

    def _makespan(self): return float(np.max(self.robot_time))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos=np.asarray([r.get("home_xyz_mm",r["base_xyz_mm"]) for r in self.robots_cfg],dtype=np.float32)
        self.robot_time=np.zeros(3,dtype=np.float32)
        self.task_start=np.asarray([t["start_xyz_mm"] for t in self.tasks_cfg],dtype=np.float32)
        self.task_end=np.asarray([t["end_xyz_mm"] for t in self.tasks_cfg],dtype=np.float32)
        self.assignment=np.full(self.n_tasks,-1,dtype=np.int32)
        self.current_task=0
        return self._get_obs(), {"makespan":0.0}

    def _get_obs(self):
        obs=[]
        for i in range(3):
            p=self.robot_pos[i]
            arm,tcp=self._min_margins_for(i,p)
            reach=self._reach(i)-float(np.linalg.norm(p[:2]-self._base(i)[:2]))
            obs.extend((p/self.xyz_scale).tolist())
            obs.extend([float(self.robot_time[i]/self.time_scale), reach/self.xyz_scale, arm/self.xyz_scale, tcp/self.xyz_scale])
        if self.current_task < self.n_tasks:
            s=self.task_start[self.current_task]; e=self.task_end[self.current_task]
            cur=self._makespan(); deltas=[]
            for i in range(3):
                pt,_,_=self._projected(i,s,e)
                rt=self.robot_time.copy(); rt[i]=pt
                deltas.append((float(np.max(rt))-cur)/self.time_scale)
        else:
            s=np.zeros(3,np.float32); e=np.zeros(3,np.float32); deltas=[0.,0.,0.]
        obs.extend((s/self.xyz_scale).tolist()); obs.extend((e/self.xyz_scale).tolist())
        obs.append(float(self.current_task/self.n_tasks)); obs.extend(deltas)
        return np.asarray(obs,dtype=np.float32)

    def step(self, action):
        if self.current_task >= self.n_tasks:
            return self._get_obs(),0.0,True,False,{"makespan":self._makespan()}
        k=self.current_task; s=self.task_start[k]; e=self.task_end[k]
        projected=[self._projected(i,s,e)[0] for i in range(3)]
        ranking=np.argsort(projected); i=int(ranking[int(action)])
        before=self._makespan(); finish,travel_t,dep_t=self._projected(i,s,e)
        # Reward-shaping collision proxy: sample direct travel + deposition while
        # other robots are held at their current TCP. Final scheduler performs
        # time-resolved hard checks against moving robots.
        worst_arm=float("inf"); worst_tcp=float("inf")
        collision=False
        path=[(self.robot_pos[i].copy(),s.copy()),(s.copy(),e.copy())]
        for a,b in path:
            n=max(2,int(np.ceil(np.linalg.norm(b-a)/25.0))+1)
            for u in np.linspace(0.0,1.0,n):
                p=a+(b-a)*u
                arm,tcp=self._min_margins_for(i,p)
                worst_arm=min(worst_arm,arm); worst_tcp=min(worst_tcp,tcp)
                if arm <= 0.0 or tcp <= 0.0: collision=True
        self.robot_time[i]=finish; self.robot_pos[i]=e; self.assignment[k]=i+1; self.current_task+=1
        after=self._makespan(); mean=max(float(np.mean(self.robot_time)),1e-6)
        imbalance=float(np.std(self.robot_time)/mean)
        near=max(0.0,(self.near_margin_mm-min(worst_arm,worst_tcp))/self.near_margin_mm)
        reward=-(self.w_makespan*(after-before))-(self.w_travel*travel_t)-(self.w_imbalance*imbalance)-(self.near_penalty*near)
        if collision: reward-=self.collision_penalty
        done=self.current_task>=self.n_tasks
        if done: reward+=self.finish_bonus
        info={"robot_id":i+1,"task_id":k,"travel_time":travel_t,"deposition_time":dep_t,"makespan":after,"collision_proxy":collision,"min_arm_margin_proxy":worst_arm,"min_tcp_margin_proxy":worst_tcp}
        return self._get_obs(),float(reward),done,False,info

    def render(self):
        print("tasks",self.current_task,"/",self.n_tasks,"makespan",self._makespan())
