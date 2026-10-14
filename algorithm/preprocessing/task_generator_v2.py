"""Experimental centerline generator for thin-wall WAAM STL with holes.

Requires: numpy, trimesh, shapely, scipy, scikit-image, pyyaml.
Does not alter the original baseline generator or the professor's Validator.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from functools import reduce
import numpy as np
import trimesh
import yaml
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely import contains_xy
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

@dataclass
class DepositionTask:
    task_id: int
    layer: int
    start_xyz_mm: np.ndarray
    end_xyz_mm: np.ndarray
    def to_dict(self):
        return dict(task_id=self.task_id, layer=self.layer,
                    start_xyz_mm=self.start_xyz_mm.tolist(), end_xyz_mm=self.end_xyz_mm.tolist())

def load_config(path):
    with open(path, encoding='utf-8') as f: return yaml.safe_load(f)

def target_polygon(section):
    """Even/odd fill: outer loop minus holes, preserving nested islands."""
    loops = []
    for path in section.discrete:
        p = Polygon(np.asarray(path)[:, :2])
        if not p.is_valid: p = p.buffer(0)
        if p.area > 1e-5: loops.append(p)
    if not loops: return Polygon()
    return reduce(lambda a,b: a.symmetric_difference(b), loops)

def skeleton_edges(poly, pixel=0.55):
    """Approximate medial-axis graph using a sub-mm occupancy raster.
    Only graph edges are deposited; no contour is deposited on the boundary.
    """
    xmin,ymin,xmax,ymax = poly.bounds
    xs = np.arange(xmin-pixel, xmax+2*pixel, pixel)
    ys = np.arange(ymin-pixel, ymax+2*pixel, pixel)
    X,Y = np.meshgrid(xs,ys)
    mask = contains_xy(poly, X, Y)
    if not mask.any(): return []
    skel = skeletonize(mask)
    pixels = set(zip(*np.nonzero(skel)))
    edges=[]
    for i,j in sorted(pixels):
        for di,dj in ((0,1),(1,-1),(1,0),(1,1)):
            q=(i+di,j+dj)
            if q not in pixels: continue
            # Avoid diagonal shortcuts across occupied orthogonal skeleton links.
            if di and dj and ((i+di,j) in pixels or (i,j+dj) in pixels): continue
            a=np.array([xs[j],ys[i]])
            b=np.array([xs[q[1]],ys[q[0]]])
            if poly.covers(LineString([a,b])):
                edges.append((a,b))
    return edges

def generate_tasks(stl_path, config_path, pixel=0.55):
    c=load_config(config_path); p=c['process']
    mesh=trimesh.load(stl_path,force='mesh')
    if isinstance(mesh,trimesh.Scene): mesh=trimesh.util.concatenate(tuple(mesh.geometry.values()))
    h=float(p['layer_height_mm']); base=float(p['build_plane_z_mm'])
    if h<=0 or pixel<=0: raise ValueError('Positive layer height and pixel required')
    tasks=[]; layer=0
    while True:
        z=base+(layer+(0.5 if p['tcp_z_reference']=='center' else 1.0))*h
        if z>=float(mesh.bounds[1,2])-1e-9:break
        sec=mesh.section(plane_origin=[0,0,z],plane_normal=[0,0,1])
        if sec is not None:
            poly=target_polygon(sec)
            for a,b in skeleton_edges(poly,pixel):
                start=np.array([a[0],a[1],z],dtype=float)
                end=np.array([b[0],b[1],z],dtype=float)
                tasks.append(DepositionTask(len(tasks),layer,start,end))
            print(f'Layer {layer}: target area={poly.area:.1f} mm2, centerline edges={sum(t.layer==layer for t in tasks)}',flush=True)
        layer+=1
    if not tasks:raise ValueError('No centerline tasks generated')
    return tasks

def build_scenario(stl_path,config_path):
    c=load_config(config_path);tasks=generate_tasks(stl_path,config_path)
    robots=[dict(robot_id=int(r['id']),base_xyz_mm=r['base_xyz_mm'],home_xyz_mm=r.get('home_xyz_mm',r['base_xyz_mm'])) for r in sorted(c['robots'],key=lambda r:int(r['id']))]
    return dict(robots=robots,process=dict(travel_speed_mm_s=float(c['process']['travel_speed_mm_s']),deposition_speed_mm_s=float(c['process']['deposition_speed_mm_s'])),tasks=[t.to_dict() for t in tasks],normalization_radius_mm=max(float(r['xy_reach_radius_mm']) for r in c['robots']),time_normalization_s=1000.0,reward={'invalid':-10.0,'finish_bonus':20.0},max_steps=max(20,4*len(tasks)))

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--stl',required=True);ap.add_argument('--config',required=True);ap.add_argument('--pixel',type=float,default=0.55)
    a=ap.parse_args();print('Total tasks:',len(generate_tasks(a.stl,a.config,a.pixel)))
