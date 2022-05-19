from typing import Tuple
from scene import Scene
import taichi as ti
from taichi.math import *
import numpy as np
from pathlib import Path
import pickle

scene = Scene(exposure=3)
scene.set_directional_light((0.1, 0.1, 0.1), 0.1, (1, 1, 1))
scene.set_floor(-0.05, (0.3, 0.4, 0.6))


# target_range = 0 #32
# total_depth = 6

# @ti.kernel
# def set_single(x: ti.template ,y: ti.template(), z: ti.template(), light: ti.template()):
#     scene.set_voxel(vec3(x,y,z), 1, vec3(0,0,0))   

# def tachi_from_cubic_dic():  
#     for i in range(total_depth, total_depth + 1):
#         current_depth = total_depth - i
#         base = 1
# #        #base = target_range*2 / (2 ** current_depth)
#         for key, value in cubic_dic.items():
#             [depth, x, y, z] = [int(item) for item in key.split('_')]
#             coord_x = x * base - target_range
#             coord_y = y * base - target_range
#             coord_z = z * base - target_range
#             set_single(coord_x, coord_y, coord_z, 1)

# tachi_from_cubic_dic()
# scene.finish()



def importer() -> dict:
    with open("./cubic_dic_7.pkl", "rb") as fp:
        return pickle.load(fp)


def prep_data(dic: dict) -> Tuple[int, int, np.ndarray]:
    """The voxel challenge forbids creating new Taichi Fields."""
    dic = importer()
    dic_l = len(dic.keys())
    data = np.ndarray(shape=(dic_l, 6), dtype=np.int32)
    
    counter = 0
    for key, value in importer().items():
        if value != "Protein":
            print(value)
#        if value not in ["Ligand", "Protein", "Mixture"]:
#            continue
#        if counter >= 10:
#            break
        [depth, x, y, z] = [int(item) for item in key.split('_')]
        y = 128-y 
        if value == "Ligand":
            data[counter] = np.array([x, y, z, 200, 200, 200])
        #else:
        #    continue
        elif value == "Protein":
            data[counter] = np.array([x, y, z, 50,99,187 ]) # 100,100,100
        else:
            data[counter] = np.array([x, y, z, 200, 200, 200]) 
        if value != "Protein":
            print(data[counter])
        counter += 1 
    return dic_l, 6, data

total_depth = 6
target_range = 32
cubic = importer()
base = 1
m, n, external_data = prep_data(cubic)
print("m, n: ", m, n)
external = ti.field(ti.i32, shape=(m, n))
external.from_numpy(external_data)

print(external_data[:10,:])
print(external_data.sum(axis = 0))
@ti.func
def compute_and_set(x, y, z, target_range, base, c1, c2, c3):
    #scene.set_voxel(vec3(x * base - target_range, y * base - target_range, z * base  - target_range), 1, vec3(1,1,1)) 
    #print(c1,c2,c3)
    scene.set_voxel(vec3(x-target_range,y-target_range,z-target_range), 1, vec3(c1/256, c2/256, c3/256))

@ti.kernel
def initialize_voxels():
    for i in range(m):
        base = 1   #target_range * 2 / (2 ** current_depth)
        compute_and_set(external[i, 0], external[i, 1], external[i, 2], target_range, base, external[i, 3], external[i, 4], external[i, 5])


initialize_voxels()
scene.finish()
