import os
import importlib

import numpy as np
import torch
import pytorch3d
# from pytorch3d.io import save_ply
import trimesh
from pytorch3d.ops import sample_points_from_meshes
import time
import types
import bpy

import pdb
import traceback

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def batch_code_to_mesh(code_list, file_path, file_name, device='cuda', sample_points=True, num_points=16384, execute_method='write_to_file'):
    batch_result = {}
    batch_result['success'] = []
    batch_result['error'] = []
    for code in code_list:
        try:
            # print('executing the following code:')
            # print(code)
            with time_limit(10):
                # enforce a time limit for blender to execute the code
                result = code_to_mesh(code, file_path, file_name, device=device, sample_points=sample_points, num_points=num_points, 
                    save_mesh=False, execute_method=execute_method)
            # print('finish executing the above code\n')
            if result.get('valid_mesh', 0) > 0:
                batch_result['success'].append(1)
                batch_result['error'].append(None)
                for key in result.keys():
                    if key in batch_result.keys():
                        batch_result[key].append(result[key])
                    else:
                        batch_result[key] = [result[key]]
            else:
                # the mesh is empty or all face areas are 0
                batch_result['success'].append(0)
                batch_result['error'].append('the mesh is empty')
        except Exception as error:
            batch_result['success'].append(0)
            print('an error occured when executing the generated code')
            print('the code is')
            print(code)
            print('the error is')
            print(error)
            print(traceback.format_exc())
            batch_result['error'].append(str(error))
            if not str(error) in ["Three-point collinear", "points can't form an arc"]:
                # sometimes a strange error may occur and could affect codes afterwards
                # we restart blender in this case
                bpy.ops.wm.read_homefile()

    batch_result['success'] = torch.Tensor(batch_result['success']).to(device)
    concat_keys = ['points', 'normals']
    for key in concat_keys:
        if key in batch_result.keys():
            batch_result[key] =  torch.stack(batch_result[key], dim=0)
    return batch_result

def code_to_mesh(code, file_path, file_name, device='cuda', sample_points=True, num_points=16384, 
                save_mesh=True, save_dir='vis', execute_method='write_to_file'):
    # start = time.time()
    if execute_method=='write_to_file':
        write_code_to_file(code, file_path, file_name)
        module_path = os.path.join(file_path, os.path.splitext(file_name)[0])
        module_path = module_path.replace('/', '.')
        code_module = None
    else:
        code_module = types.SimpleNamespace()
        updated_code = write_code_to_file(code, None, None, write_to_file=False)
        exec(updated_code, code_module.__dict__)
        module_path = None
    result = load_mesh(module_path, code_module=code_module, device=device, sample_points=sample_points, num_points=num_points, 
                save_mesh=save_mesh, save_dir=save_dir)
    # print('code to mesh time', time.time()-start)
    return result

def write_code_to_file(code, file_path, file_name, write_to_file=True):
    prefix = 'import bpy\nimport bmesh\nimport torch\nfrom llama_recipes.blender_scripts.bpy_lib import *\n\ndef obtain_mesh():\n    delete_all()\n\n'
    suffix = '\n\n    vertices, faces = get_faces()\n    return vertices, faces'
    before, after = code.split('delete_all()')
    after = after.strip('\n')
    lines = after.split('\n')
    lines = ['    ' + l for l in lines]
    lines = '\n'.join(lines)
    content = prefix + lines + suffix
    if write_to_file:
        os.makedirs(file_path, exist_ok=True)
        if os.path.exists(os.path.join(file_path, file_name)):
            os.remove(os.path.join(file_path, file_name))
        f = open(os.path.join(file_path, file_name), "w")
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
        f.close()
    return content

def load_mesh(module_path, code_module=None, device='cpu', sample_points=False, num_points=16384, save_mesh=False, save_dir=None):
    if code_module is None:
        code_module = importlib.import_module(module_path)
        importlib.reload(code_module)
    result = {}
    verts, faces = code_module.obtain_mesh()
    verts = verts.to(device)
    faces = faces.to(device)
    result['verts'] = verts
    result['faces'] = faces
    # verts, faces are cpu tensors of shape N,3
    if save_mesh:
        os.makedirs(save_dir, exist_ok=True)
        # save_ply(os.path.join(save_dir, 'mesh.ply'), verts, faces)
        mesh_save = trimesh.Trimesh(vertices=verts.detach().cpu().numpy(), faces=faces.detach().cpu().numpy())
        mesh_save.export(os.path.join(save_dir, 'mesh.ply'))
    if sample_points:
        # try:
        #     mesh = pytorch3d.structures.Meshes([verts], [faces])
        #     print('verts and faces', verts, faces)
        #     print('verts and faces shape', verts.shape, faces.shape)
        #     points, normals = sample_points_from_meshes(mesh, num_points, return_normals=True) # of shape 1,N,3
        #     points, normals = points[0], normals[0]
        # except Exception as error:
        #     print('an error occured while sampling points from mesh surface')
        #     print('the error is')
        #     print(error)
        #     print('the verts are', verts)
        #     print('the faces are', facess)
        #     print('the verts shape is', verts.shape)
        #     print('the faces shape is', facess.shape)
        
        mesh = pytorch3d.structures.Meshes([verts], [faces])
        if (not mesh.isempty()) and mesh.faces_areas_packed().sum()>0:  
            points, normals = sample_points_from_meshes(mesh, num_points, return_normals=True) # of shape 1,N,3
            points, normals = points[0], normals[0]
            result['points'] = points
            result['normals'] = normals
            result['valid_mesh'] = 1
        else:
            result['valid_mesh'] = 0
        if save_mesh and result['valid_mesh']>0:
            point_and_normal = torch.cat([points, normals], dim=1).detach().cpu().numpy() # N,6
            np.savetxt(os.path.join(save_dir, 'pcd.xyz'), point_and_normal)
    return result


if __name__ == '__main__':
    # f = open("code.py", "r")
    # content = f.read()
    # print(content)
    code = "import bpy\nfrom math import radians, pi\nfrom bpy_lib import *\n\ndelete_all()\n\ncreate_section_shape('rectangle_points', {'name': 'rectangle_points', 'points': [[0.0, 0.0, 0.0], [0.0, 0.15, 0.0], [0.21, 0.0, 0.0], [0.21, 0.15, 0.0]]})\ncreate_concatcurve_translation('translation', 'rectangle_points', control_points=[[-0.75, -0.01, -0.35], [-0.3, 0.02, 0.13], [0.69, -0.02, -0.06], [-0.91, 0.01, -0.07]], points_radius=[1, 1, 1, 1], fill_caps=False)"
    # write_code_to_file(code, 'generated_blender_code', 'test.py')
    # module_path = 'generated_blender_code'+'.'+ 'test'
    # file_path = 'generated_blender_code'
    # file_name = 'test.py'
    # code_module = importlib.import_module(module_path)
    # importlib.reload(code_module)
    # vertices, faces = code_module.obtain_mesh()
    # result = load_mesh(module_path, device='cuda', sample_points=True, num_points=16384, save_mesh=True, save_dir='vis')
    result = code_to_mesh(code, file_path=None, file_name=None, device='cuda', sample_points=True, num_points=16384, 
                save_mesh=True, save_dir='vis', execute_method='exec')
    pdb.set_trace()
