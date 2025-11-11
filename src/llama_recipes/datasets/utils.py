import os
import numpy as np

from llama_recipes.datasets.data_collator import pad_tensor
import pdb

def xyz_pcd_files_to_npz(folder, save_path):
    file_list = os.listdir(folder)
    pcd_list = []
    for f in file_list:
        data = np.loadtxt(os.path.join(folder, f))
        pcd_list.append(data)
    points, points_mask = pad_tensor(pcd_list)
    points, points_mask = points.detach().cpu().numpy(), points_mask.detach().cpu().numpy()
    save_folder = os.path.split(save_path)[0]
    os.makedirs(save_folder, exist_ok=True)
    np.savez(save_path, points=points, points_mask=points_mask)
    # pdb.set_trace()

if __name__ == '__main__':
    folder = '/cpfs01/user/lvzhaoyang/topology_generation/llama-recipes_gitee/recipes/inference/local_inference/custom_inference/recontruction_exps/exp4/original_xyz_files'
    save_path = '/cpfs01/user/lvzhaoyang/topology_generation/llama-recipes_gitee/recipes/inference/local_inference/custom_inference/recontruction_exps/exp4/pcd.npz'
    xyz_pcd_files_to_npz(folder, save_path)