import numpy as np
import pdb
import os
from tqdm import tqdm

if __name__ == '__main__':
    folder = '/cpfs01/user/lvzhaoyang/topology_generation/llama-recipes_gitee/src/llama_recipes/datasets/shape2code_dataset/data/train/outputs'
    files = os.listdir(folder)
    files = sorted(files)
    maxs = []
    mins = []
    for f in tqdm(files):
        path = os.path.join(folder, f)
        data = np.load(path)
        maxs.append(data['samples'].max())
        mins.append(data['samples'].min())
    pdb.set_trace()