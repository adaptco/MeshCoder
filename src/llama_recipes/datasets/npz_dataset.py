import torch
import numpy as np
import os
import torch.distributed as dist
import pdb

class ShapeNpzDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, scale=1, noise_magnitude=0.025, rank=0, world_size=1):
        
        data = np.load(data_dir)
        self.input_data = data['points'] # B,npoints,3
        self.labels = data['label']
        data.close()

        self.noise_magnitude = noise_magnitude
        self.scale = scale
        
        if world_size > 1:
            num_samples_per_rank = int(np.ceil(self.input_data.shape[0] / world_size))
            start = rank * num_samples_per_rank
            end = (rank+1) * num_samples_per_rank
            self.input_data = self.input_data[start:end]
            self.labels = self.labels[start:end]
            self.num_samples_per_rank = num_samples_per_rank
        else:
            self.num_samples_per_rank = self.input_data.shape[0]
        
        self.points = self.input_data[:,:,0:3] / 2 / scale
        self.normals = self.input_data[:,:,3:]
        print('dataset %s:' % data_dir)
        print('points:', self.points.shape)
        print('normals:', self.normals.shape)
        print('labels:', self.labels.shape)
        self.len = self.points.shape[0]

        self.input_data = None

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        points = self.points[index]
        normals = self.normals[index]
        label = self.labels[index]
        # label = 0
        if self.noise_magnitude > 0:
            points = points + self.noise_magnitude * np.random.randn(*points.shape).astype(np.float32)
            normals = normals + self.noise_magnitude * np.random.randn(*normals.shape).astype(np.float32)
            # in this case, the normals are not normalized, we may need to normalize it after it is gathered into cuda tensors 
        
        points = points * self.scale * 2 # now it roughly range from -scale to scale

        data = {}
        data['points'] = points 
        data['normals'] = normals 
        data['label'] = label
        return data

class GeneralNpzDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, scale=1, noise_magnitude=0, rank=0, local_rank=0, world_size=1,
                        cache_dir = None, cluster='aliyun',
                        data_key='points', data_key_split_names=None, data_key_split_dims=None,
                        need_scale_keys = ['points'], need_add_noise_keys = ['points', 'normals']):
        
        # an example to split data_key is
        # data_key_split_names = ['points', 'normals']
        # data_key_split_dims = [0,3,6]

        self.need_scale_keys = need_scale_keys
        self.need_add_noise_keys = need_add_noise_keys
        ossutil_format = "ossutil" if cluster == "aliyun" else "~/ossutil"

        if data_dir.startswith('oss://'):
            assert not cache_dir is None
            data_folder, filename = os.path.split(data_dir)
            target_file = os.path.join(cache_dir, filename)
            if rank == 0 or world_size == 1:
                print('copy the original dataset from %s to %s' % (data_dir, target_file))
                # endpoint_dict = {'aliyun':'-e http://oss-cn-wulanchabu-internal.aliyuncs.com', 's_cluster':''} Environment variable 'TORCHELASTIC_HEALTH_CHECK_PORT' not found. Do not start health check.
                endpoint_dict = {'aliyun_external':'-e http://oss-cn-wulanchabu.aliyuncs.com', 'aliyun':'-e http://oss-cn-wulanchabu-internal.aliyuncs.com', 's_cluster':''}
                # command = '%s cp %s %s %s > ossutil.txt' % (ossutil_format, data_dir, target_file, endpoint_dict[cluster])   # ~/.ossutil to ossutil
                command = '%s cp %s %s %s' % (ossutil_format, data_dir, target_file, endpoint_dict[cluster])
                print(f"command: {command}")
                os.system(command)
            if world_size > 1:
                print('rank %d enter npz dataset copy barrier' % rank)
                dist.barrier(device_ids=[local_rank])
                # it seems that dist.barrier only block the most close cuda operation, 
                # following cpu operations are not blocked
                torch.rand(2,4).to('cuda:%d' % local_rank)
                print('rank %d ends npz dataset copy barrier' % rank)
            # start reading files
            data = np.load(target_file, allow_pickle=True)
            # max_retries = 3
            # retry_count = 0
            # while not os.path.exists(target_file):
            #     if (rank == 0 or world_size == 1) and retry_count < max_retries:
            #         endpoint_dict = {'aliyun_external':'-e http://oss-cn-wulanchabu.aliyuncs.com', 'aliyun':'-e http://oss-cn-wulanchabu-internal.aliyuncs.com', 's_cluster':''}
            #         command = '%s cp %s %s %s > ossutil.txt' % (ossutil_format, data_dir, target_file, endpoint_dict[cluster])   # ~/.ossutil to ossutil
            #         print(f"command: {command}")
            #         # 执行命令并检查返回值
            #         ret_code = os.system(command)
            #         print(f"RETURN CODE: {ret_code}")
            #         if ret_code == 0:
            #             break
            #         retry_count += 1
            #     if retry_count == max_retries and not os.path.exists(target_file):
            #         print("Retry times till max reties")
            #         break
            #     if world_size > 1:
            #         print('rank %d enter npz dataset read retry barrier' % rank)
            #         dist.barrier(device_ids=[local_rank])
            #         print('rank %d ends npz dataset read retry barrier' % rank)
            # data = np.load(target_file)
            if world_size > 1:
                print('rank %d enter npz dataset read barrier' % rank)
                dist.barrier(device_ids=[local_rank])
                torch.rand(2,4).to('cuda:%d' % local_rank)
                print('rank %d ends npz dataset read barrier' % rank)
            # if rank == 0 or world_size==1:
            #     os.remove(target_file)
        else:
            data = np.load(data_dir)
        self.data_dict = {}
        # we assume all keys are same length numpy arrays
        for name in data._files:
            name = os.path.splitext(name)[0]
            if name == data_key and data_key_split_names is not None:
                # we split the last dim of data[data_key]
                for i in range(len(data_key_split_names)):
                    start_idx = data_key_split_dims[i]
                    end_idx = data_key_split_dims[i+1]
                    self.data_dict[data_key_split_names[i]] = data[data_key][...,start_idx:end_idx]
                data_key = data_key_split_names[0]
                # later we use data_key to determine the length of the dataset
            else:
                try:
                    self.data_dict[name] = data[name]
                except:
                    print(data)
                    print(name)
                    import traceback; print(traceback.print_exc())
        data.close()

        self.noise_magnitude = noise_magnitude
        self.scale = scale
        
        num_samples = self.data_dict[data_key].shape[0]
        if world_size > 1:
            num_samples_per_rank = int(np.ceil(num_samples / world_size))
            start = rank * num_samples_per_rank
            end = (rank+1) * num_samples_per_rank
            for key in self.data_dict.keys():
                self.data_dict[key] = self.data_dict[key][start:end]
            self.num_samples_per_rank = num_samples_per_rank
        else:
            self.num_samples_per_rank = num_samples
        
        for key in self.need_scale_keys:
            if key in self.data_dict.keys():
                self.data_dict[key] = self.data_dict[key] * scale

        print('dataset %s:' % data_dir)
        for key in self.data_dict.keys():
            print(key, ':', self.data_dict[key].shape)

        self.len = self.data_dict[data_key].shape[0]


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        result_dict = {}
        for key in self.data_dict.keys():
            result_dict[key] = self.data_dict[key][index]
        
        if self.noise_magnitude > 0:
            for key in self.need_add_noise_keys:
                if key in result_dict.keys():
                    result_dict[key] = (result_dict[key] + 
                            self.noise_magnitude * np.random.randn(*result_dict[key].shape).astype(result_dict[key].dtype))
        
        return result_dict

def get_shape(x):
    try:
        return x.shape
    except:
        return len(x)

if __name__ == '__main__':
    # import pdb
    # path='../visualization_tools/shapenet_psr_generated_data_2048_pts_epoch_0800_iter_767199.npz'
    # dataset = ShapeNpzDataset(path, scale=1, noise_magnitude=0.025, rank=3, world_size=4)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # for i, data in enumerate(dataloader):
    #     print('Progress [%d|%d] % .3f' % (i, len(dataloader), i/len(dataloader)), 
    #             data['points'].shape, data['normals'].shape, data['label'].shape)
    #     pdb.set_trace()

    # path='../exps/exp_shapenet_psr_generation/ddpm_keypoint_training_exps/T1000_betaT0.02_shapenet_psr_keypoint_generation_batchsize_32_with_ema_centered_to_centroid/eval_result/model_ema_0.99990/shapenet_psr_generated_data_16_pts_epoch_1000_iter_958999.npz'
    path = '/cpfs01/user/lvzhaoyang/mesh/slide_sdf/exps/exp_shapenet_psr_generation/position_ddpm_training_exps/64_128_keypoints/T1000_betaT0.02_shapenetv2_128_256_keypoint_generation_batchsize_128_depth_12_sequence_length_as_condition_class_conditional/eval_result/model_ema_0.99900/generated_data_256_pts_epoch_4000_iter_1147999.npz'
    dataset = GeneralNpzDataset(path, scale=1, noise_magnitude=0, rank=3, world_size=4, data_key='points', data_key_split_names=None, data_key_split_dims=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)
    
    for i, data in enumerate(dataloader):
        print('Progress [%d|%d] % .3f' % (i, len(dataloader), i/len(dataloader)))
        shape_info = [key+': '+str(get_shape(data[key])) for key in data.keys()]
        print(' '.join(shape_info))
        pdb.set_trace()