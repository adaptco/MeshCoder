try:
    from petrel_client.client import Client
except:
    print('Petrel oss client is not installed')
import numpy as np
import io
from PIL import Image
import cv2
import torch
import time
import pdb

def path_mapping(path):
    return path

class Image_OSS_IO():
    def __init__(self, conf_path = '~/petreloss.conf', client=None, path_mapping=path_mapping, disable_client=False):
        # we can explicitly disable_client, in this case, we need to make sure only file system path is used
        if client is None and not disable_client:
            self.client = Client(conf_path)
        else:
            self.client = client

        self.path_mapping = path_mapping
        return

    def read(self, path, return_numpy_array=True, read_method='pil'):
        # path is the path to an image file
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            img_bytes = self.client.get(path)
            if read_method=='pil':
                image = Image.open(io.BytesIO(img_bytes))
            else:
                # pdb.set_trace()
                nparr = np.fromstring(img_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_ANYDEPTH)
                # image = cv2.imread(io.BytesIO(img_bytes), cv2.IMREAD_ANYDEPTH)
        else:
            if read_method=='pil':
                image = Image.open(path)
            else:
                image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if return_numpy_array and not isinstance(image, np.ndarray):
            image = np.array(image)
        return image

    def write(self, image, path, format='png'):
        # image is an numpy arry of shape H,W,3, dtype unit 8
        # path is a normal path or ceph url, e.g., 's3://ZylyuBucket/folder/0999_label_0_array.png'
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            with io.BytesIO() as f:
                Image.fromarray(image).save(f, format=format)
                self.client.put(path, f.getvalue())
        else:
            Image.fromarray(image).save(path)

class Npz_OSS_IO():
    def __init__(self, conf_path = '~/petreloss.conf', client=None, path_mapping=path_mapping, disable_client=False):
        if client is None and not disable_client:
            self.client = Client(conf_path)
        else:
            self.client = client

        self.path_mapping = path_mapping
        return 

    def read(self, path, update_cache=False):
        # path is the path to an .npz file
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            # time1 = time.time()
            bbytes = io.BytesIO(self.client.get(path, update_cache=update_cache))
            # time2 = time.time()
            data = np.load(bbytes, allow_pickle=True)
            # time3 = time.time()
            # print('read bytes time %.4f load numpy time %.4f total ceph read time %.4f' % (time2-time1, time3-time2, time3-time1))
        else:
            data = np.load(path)
        return data

    def write(self, path, *args, **kwds):
        # path is a normal path or ceph url, e.g., 's3://ZylyuBucket/folder/0999_label_0_array.png'
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            with io.BytesIO() as f:
                np.savez(f, *args, **kwds)
                self.client.put(path, f.getvalue())
        else:
            np.savez(path, *args, **kwds)

class Torch_OSS_IO():
    def __init__(self, conf_path = '~/petreloss.conf', client=None, path_mapping=path_mapping, disable_client=False):
        if client is None and not disable_client:
            self.client = Client(conf_path)
        else:
            self.client = client

        self.path_mapping = path_mapping
        return

    def read(self, path, map_location='cpu'):
        # path is the path to an .npz file
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            # checkpoint = torch.load(io.BytesIO(self.client.get(path)), map_location=map_location)
            with io.BytesIO(self.client.get(path)) as f:
                checkpoint = torch.load(f, map_location=map_location)
        else:
            checkpoint = torch.load(path, map_location=map_location)
        return checkpoint

    def write(self, path, checkpoint):
        # path is a normal path or ceph url, e.g., 's3://ZylyuBucket/folder/0999_label_0_array.png'
        path = self.path_mapping(path) 
        # we may want to make some modifications to the input path, e.g., transform a file sys path to an s3 oss path
        if path.startswith('s3://'):
            with io.BytesIO() as f:
                torch.save(checkpoint, f)
                self.client.put(path, f.getvalue())
        else:
            torch.save(checkpoint, path)

