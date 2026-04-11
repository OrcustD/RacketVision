import os
from typing import List
import parse
import torch
import numpy as np
import json
import pandas as pd
from torch.utils.data import Dataset
import cv2
from tqdm import tqdm
from mmengine.registry import DATASETS


@DATASETS.register_module(name='Uball')
class UniBallDataset(Dataset):
    def __init__(self,
                 root_dir='data/tabletennis',
                 split='train',
                 seq_len=8,
                 sliding_step=1,
                 data_mode='heatmap',
                 bg_mode='',
                 height=288,
                 width=512,
                 debug=False,
                 magnitude=1,
                 sigma=2.5,
                 img_format='jpg',
                 heatmap_mode='gaussian',
                 first_frame='0000'):
        self.metainfo = {}
        if os.path.exists(f'{root_dir}/info/metainfo.json'):
            self.metainfo = json.load(open(f'{root_dir}/info/metainfo.json'))

        self.HEIGHT, self.WIDTH = height, width
        self.img_format = img_format

        self.heatmap_mode = heatmap_mode
        self.mag = 1
        self.sigma = sigma

        self.root_dir = root_dir
        self.split = split
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data_mode = data_mode
        self.bg_mode = bg_mode
        self.first_frame = first_frame
        self.magnitude = magnitude

        self.rally_dict = self.get_rally_dict()
        self.img_config = self.gen_rally_img_config_file()
        self.data_dict = self.gen_input_file()
        if debug:
            self.data_dict = {k: v[:256] for k, v in self.data_dict.items()}

    def full_init(self):
        pass

    def get_rally_dict(self):
        datalist = json.load(open(os.path.join(self.root_dir, 'info', f'{self.split}.json')))
        rally_dirs = [os.path.join(self.root_dir, 'all', match, 'frame', rally) for match, rally in datalist]
        rally_dict = {'i2p': {i: rally_dir for i, rally_dir in enumerate(rally_dirs)},
                      'p2i': {rally_dir: i for i, rally_dir in enumerate(rally_dirs)}}
        return rally_dict

    def gen_rally_img_config_file(self):
        if 'image_shape' in self.metainfo:
            num_rally = len(self.rally_dict['i2p'])
            h, w, c = self.metainfo['image_shape']
            w_scaler, h_scaler = w / self.WIDTH, h / self.HEIGHT
            img_scaler = [(w_scaler, h_scaler)] * num_rally
            img_shape = [(w, h)] * num_rally
        else:
            img_scaler = []
            img_shape = []
            for rally_i, rally_dir in tqdm(self.rally_dict['i2p'].items()):
                h, w, c = cv2.imread(os.path.join(rally_dir, f'{self.first_frame}.{self.img_format}')).shape
                w_scaler, h_scaler = w / self.WIDTH, h / self.HEIGHT
                img_scaler.append((w_scaler, h_scaler))
                img_shape.append((w, h))
        return dict(img_scaler=img_scaler, img_shape=img_shape)

    def _get_rally_i(self, rally_dir):
        if rally_dir not in self.rally_dict['p2i'].keys():
            return None
        else:
            return self.rally_dict['p2i'][rally_dir]

    def gen_input_file(self):
        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        frame_file = np.array([]).reshape(0, self.seq_len)
        coor = np.array([], dtype=np.float32).reshape(0, self.seq_len, 2).reshape(0, 1, 2)
        vis = np.array([], dtype=np.float32).reshape(0, self.seq_len).reshape(0, 1)
        for _, rally_dir in self.rally_dict['i2p'].items():
            data_dict = self.gen_input_from_rally_dir(rally_dir)
            id = np.concatenate((id, data_dict['id']), axis=0)
            frame_file = np.concatenate((frame_file, data_dict['frame_file']), axis=0)
            coor = np.concatenate((coor, data_dict['coor']), axis=0)
            vis = np.concatenate((vis, data_dict['vis']), axis=0)
        return dict(id=id, frame_file=frame_file, coor=coor, vis=vis)

    def gen_input_from_rally_dir(self, rally_dir):
        rally_i = self._get_rally_i(rally_dir)

        file_format_str = os.path.join('{}', 'frame', '{}')
        match_dir, rally_id = parse.parse(file_format_str, rally_dir)

        csv_file = os.path.join(match_dir, 'csv', f'{rally_id}_ball.csv')
        assert os.path.exists(csv_file), f'{csv_file} does not exist.'
        label_df = pd.read_csv(csv_file, encoding='utf8').sort_values(by='Frame').fillna(0)
        fids, x, y, v = np.array(label_df['Frame']), np.array(label_df['X']), np.array(label_df['Y']), np.array(label_df['Visibility'])

        id = np.array([], dtype=np.int32).reshape(0, self.seq_len, 2)
        frame_file = np.array([]).reshape(0, self.seq_len)
        coor = np.array([], dtype=np.float32).reshape(0, 1, 2)
        vis = np.array([], dtype=np.float32).reshape(0, 1)

        for i, fid in enumerate(fids):
            tmp_idx, tmp_frames = [None] * self.seq_len, [None] * self.seq_len
            tmp_coor, tmp_vis = [], []
            seq_i = self.seq_len - 1
            curr_i = fid
            for _ in range(self.seq_len):
                curr_i = max(0, curr_i)
                tmp_idx[seq_i] = (rally_i, curr_i)
                tmp_frames[seq_i] = os.path.join(rally_dir, f'{curr_i:04d}.{self.img_format}')
                curr_i -= self.sliding_step
                seq_i -= 1
            tmp_coor.append((x[i], y[i]))
            tmp_vis.append(v[i])

            id = np.concatenate((id, [tmp_idx]), axis=0)
            frame_file = np.concatenate((frame_file, [tmp_frames]), axis=0)
            coor = np.concatenate((coor, [tmp_coor]), axis=0)
            vis = np.concatenate((vis, [tmp_vis]), axis=0)

        return dict(id=id, frame_file=frame_file, coor=coor, vis=vis)

    def get_heatmap(self, cx, cy):
        if cx == cy == 0:
            return np.zeros((1, self.HEIGHT, self.WIDTH))
        if self.heatmap_mode == 'hard':
            x, y = np.meshgrid(np.linspace(1, self.WIDTH, self.WIDTH), np.linspace(1, self.HEIGHT, self.HEIGHT))
            heatmap = ((y - (cy + 1)) ** 2) + ((x - (cx + 1)) ** 2)
            heatmap[heatmap <= self.sigma ** 2] = 1.
            heatmap[heatmap > self.sigma ** 2] = 0.
            heatmap = heatmap * self.magnitude
        elif self.heatmap_mode == 'gaussian':
            x = np.arange(self.WIDTH)
            y = np.arange(self.HEIGHT)
            xx, yy = np.meshgrid(x, y)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            heatmap = np.exp(-0.5 * (dist / self.sigma) ** 2)
            if np.max(heatmap) > 0:
                heatmap = heatmap / np.max(heatmap)
        elif self.heatmap_mode == 'binary_gaussian':
            x = np.arange(self.WIDTH)
            y = np.arange(self.HEIGHT)
            xx, yy = np.meshgrid(x, y)
            dist = (xx - cx) ** 2 + (yy - cy) ** 2
            heatmap = np.zeros_like(dist)
            heatmap = np.clip(self.magnitude * np.exp(-dist / self.sigma ** 2) / np.exp(-1), 0.0, 1.0)
            heatmap[heatmap < 0.5] = 0.0
        else:
            raise NotImplementedError
        heatmap = heatmap * self.mag
        return heatmap.reshape(1, self.HEIGHT, self.WIDTH)

    def __len__(self):
        return len(self.data_dict['id'])

    def __getitem__(self, idx):
        data_idx = self.data_dict['id'][idx]
        frame_file = self.data_dict['frame_file'][idx]
        coor = self.data_dict['coor'][idx]
        vis = self.data_dict['vis'][idx]
        w, h = self.img_config['img_shape'][data_idx[0][0]]
        w_scaler, h_scaler = self.img_config['img_scaler'][data_idx[0][0]]

        if len(self.bg_mode) > 0:
            file_format_str = os.path.join('{}', 'frame', '{}', '{}.'+self.img_format)
            match_dir, rally_id, _ = parse.parse(file_format_str, frame_file[0])
            median_file = os.path.join(match_dir, 'median.npz') if os.path.exists(os.path.join(match_dir, 'median.npz')) else os.path.join(match_dir, 'frame', rally_id, 'median.npz')
            assert os.path.exists(median_file), f'{median_file} does not exist.'
            median_img = np.load(median_file)['median']

        imgs = np.array([cv2.imread(f) for f in frame_file])
        if self.bg_mode == 'subtract':
            medians = np.array([median_img] * imgs.shape[0])
            imgs = np.sum(np.absolute(imgs - medians), 3).astype('float32')
        elif self.bg_mode == 'subtract_concat':
            medians = np.array([median_img] * imgs.shape[0])
            diff_imgs = np.sum(np.absolute(imgs - medians), 3).astype('float32')
            imgs = np.concatenate((imgs, diff_imgs), axis=3)
        else:
            imgs = imgs.astype('float32')

        imgs = np.array([cv2.resize(img, (self.WIDTH, self.HEIGHT)) for img in imgs])
        imgs = np.moveaxis(imgs, -1, 1)

        heatmap = np.array([self.get_heatmap(int(coor[i][0] / w_scaler), int(coor[i][1] / h_scaler)) for i in range(coor.shape[0])])
        heatmap = heatmap.reshape(-1, self.HEIGHT, self.WIDTH)
        frames = imgs

        if self.bg_mode == 'concat':
            median_img = cv2.resize(median_img, (self.WIDTH, self.HEIGHT))
            median_img = np.moveaxis(median_img, -1, 0).reshape(1, -1, self.HEIGHT, self.WIDTH)
            frames = np.concatenate((median_img, frames), axis=0)

        frames /= 255.
        coor[:, 0] = coor[:, 0] / w_scaler
        coor[:, 1] = coor[:, 1] / h_scaler
        frames = frames.reshape(-1, self.HEIGHT, self.WIDTH)

        sample = {
            'data_idx': torch.tensor(data_idx),
            'frames': torch.tensor(frames, dtype=torch.float32),
            'heatmaps': torch.tensor(heatmap, dtype=torch.float32),
            'coor': torch.tensor(coor, dtype=torch.float32),
            'vis': torch.tensor(vis, dtype=torch.float32)
        }
        return sample
