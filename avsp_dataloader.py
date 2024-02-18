import os
import random
import warnings
import numpy as np
import torch
import torchaudio
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import ImageFile
import scipy.io


def resize_fixation(image, row, col):
    resized_fixation = np.zeros((row, col))
    ratio_row = row / image.shape[0]
    ratio_col = col / image.shape[1]

    coords = np.argwhere(image)
    for coord in coords:
        coord_r = int(np.round(coord[0] * ratio_row))
        coord_c = int(np.round(coord[1] * ratio_col))
        if coord_r == row:
            coord_r -= 1
        if coord_c == col:
            coord_c -= 1
        resized_fixation[coord_r, coord_c] = 1

    return resized_fixation


def generate_fixation(coords, row=128, col=128, org_row=480, org_col=640):
    resize_fixation = np.zeros((row, col))
    ratio_row = row / org_row
    ratio_col = col / org_col

    for coord in coords:
        coord_r = int(np.round(coord[0] * ratio_row))
        coord_c = int(np.round(coord[1] * ratio_col))
        if coord_r == row:
            coord_r -= 1
        if coord_c == col:
            coord_c -= 1
        resize_fixation[coord_r, coord_c] = 1

    return resize_fixation


def get_audio_spectrogram(audio_path, start_idx, videos_fps, len_snippet=16, sample_rate=16000, spectro_shape=(257,111)):
    if os.path.exists(audio_path):
        audio, sr = torchaudio.load(audio_path)
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(audio)
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0).unsqueeze(0)

        start = int(np.round((start_idx / float(videos_fps)) * sample_rate))
        end = int(np.round(((start_idx + len_snippet + 1) / float(videos_fps)) * sample_rate))

        audio = audio[:, start: end]
        audio = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160, )(audio)
        audio = torch.log(audio + 1e-6)

        means = audio.mean(dim=1, keepdim=True)
        stds = audio.std(dim=1, keepdim=True)
        aud = (audio - means) / (stds + 1e-6)

        tmp = torch.zeros(1, spectro_shape[0], spectro_shape[1]) + 0.02

        if audio.shape[-1] <= spectro_shape[1]:
            tmp[:, :, :audio.shape[-1]] = aud
            aud = tmp
        else:
            tmp = aud[:, :, :spectro_shape[1]]
            aud = tmp

    else:
        aud = torch.zeros((1, spectro_shape[0], spectro_shape[1])) + 0.02
    return aud


class AudioVisualDataset(Dataset):
    def __init__(self, data_root, dataset_name='DIEM', split=1, len_clip=32, mode='train', use_sound=True,
                 size=(224, 224)):
        self.path_data = data_root
        self.use_sound = use_sound
        self.mode = mode
        self.len_snippet = len_clip
        self.size = size
        self.img_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor(),
            transforms.Normalize(
                IMAGENET_DEFAULT_MEAN,
                IMAGENET_DEFAULT_STD,
            )
        ])
        self.list_num_frame = []
        self.dataset_name = dataset_name
        if dataset_name == 'DIEM':
            file_name = 'DIEM_list_{}_fps.txt'.format(mode)
        else:
            file_name = '{}_list_{}_{}_fps.txt'.format(dataset_name, mode, split)

        self.videos_names = []
        self.videos_fps = {}
        self.videos_frame_num = {}
        self.list_indata = []
        with open(os.path.join(self.path_data, 'fold_lists', file_name), 'r') as f:
            for line in f.readlines():
                name, frame_num, fps = line.split(' ')
                self.list_indata.append(name)
                self.videos_names.append(name)
                self.videos_frame_num[name] = frame_num
                self.videos_fps[name] = fps

        self.list_indata.sort()

        if self.mode == 'train':

            self.list_num_frame = [len(os.listdir(os.path.join(self.path_data, 'annotations', dataset_name, v, 'maps')))
                                   for v in self.list_indata]

        elif self.mode == 'test' or self.mode == 'val':
            print("val set")
            for v in self.list_indata:
                frames = os.listdir(os.path.join(self.path_data, 'annotations', dataset_name, v, 'maps'))
                frames.sort()
                for i in range(0, len(frames) - self.len_snippet, 2 * self.len_snippet):
                    if self.check_frame(os.path.join(self.path_data, 'annotations', dataset_name, v, 'maps',
                                                     'eyeMap_%05d.jpg' % (i + self.len_snippet))):
                        self.list_num_frame.append((v, i))

        print(self.mode, len(self.list_indata))

    def check_frame(self, path):
        img = cv2.imread(path, 0)
        return img.max() != 0

    def __len__(self):
        return len(self.list_num_frame)

    def __getitem__(self, idx):
        # print(self.mode)
        if self.mode == "train":
            video_name = self.list_indata[idx]
            while 1:
                start_idx = np.random.randint(0, self.list_num_frame[idx] - self.len_snippet + 1)
                if self.check_frame(os.path.join(self.path_data, 'annotations', self.dataset_name, video_name, 'maps',
                                                 'eyeMap_%05d.jpg' % (start_idx + self.len_snippet))):
                    break
                else:
                    print("No saliency defined in train dataset")
        elif self.mode == "test" or self.mode == "val":
            (video_name, start_idx) = self.list_num_frame[idx]

        path_clip = os.path.join(self.path_data, 'video_frames', self.dataset_name, video_name)
        path_annt = os.path.join(self.path_data, 'annotations', self.dataset_name, video_name, 'maps')
        path_fix = os.path.join(self.path_data, 'annotations', self.dataset_name, video_name)

        clip_img = []

        for i in range(self.len_snippet):
            img = Image.open(os.path.join(path_clip, 'img_%05d.jpg' % (start_idx + i + 1))).convert('RGB')
            sz = img.size
            clip_img.append(self.img_transform(img))

        clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0)).permute(1, 0, 2, 3)
        gt = np.array(
            Image.open(os.path.join(path_annt, 'eyeMap_%05d.jpg' % (start_idx + self.len_snippet))).convert('L'))
        gt = gt.astype('float')

        if self.mode == "train":

            gt = cv2.resize(gt, (self.size[1], self.size[0]))
        else:
            gt = cv2.resize(gt, (self.size[1], self.size[0]))

        if np.max(gt) > 1.0:
            gt = gt / 255.0
        assert gt.max() != 0, (start_idx, video_name)

        fix = np.array(
            scipy.io.loadmat(os.path.join(path_fix, 'fixMap_%05d.mat' % (start_idx + self.len_snippet)))['eyeMap'])
        fix = resize_fixation(fix, row=224, col=384)

        if self.use_sound:
            audio_path = os.path.join(self.path_data, 'video_audio', self.dataset_name, video_name, video_name + ".wav")
            aud = get_audio_spectrogram(audio_path, start_idx, self.videos_fps[video_name],
                                        len_snippet=self.len_snippet, sample_rate=16000)
            return clip_img, aud, torch.FloatTensor(gt)
        return clip_img, gt
