import os
import argparse
import numpy as np
import cv2
import torch
import torchvision
import torchaudio
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
from torchvision import transforms
import glob
from config import cfg
from model.model_utils import AudioVisualSaliencyModel as SalModel
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    return img


def get_audio_feature(audio_path, start_idx, fps, len_snippet=32, mode=False, num_frames=None):
    # print(audio_path)
    spectro_shape = (257, 111)
    if os.path.exists(audio_path):
        audio, sr = torchaudio.load(audio_path)
        mm = 16000
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=mm)(audio)
        if audio.shape[0] == 2:
            audio = torch.mean(audio, dim=0).unsqueeze(0)
        if num_frames is not None:
            mm = audio.shape[-1]
            start = int(np.round((start_idx / num_frames * mm)))
            end = int(np.round(((start_idx + len_snippet + 1) / num_frames * mm)))
        else:
            start = int(np.round((start_idx / float(fps)) * mm))
            end = int(np.round(((start_idx + len_snippet + 1) / float(fps)) * mm))
        # print(start, end)
        audio = audio[:, start: end]
        if mode:
            audio = torch.flip(audio, [1])
        audio = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=160, )(audio)
        audio = torch.log(audio + 1e-6)

        # stardard normalize
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


def blur(img):
    k_size = 11
    bl = cv2.GaussianBlur(img, (k_size, k_size), 0)
    return bl


@torch.no_grad()
def process(model, frames, frame_idx, vname, img_size, audio_feature=None, args=None, labels=None):
    if args.use_sound:
        frames = frames.to(device, non_blocking=True)
        audio_feature = audio_feature.to(device, non_blocking=True)
        if labels is None:
            pred_map = model(frames, audio_feature)[0].detach().cpu().numpy()[0]
        else:
            pred_map = model(frames, audio_feature, labels=labels)[0].detach().cpu().numpy()[0]
    else:
        pred_map = model(frames).detach().cpu().numpy()[0]
        # print(pred_map.shape)

    pred_map = blur(pred_map)
    pred_map = np.exp(pred_map)
    pred_map = cv2.resize(pred_map, img_size)
    pred_map = normalize(pred_map)
    pred_map = np.round(pred_map * 255).astype(np.uint8)
    os.makedirs(os.path.join(args.save_path, vname), exist_ok=True)
    cv2.imwrite(os.path.join(args.save_path, vname, frame_idx), pred_map)


def inference_dataset(model, args):
    len_temporal = args.clip_size

    # Select dataset
    if args.dataset == 'DIEM':
        file_name = 'DIEM_list_test_fps.txt'
    else:
        file_name = '{}_list_test_{}_fps.txt'.format(args.dataset, args.split)

    list_data = []
    videos_fps = {}
    with open(os.path.join(args.path_data, 'fold_lists', file_name), 'r') as f:
        for line in f.readlines():
            name, frame_num, fps = line.split(' ')
            list_data.append(name)
            videos_fps[name] = fps
    list_data.sort()
    print(list_data)

    for vname in list_data:
        print("Processing: " + vname)
        audio_path = os.path.join(args.path_data, 'video_audio', args.dataset, vname, vname + ".wav")
        list_frames = glob.glob(os.path.join(args.path_data, 'video_frames', args.dataset, vname, "*.jpg"))
        list_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0].split('_')[1]))
        os.makedirs(os.path.join(args.save_path, vname), exist_ok=True)

        # process in a sliding window fashion
        if len(list_frames) >= 2 * len_temporal - 1:

            snippet = []
            # snippet_one = []
            for i in tqdm(range(len(list_frames))):
                img_tensor, img_size = torch_transform(list_frames[i])
                img_size = (640, 480)

                snippet.append(img_tensor)

                if i >= len_temporal - 1:
                    clip = torch.FloatTensor(torch.stack(snippet)).unsqueeze(0).to(device="cuda")
                    clip = clip.permute(0, 2, 1, 3, 4)

                    process(model, clip, os.path.basename(list_frames[i]), vname, img_size,
                            audio_feature=get_audio_feature(audio_path=audio_path,
                                                            start_idx=i - len_temporal + 1,
                                                            fps=videos_fps[vname]).unsqueeze(0),
                            args=args)

                    # process first (len_temporal-1) frames
                    if i < 2 * len_temporal - 2:
                        audio_fea = get_audio_feature(audio_path=audio_path,
                                                      start_idx=i - len_temporal + 1, fps=videos_fps[vname],
                                                      mode=True).unsqueeze(0)
                        process(model, torch.flip(clip, [2]), os.path.basename(list_frames[i - len_temporal + 1]),
                                vname, img_size,
                                audio_feature=audio_fea,
                                args=args)
                    del snippet[0]
        else:
            print('More frames are needed')

def torch_transform(path):
    img_transform = transforms.Compose([
        transforms.Resize((224, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            IMAGENET_DEFAULT_MEAN,
            IMAGENET_DEFAULT_STD, )
    ])
    img = Image.open(path).convert('RGB')
    sz = img.size
    img = img_transform(img)
    return img, sz

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight',
                        default="./output/mvitv2_small_224_384_16_s2.pt",
                        type=str)
    parser.add_argument('--save_path', default='./output', type=str)
    parser.add_argument('--split', default=2, type=int)
    parser.add_argument('--path_data', default='./AuViDataset', type=str)
    parser.add_argument('--dataset', default='AVAD', type=str)
    parser.add_argument('--clip_size', default=16, type=int)
    parser.add_argument('--use_sound', default=True, type=bool)

    args = parser.parse_args()
    print(args)
    os.makedirs(args.save_path, exist_ok=True)

    len_temporal = args.clip_size

    model = SalModel(cfg=cfg)
    model.load_state_dict(torch.load(args.weight), strict=False)

    model = model.to(device)
    torch.backends.cudnn.benchmark = True
    model.eval()

    inference_dataset(model, args)