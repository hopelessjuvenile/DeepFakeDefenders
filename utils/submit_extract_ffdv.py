import os
from glob import glob
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.models as models
from torchvision import transforms
import cv2
from PIL import Image
from torch.nn.modules.container import Sequential
from typing import *
import moviepy.editor as mp
import librosa
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image
import gc
from utils.cov_transform import PerturbCovarianceTransform

os.environ["FFDV_TRAINING_ROOT_DIR"] = "/hy-tmp/multi-ffdv/data"


def min_max_normalize(cov_matrix):
    min_value = np.min(cov_matrix)
    max_value = np.max(cov_matrix)

    range_value = max_value - min_value

    if range_value != 0:
        normalized_matrix = (cov_matrix - min_value) / range_value
    else:
        normalized_matrix = np.zeros_like(cov_matrix)

    return normalized_matrix


def get_spect_transforms():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomApply([transforms.RandomHorizontalFlip()], p=0.5),
        transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_spect_transforms_test():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_PerturbCovarianceTransform():
    return PerturbCovarianceTransform(perturbation_level=0.01, prob=0.5)


def extract_video_feature(video_path: str = None, 
                          model: Sequential = None, 
                          device: torch.device = None) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    interval = 1
    frame_count = 0
    hist_vectors = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式
            hist_r = cv2.calcHist([frame_rgb], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([frame_rgb], [1], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([frame_rgb], [2], None, [256], [0, 256]).flatten()
            hist_vector = hist_r + hist_g + hist_b
            hist_vectors.append(hist_vector)
        frame_count += 1

    cap.release()
    
    hist_matrix = np.array(hist_vectors)
    cov_matrix = np.cov(hist_matrix, rowvar=False)
    top_k_indices = np.argsort(np.sum(cov_matrix, axis=1))[: 128]
    top_k_vectors = cov_matrix[top_k_indices, : 128]
    result = min_max_normalize(top_k_vectors)
    return result


def extract_audio_feature(audio_path: str = None,
                          video_path: str = None,
                          n_mels: int = 128):
    if not os.path.exists(audio_path):
        video = mp.VideoFileClip(video_path)
        try:
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
        except IndexError:
            print(f"The error video path is; {video_path}")
            print(f"Audio duration is: {video.audio.duration}")
        y, fs = librosa.load(audio_path)
    else:
        y, fs = librosa.load(audio_path)
    if os.path.isfile(audio_path):
        os.remove(audio_path)

    mel_spect = librosa.feature.melspectrogram(y=y, sr=fs, n_mels=n_mels)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

    mel_spect_normalized = cv2.normalize(mel_spect, None, 0, 255, cv2.NORM_MINMAX)
    mel_spect_normalized = mel_spect_normalized.astype(np.uint8)
    result = cv2.resize(mel_spect_normalized, (256, 256), interpolation=cv2.INTER_LINEAR)

    return result


def get_feature(root_split_path: str = None,
                video_name: str = None,
                mode: str = None, 
                device: torch.device = None) -> Dict[str, Union[Tensor, Any]]:
    temp_wav = os.path.join(root_split_path, "wav_file")
    try:
        if not os.path.exists(temp_wav):
            os.mkdir(temp_wav)
    except FileExistsError:
        pass
    video_path = os.path.join(root_split_path, video_name)
    audio_path = os.path.join(temp_wav, video_name + ".wav")

    video_feature, audio_feature = extract_feature(video_path, audio_path, device)

    audio_feature = Image.fromarray(audio_feature).convert('RGB')
    video_feature = torch.tensor(video_feature)

    if mode == "train":
        cov_transform = get_PerturbCovarianceTransform()
        spect_transform = get_spect_transforms()
        video_feature = cov_transform(video_feature)
    else:
        spect_transform = get_spect_transforms_test()
    audio_feature = spect_transform(audio_feature)
    return {"video": video_feature, "audio": audio_feature}


def extract_feature(extract_path: str = None,
                    audio_path: str = None, 
                    device: torch.device = None):

    video_feature = extract_video_feature(extract_path, device)
    audio_feature = extract_audio_feature(audio_path, extract_path)

    return video_feature, audio_feature


if __name__ == '__main__':
    root_path = os.getenv("FFDV_TRAINING_ROOT_DIR")
    gt_filename_list = {"trainset": "trainset_label.txt", "valset": "valset_label.txt", "testset": "testset_label.csv"}
    # gt_filename_list = {"testset": "prediction.txt_p2.csv"}
    for split in gt_filename_list.keys():
        split_path = os.path.join(root_path, split)
        feature_path = os.path.join(root_path, split + "_raw")
        if not os.path.exists(feature_path):
            os.mkdir(feature_path)
        video_list = os.listdir(split_path)
        for video in tqdm(video_list, total=len(video_list), desc=f"[{split}]:"):
            video_path = os.path.join(split_path, video)
            extract_feature(video_path, feature_path, video, video.split('.')[0] + ".wav")
    # result = get_feature("D:\\multi-ffdv", "f0147d577a20db83c5f636840f1c31a1.mp4", "f0147d577a20db83c5f636840f1c31a1.wav")
    # result = extract_audio_feature("../test.wav", "../test.mp4")
    # mask = result["audio"] != 1.0
    # indices = torch.nonzero(mask)
    # print(indices.shape)
    # no_one = result["audio"][indices[:, 0], indices[:, 1], indices[:, 2]]
    # print(result["video"].shape, result["audio"].shape)
