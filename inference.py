import os
import torch
import logging
from model import FFDV_Cov
from datetime import datetime
import argparse
from utils.submit_extract_ffdv import extract_video_feature, extract_audio_feature, get_spect_transforms_test
from PIL import Image
import glob


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

LOG_FORMAT = "Time:%(asctime)s - Level:%(levelname)s - Message:%(message)s"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(LOG_FORMAT)

file_handler = logging.FileHandler(f'log/{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}_FFDV.log')
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
file_handler.addFilter(lambda record: record.levelno == logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.INFO)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def logging_combine(logger_c: logging.Logger = None,
                    msg: str = None,
                    flag: bool = False) -> None:
    logger_c.debug(msg + "\n\n") if flag else logger_c.debug(msg)
    logger_c.info(msg + "\n\n") if flag else logger_c.info(msg)


def extract_feature(video_path, audio_path):
    video_feature = extract_video_feature(video_path)
    audio_feature = extract_audio_feature(audio_path, video_path)

    audio_feature = Image.fromarray(audio_feature)
    audio_feature = audio_feature.convert('RGB')
    video_feature = torch.tensor(video_feature)

    spect_transform = get_spect_transforms_test()
    audio_feature = spect_transform(audio_feature)

    return video_feature.unsqueeze(0), audio_feature.unsqueeze(0)


def test(args):
    pretrain_path = "pretrain_weight/resnet18_no_linear.pth"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FFDV_Cov(num_class=2, extra_model_path=pretrain_path, temporal=128, device=device, mode="inference")
    model = model.cuda(device=device)
    model.eval()

    model_dict = torch.load("checkpoint/best_fusion_cos.pth")

    last_epoch = model_dict["epoch"]
    model.load_state_dict(model_dict["model_state"])

    if args.video_path is not None:
        video_feature, audio_feature = extract_feature(args.video_path, os.path.join(os.path.dirname(args.video_path), "inference_video.wav"))

        logging_combine(logger, f"Success load {args.video_path}!!!")

        video_feature, audio_feature = video_feature.to(device), audio_feature.to(device)
        score = model(video_feature, audio_feature)
        logging_combine(logger, f"The {args.video_path} output score is: {score[:, 1]}")
    elif args.video_dir is not None:
        video_list = glob.glob(os.path.join(args.video_dir, "*.mp4"))
        for video in video_list:
            video_feature, audio_feature = extract_feature(video, os.path.join(os.path.dirname(video), "inference_video.wav"))
            video_feature, audio_feature = video_feature.to(device), audio_feature.to(device)
            score = model(video_feature, audio_feature)
            logging_combine(logger, f"The {video} output score is: {score[:, 1]}")
    else:
        logging_combine(logger, "Please input the right path of video/video directory")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="inference video")
    parser.add_argument("--video_path", help="input video name")
    parser.add_argument("--video_dir", help="input video dir")
    args = parser.parse_args()
    test(args)
