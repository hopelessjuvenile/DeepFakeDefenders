import torch.utils.data as data
import os
from utils.extract_ffdv import get_feature
from typing import *
import mmap
import torch
from mmap import ACCESS_READ
import logging

dataset_logger = logging.getLogger("main_logger")


class FFDV_Dataset(data.Dataset):
    def __init__(self, root_path: str = None,
                 mode: str = None,
                 gt_list: Optional[Dict[str, str]] = None):
        self.root_path = root_path
        self.data_dict = list()
        self.mode = mode
        self.gt_list = gt_list
        self.root_split_path = os.path.join(self.root_path, self.mode + "_raw")
        self.make_data_json()

    def make_data_json(self):
        with open(os.path.join(self.root_path, self.gt_list[self.mode]), "r+") as f:
            with mmap.mmap(f.fileno(), length=0, access=ACCESS_READ) as mm:
                mm.readline()
                while True:
                    line = mm.readline()
                    if not line:
                        break
                    video_name = line.decode('utf-8').rstrip().split(',')[0].split('.')[0]
                    # if self.mode == "testset":
                    #     self.data_dict.append({"video_name": video_name + ".mp4",
                    #                            "audio_name": video_name + ".wav",
                    #                            "gt": line.decode('utf-8').rstrip().split(',')[1]})
                    # else:
                    self.data_dict.append({"video_name": video_name + ".mp4.npy",
                                           "audio_name": video_name + ".mp4.jpg",
                                           "gt": line.decode('utf-8').rstrip().split(',')[1]})
        dataset_logger.info("data json loading have been successful")

    def __getitem__(self, item):
        if self.mode == "testset":
            video_name, audio_name, gt = self.data_dict[item]["video_name"], self.data_dict[item]["audio_name"], float(self.data_dict[item]["gt"])
            # video_raw = get_feature(self.root_split_path, video_name, audio_name, mode="testset")
            video_raw = get_feature(self.root_split_path, video_name, audio_name)
        else:
            video_name, audio_name, gt = self.data_dict[item]["video_name"], self.data_dict[item]["audio_name"], int(self.data_dict[item]["gt"])
            video_raw = get_feature(self.root_split_path, video_name, audio_name)
        file_name = ".".join(video_name.split(".")[:-1])
        return {"data": video_raw, "gt": torch.tensor(gt), "file": file_name}

    def __len__(self):
        return len(self.data_dict)


if __name__ == '__main__':
    import mmap
    gt_filename_list = {"trainset": "trainset_label.txt", "valset": "valset_label.txt", "testset": "testset_label.csv"}
    dataset = FFDV_Dataset(root_path=os.getenv("FFDV_TRAINING_ROOT_DIR"), mode="testset", gt_list=gt_filename_list)
    print(dataset.__len__())
    raw_data = dataset[111]
    print(raw_data["data"]["video"].shape)
    print(raw_data["data"]["audio"].shape)
