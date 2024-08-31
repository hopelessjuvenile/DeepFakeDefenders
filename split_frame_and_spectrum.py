import os
from mmap import mmap, ACCESS_READ

os.environ["FFDV_TRAINING_ROOT_DIR"] = "/hy-tmp/multi-ffdv/data"
t_filename_list = {"trainset": "trainset_label.txt", "valset": "valset_label.txt"}


if __name__ == '__main__':
    video_path = os.getenv("FFDV_TRAINING_ROOT_DIR")
    for gt_mode in t_filename_list.keys():
        frame_path = os.path.join(video_path, f'{gt_mode}_frame')
        if not os.path.exists(frame_path):
            os.mkdir(frame_path)
        with open(os.path.join(video_path, t_filename_list[gt_mode]), "r+") as f:
            with mmap(f.fileno(), length=0, access=ACCESS_READ) as mm:
                while True:
                    line = mm.readline()
                    if not line:
                        break
                    video_name = line.decode('utf-8').rstrip().split(',')[0]
                    # data_dict.append({"video_name": line.decode('utf-8').rstrip().split(',')[0],
                    #                        "gt": line.decode('utf-8').rstrip().split(',')[1]})