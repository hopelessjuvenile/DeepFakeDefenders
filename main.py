import os
import torch
import logging
from dataset.ffdv_dataset import FFDV_Dataset
# from dataset.submit_dataset import FFDV_Dataset
from torch.utils.data import DataLoader
from model import FFDV_Cov
import torch.nn as nn
from torch.optim import Adam
import torch.optim as optim
from tqdm import tqdm
import time
from datetime import datetime
from iopath.common.file_io import g_pathmgr
import torch.utils.data as data
from typing import *
from sklearn.metrics import roc_auc_score
import csv
import argparse
import yaml


torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

INTERVAL_EPOCH = 1
INTERVAL_STEP = 900
BATCH_SIZE = 128
EPOCH_TOTAL = 30
LOG_FORMAT = "Time:%(asctime)s - Level:%(levelname)s - Message:%(message)s"
BEST_AUC = 0.0

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


def save_model(weight: Dict[str, torch.Tensor] = None,
               cur_epoch: int = None,
               optimizer: torch.optim.Optimizer = None,
               scheduler: torch.optim.lr_scheduler.StepLR = None,
               checkpoint_path: str = None):
    checkpoint = {
        "epoch": cur_epoch + 1,
        "model_state": weight,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    with g_pathmgr.open(checkpoint_path, "wb") as f:
        torch.save(checkpoint, f)
    logging_combine(logger, f"Model [{checkpoint_path}/epoch {cur_epoch + 1}] save success!")


def train(model: nn.Module = None,
          train_loader: data.DataLoader = None,
          val_loader: data.DataLoader = None,
          criterion=None,
          optimizer: torch.optim.Optimizer = None,
          scheduler: torch.optim.lr_scheduler.StepLR = None,
          epoch_total: int = None,
          last_epoch: int = 0,
          device: torch.device = None):

    global BEST_AUC, INTERVAL_EPOCH
    for epoch in range(last_epoch, last_epoch + epoch_total, 1):
        start = time.time()
        for it, data in tqdm(enumerate(train_loader),
                             desc="[Epoch %s/train]: " % (epoch + 1),
                             total=train_loader.__len__()):
            videos, audios, gt = torch.tensor(data["data"]["video"]), data["data"]["audio"], data["gt"]
            videos, audios, gt = videos.cuda(device), audios.cuda(device), gt.cuda(device)
            output = model(videos, audios)
            # output = output.squeeze(1)
            loss = criterion(output, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if it % INTERVAL_STEP == 0:
                logging_combine(logger, f"Current Loss(CrossEntropyLoss): {loss.item()}")
        end = time.time()
        logging_combine(logger, "running time of epoch %s : %s s" % ((epoch + 1), (end - start)))
        scheduler.step()
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        checkpoint_path = f"./checkpoint/FFDV_Cov_{epoch}_{now}_fusion_cos_submit.pth"
        save_model(model.state_dict(), epoch, optimizer, scheduler, checkpoint_path)
        if epoch % INTERVAL_EPOCH == 0:
            auc = evaluate(model.state_dict(), val_loader, epoch, device)
            if auc > BEST_AUC:
                BEST_AUC = auc
                best_checkpoint_path = f"./checkpoint/best_fusion_cos_submit.pth"
                save_model(model.state_dict(), epoch, optimizer, scheduler, best_checkpoint_path)
                logging_combine(logger, f"current best AUC@1 is : {BEST_AUC}")


@torch.no_grad()
def evaluate(weight: Dict[str, torch.Tensor] = None,
             val_loader: data.DataLoader = None,
             epoch: int = None,
             device: torch.device = None):
    preds_all = torch.tensor([])
    preds_all_without_argmax = torch.tensor([])
    label_all = torch.tensor([])
    file_names = list()
    
    model = FFDV_Cov(num_class=2, extra_model_path="../resnet18_no_linear.pth", mode="val", temporal=128, device=device)
    model.load_state_dict(weight)
    model = model.cuda(device=device)

    for it, data in tqdm(enumerate(val_loader),
                         desc="[Epoch %s/val]: " % (epoch + 1),
                         total=val_loader.__len__()):
        videos, audios, gt = torch.tensor(data["data"]["video"]), data["data"]["audio"], data["gt"]
        videos, audios, gt = videos.cuda(device), audios.cuda(device), gt.cuda(device)
        file = data["file"]
        output = model(videos, audios)
        output = output[:, 1]
        # output = F.softmax(output, dim=1)[:, 1]
        # preds = torch.argmax(output, dim=1)
        # print(output)
        preds_all = torch.cat((preds_all, output.cpu()), dim=0)
        preds_all_without_argmax = torch.cat((preds_all_without_argmax, output.cpu()), dim=0)
        label_all = torch.cat((label_all, gt.cpu()), dim=0)
        file_names.extend(file)
        # preds_all_without_argmax.append(output.cpu())
        # preds_all.append(preds.cpu())
        # label_all.append(gt.cpu())

    # preds_all = torch.cat(preds_all, dim=0)
    # label_all = torch.cat(label_all, dim=0)

    # Calculate the accuracy
    # correct = preds_all.eq(label_all).sum().item()
    # total = len(label_all)
    # accuracy = correct / total * 100.0
    write_csv(preds_all, file_names, "prediction.txt.csv")
    
    auc = roc_auc_score(label_all, preds_all)
    return auc


def write_csv(pred: torch.Tensor = None,
              file_names: List = None, output_file: str = None):
    if pred is None or file_names is None:
        raise ValueError("Both 'pred' and 'file_names' must be provided.")
    if len(pred) != len(file_names):
        raise ValueError("'pred' and 'file_names' must have the same length.")

    if isinstance(pred, torch.Tensor):
        pred = pred.tolist()

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['video_name', 'y_pred'])  # Write header
        for name, y in zip(file_names, pred):
            writer.writerow([name, y])  # Write data rows


def main(eval_checkpoint=False, cfg=None):
    os.environ["FFDV_TRAINING_ROOT_DIR"] = os.path.join(os.getcwd(), "data")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_path = "pretrain_weight/resnet18_no_linear.pth"
    gt_filename_list = {"trainset": cfg["label_name"]["trainset"], "valset": cfg["label_name"]["valset"], "testset": cfg["label_name"]["testset"]}
    train_dataset = FFDV_Dataset(root_path=os.getenv("FFDV_TRAINING_ROOT_DIR"), mode="trainset", gt_list=gt_filename_list)
    val_dataset = FFDV_Dataset(root_path=os.getenv("FFDV_TRAINING_ROOT_DIR"), mode="valset", gt_list=gt_filename_list)
    test_dataset = FFDV_Dataset(root_path=os.getenv("FFDV_TRAINING_ROOT_DIR"), mode="testset", gt_list=gt_filename_list)

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    logging_combine(logger, f"Train dataloader loading complete! dataset length is: {train_loader.__len__()}, original length is: {train_loader.__len__() * BATCH_SIZE}")
    logging_combine(logger, f"Val dataloader loading complete! dataset length is: {val_loader.__len__()}, original length is: {val_loader.__len__() * BATCH_SIZE}")
    logging_combine(logger, f"Test dataloader loading complete! dataset length is: {test_loader.__len__()}, original length is: {test_loader.__len__() * BATCH_SIZE}")

    model = FFDV_Cov(num_class=2, extra_model_path=pretrain_path, temporal=128, device=device)
    model = model.cuda(device=device)
    model.train()

    criterion = nn.CrossEntropyLoss().cuda()
    # criterion = nn.BCELoss().cuda()
    optimizer = Adam(model.parameters(), 0.003)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)

    if eval_checkpoint:
        model_dict = torch.load(eval_checkpoint)

        last_epoch = model_dict["epoch"]
        model.load_state_dict(model_dict["model_state"])
        optimizer.load_state_dict(model_dict["optimizer_state"])
        scheduler.load_state_dict(model_dict["scheduler_state"])
        
        if cfg["evaluate"] == "val":
            auc = evaluate(model.state_dict(), val_loader, last_epoch, device)
        if cfg["evaluate"] == "test":
            auc = evaluate(model.state_dict(), test_loader, last_epoch, device)
        logging_combine(logger, f"last epoch {last_epoch} auc is: {auc * 100}%")
        return
    # if testing testset please use annotation code below (because we don't have testset ground truth)
    if cfg["evaluate"] == "test":
        train(model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCH_TOTAL, 0, device)
    if cfg["evaluate"] == "val":
        train(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCH_TOTAL, 0, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="multi-ffdv training")
    parser.add_argument("--checkpoint", help="whether just evaluate specific model")
    args = parser.parse_args()

    with open('cfg/cfg.yml', 'r') as file:
        config = yaml.safe_load(file)
    main(args.checkpoint, config)
