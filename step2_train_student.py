import os
import tqdm
import argparse
import pprint

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
import skimage
import os
import glob
from skimage.io import imread
import skimage
import math
import time
#from utils.imresize import imresize
from datasets import get_train_dataloader, get_valid_dataloader
from transforms import get_transform
from models import get_model
from losses import get_loss
from optimizers import get_optimizer
from schedulers import get_scheduler
from visualizers import get_visualizer
from tensorboardX import SummaryWriter

import utils.config
import utils.checkpoint
from utils.metrics import get_psnr
from utils.utils import quantize

device = None
model_type = None


def adjust_learning_rate(config, epoch):
    lr = config.optimizer.params.lr * (0.5 ** (epoch // config.scheduler.params.step_size))
    return lr


def train_single_epoch(config, student_model, teacher_model, dataloader, criterion,
                       optimizer, epoch, writer, visualizer, postfix_dict):
    student_model.train()
    teacher_model.eval()
    batch_size = config.train.batch_size
    total_size = len(dataloader.dataset)
    total_step = math.ceil(total_size / batch_size)

    log_dict = {}

    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)
    for i, (LR_patch, HR_patch, filepath) in tbar:
        if not HR_patch.is_cuda:
            HR_patch = HR_patch.to(device)
            LR_patch = LR_patch.to(device)

        optimizer.zero_grad()

        teacher_pred_dict = teacher_model.forward(LR=LR_patch, HR=HR_patch)
        student_pred_dict = student_model.forward(LR=LR_patch, teacher_pred_dict=teacher_pred_dict)
        loss = criterion['train'](teacher_pred_dict, student_pred_dict, HR_patch)
        for k, v in loss.items():
            log_dict[k] = v.item()

        loss['loss'].backward()
        if 'gradient_clip' in config.optimizer:
            clip = config.optimizer.gradient_clip
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip)

        optimizer.step()

        f_epoch = epoch + i / total_step

        log_dict['lr'] = optimizer.param_groups[0]['lr']
        for key, value in log_dict.items():
            if 'train/{}'.format(key) in postfix_dict:
                postfix_dict['train/{}'.format(key)] = value

        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)

        if i % 1000 == 0:
            log_step = int(f_epoch * 10000)
            if writer is not None:
                for key, value in log_dict.items():
                    writer.add_scalar('train/{}'.format(key), value, log_step)


def evaluate_single_epoch(config, student_model, teacher_model, dataloader,
                          criterion, epoch, writer,
                          visualizer, postfix_dict, eval_type):
    teacher_model.eval()
    student_model.eval()
    with torch.no_grad():
        batch_size = config.eval.batch_size
        total_size = len(dataloader.dataset)
        total_step = math.ceil(total_size / batch_size)

        tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

        total_psnr = 0
        total_loss = 0
        total_iter = 0
        for i, (LR_img, HR_img, filepath) in tbar:
            HR_img = HR_img.to(device)
            LR_img = LR_img.to(device)

            teacher_pred_dict = teacher_model.forward(LR=LR_img,HR=HR_img)
            student_pred_dict = student_model.forward(LR=LR_img, teacher_pred_dict=teacher_pred_dict)
            pred_hr = student_pred_dict['hr']
            total_loss += criterion['val'](pred_hr, HR_img).item()

            pred_hr = quantize(pred_hr, config.data.rgb_range)
            total_psnr += get_psnr(pred_hr, HR_img, config.data.scale,
                                  config.data.rgb_range,
                                  benchmark=eval_type=='test')

            f_epoch = epoch + i / total_step
            desc = '{:5s}'.format(eval_type)
            desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
            tbar.set_description(desc)
            tbar.set_postfix(**postfix_dict)

            if writer is not None and i < 5:
                fig = visualizer(LR_img, HR_img,
                                 student_pred_dict, teacher_pred_dict)
                writer.add_figure('{}/{:04d}'.format(eval_type, i), fig,
                                 global_step=epoch)
            total_iter = i

        log_dict = {}
        avg_loss = total_loss / (total_iter+1)
        avg_psnr = total_psnr / (total_iter+1)
        log_dict['loss'] = avg_loss
        log_dict['psnr'] = avg_psnr

        for key, value in log_dict.items():
            if writer is not None:
                writer.add_scalar('{}/{}'.format(eval_type, key), value, epoch)
            postfix_dict['{}/{}'.format(eval_type, key)] = value

        return avg_psnr


def train(config, student_model, teacher_model, dataloaders, criterion,
          optimizer, scheduler, writer, visualizer, start_epoch):
    num_epochs = config.train.num_epochs
    if torch.cuda.device_count() > 1:
        teacher_model = torch.nn.DataParallel(teacher_model)
        student_model = torch.nn.DataParallel(student_model)

    postfix_dict = {'train/lr': 0.0,
                    'train/loss': 0.0,
                    'val/psnr': 0.0,
                    'val/loss': 0.0}
    best_psnr = 0.0
    best_epoch = 0
    for epoch in range(start_epoch, num_epochs):

        # val phase
        psnr = evaluate_single_epoch(config, student_model, teacher_model,
                                     dataloaders['val'],
                                     criterion, epoch, writer,
                                     visualizer, postfix_dict,
                                     eval_type='val')
        if config.scheduler.name == 'reduce_lr_on_plateau':
            scheduler.step(psnr)
        elif config.scheduler.name != 'reduce_lr_on_plateau':
            scheduler.step()

        utils.checkpoint.save_checkpoint(config, student_model, optimizer,
                                         epoch, 0,
                                         model_type='student')
        if psnr > best_psnr:
            best_psnr = psnr
            best_epoch = epoch

        # train phase
        train_single_epoch(config, student_model, teacher_model,
                           dataloaders['train'],
                           criterion, optimizer, epoch, writer,
                           visualizer, postfix_dict)


    return {'best_psnr': best_psnr, 'best_epoch': best_epoch}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(config):
    teacher_model = get_model(config, 'teacher').to(device)
    student_model = get_model(config, 'student').to(device)
    print('The nubmer of parameters : %d'%count_parameters(student_model))
    criterion = get_loss(config)


    # for teacher
    optimizer_t = None
    checkpoint_t = utils.checkpoint.get_initial_checkpoint(config,
                                                         model_type='teacher')
    if checkpoint_t is not None:
        last_epoch_t, step_t = utils.checkpoint.load_checkpoint(teacher_model,
                                 optimizer_t, checkpoint_t, model_type='teacher')
    else:
        last_epoch_t, step_t = -1, -1
    print('teacher model from checkpoint: {} last epoch:{}'.format(
        checkpoint_t, last_epoch_t))

    # for student
    optimizer_s = get_optimizer(config, student_model)
    checkpoint_s = utils.checkpoint.get_initial_checkpoint(config,
                                                         model_type='student')
    if checkpoint_s is not None:
        last_epoch_s, step_s = utils.checkpoint.load_checkpoint(student_model,
                                 optimizer_s, checkpoint_s, model_type='student')
    else:
        last_epoch_s, step_s = -1, -1
    print('student model from checkpoint: {} last epoch:{}'.format(
        checkpoint_s, last_epoch_s))

    scheduler_s = get_scheduler(config, optimizer_s, last_epoch_s)

    print(config.data)
    dataloaders = {'train':get_train_dataloader(config, get_transform(config)),
                   'val':get_valid_dataloader(config)}
                   #'test':get_test_dataloader(config)}
    writer = SummaryWriter(config.train['student' + '_dir'])
    visualizer = get_visualizer(config)
    result = train(config, student_model, teacher_model, dataloaders,
          criterion, optimizer_s, scheduler_s, writer,
          visualizer, last_epoch_s+1)
    
    print('best psnr : %.3f, best epoch: %d'%(result['best_psnr'], result['best_epoch']))

def parse_args():
    parser = argparse.ArgumentParser(description='student network')
    parser.add_argument('--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    global device
    import warnings
    global model_type
    model_type = 'student'

    warnings.filterwarnings("ignore")

    print('train %s network'%model_type)
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')

    config = utils.config.load(args.config_file)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(config.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pprint.PrettyPrinter(indent=2).pprint(config)
    utils.prepare_train_directories(config, model_type=model_type)
    run(config)

    print('success!')


if __name__ == '__main__':
    main()



