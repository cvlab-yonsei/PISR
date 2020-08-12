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
from datasets import get_test_dataloader
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


def evaluate_single_epoch(config, student_model, dataloader_dict, eval_type):
    student_model.eval()
    log_dict = {}
    with torch.no_grad():
        for name, dataloader in dataloader_dict.items():
            print('evaluate %s'%(name))
            batch_size = config.eval.batch_size
            total_size = len(dataloader.dataset)
            total_step = math.ceil(total_size / batch_size)

            tbar = tqdm.tqdm(enumerate(dataloader), total=total_step)

            total_psnr = 0
            total_iter = 0
            for i, (LR_img, HR_img, filepath) in tbar:
                HR_img = HR_img.to(device)
                LR_img = LR_img.to(device)

                student_pred_dict = student_model.forward(LR=LR_img)
                pred_hr = student_pred_dict['hr']
                pred_hr = quantize(pred_hr, config.data.rgb_range)
                total_psnr += get_psnr(pred_hr, HR_img, config.data.scale,
                                      config.data.rgb_range,
                                      benchmark=eval_type=='test')

                f_epoch = i / total_step
                desc = '{:5s}'.format(eval_type)
                desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
                tbar.set_description(desc)
                total_iter = i

            avg_psnr = total_psnr / (total_iter+1)
            log_dict[name] = avg_psnr
            print('%s : %.3f'%(name, avg_psnr))
            
    return log_dict


def evaluate(config, student_model, dataloaders, start_epoch):
    # test phase
    result = evaluate_single_epoch(config, student_model,
                          dataloaders, eval_type='test')
    
    return result


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run(config):
    student_model = get_model(config, 'student').to(device)
    print('The nubmer of parameters : %d'%count_parameters(student_model))

    # for student
    optimizer_s = None
    checkpoint_s = utils.checkpoint.get_initial_checkpoint(config,
                                                         model_type='student')
    if checkpoint_s is not None:
        last_epoch_s, step_s = utils.checkpoint.load_checkpoint(student_model,
                                 optimizer_s, checkpoint_s, model_type='student')
    else:
        last_epoch_s, step_s = -1, -1
    print('student model from checkpoint: {} last epoch:{}'.format(
        checkpoint_s, last_epoch_s))
    
    print(config.data)
    dataloaders = get_test_dataloader(config)
    result = evaluate(config, student_model, dataloaders, last_epoch_s+1)
    print(result)

    
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



