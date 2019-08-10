import argparse

import torch
import torch.nn.parallel
import datasets
from utils import AverageMeter, img_cvt
import soft_renderer as sr
import soft_renderer.functional as srf
import models
import time
import os
import imageio
import numpy as np

BATCH_SIZE = 100
IMAGE_SIZE = 64
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')

PRINT_FREQ = 100
SAVE_FREQ = 100

MODEL_DIRECTORY = './data/models'
DATASET_DIRECTORY = './data/datasets'

SIGMA_VAL = 0.01
IMAGE_PATH = ''

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('-eid', '--experiment-id', type=str)
parser.add_argument('-d', '--model-directory', type=str, default=MODEL_DIRECTORY)
parser.add_argument('-dd', '--dataset-directory', type=str, default=DATASET_DIRECTORY)

parser.add_argument('-cls', '--class-ids', type=str, default=CLASS_IDS_ALL)
parser.add_argument('-is', '--image-size', type=int, default=IMAGE_SIZE)
parser.add_argument('-bs', '--batch-size', type=int, default=BATCH_SIZE)
parser.add_argument('-img', '--image-path', type=str, default=IMAGE_PATH)

parser.add_argument('-sv', '--sigma-val', type=float, default=SIGMA_VAL)

parser.add_argument('-pf', '--print-freq', type=int, default=PRINT_FREQ)
parser.add_argument('-sf', '--save-freq', type=int, default=SAVE_FREQ)
args = parser.parse_args()

# setup model & optimizer
model = models.Model('data/obj/sphere/sphere_642.obj', args=args)
model = model.cuda()

state_dicts = torch.load(args.model_directory)
model.load_state_dict(state_dicts['model'], strict=False)
model.eval()

dataset_val = datasets.ShapeNet(args.dataset_directory, args.class_ids.split(','), 'val')

directory_output = './data/results/test'
os.makedirs(directory_output, exist_ok=True)
directory_mesh = os.path.join(directory_output, args.experiment_id)
os.makedirs(directory_mesh, exist_ok=True)


def test():
    end = time.time()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses1 = AverageMeter()

    iou_all = []

    for class_id, class_name in dataset_val.class_ids_pair:

        directory_mesh_cls = os.path.join(directory_mesh, class_id)
        os.makedirs(directory_mesh_cls, exist_ok=True)
        iou = 0
        
        for i, (im, vx) in enumerate(dataset_val.get_all_batches_for_evaluation(args.batch_size, class_id)):
            images = torch.autograd.Variable(im).cuda()
            voxels = vx.numpy()

            batch_iou, vertices, faces = model(images, voxels=voxels, task='test')
            iou += batch_iou.sum()
            
            batch_time.update(time.time() - end)
            end = time.time()

            # save demo images
            for k in range(vertices.size(0)):
                obj_id = (i * args.batch_size + k)
                if obj_id % args.save_freq == 0:
                    mesh_path = os.path.join(directory_mesh_cls, '%06d.obj' % obj_id)
                    input_path = os.path.join(directory_mesh_cls, '%06d.png' % obj_id)
                    srf.save_obj(mesh_path, vertices[k], faces[k])
                    imageio.imsave(input_path, img_cvt(images[k]))

            # print loss
            if i % args.print_freq == 0:
                print('Iter: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f}\t'
                      'IoU {2:.3f}\t'.format(i, ((dataset_val.num_data[class_id] * 24) // args.batch_size), 
                                             batch_iou.mean(),
                                             batch_time=batch_time))

        iou_cls = iou / 24. / dataset_val.num_data[class_id] * 100
        iou_all.append(iou_cls)
        print('=================================')
        print('Mean IoU: %.3f for class %s' % (iou_cls, class_name))
        print('\n')

    print('=================================')
    print('Mean IoU: %.3f for all classes' % (sum(iou_all) / len(iou_all)))


test()