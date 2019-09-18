# Code to produce colored segmentation output in Pytorch for all cityscapes subsets  
# Sept 2019
# Kailun Yang
#######################

import numpy as np
import torch
import os
import importlib

from PIL import Image
from argparse import ArgumentParser

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from torchvision.transforms import ToTensor, ToPILImage

from dataset import cityscapes
from swaftnet import Net
from transform import Relabel, ToLabel, Colorize

from resnet.resnet_single_scale import *

import visdom

NUM_CHANNELS = 3
NUM_CLASSES = 28

image_transform = ToPILImage()
input_transform_cityscapes = Compose([
    Resize((512,1024*4),Image.BILINEAR),
    ToTensor(),
])

def main(args):

    modelpath = args.loadDir + args.loadModel
    weightspath = args.loadDir + args.loadWeights

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    resnet = resnet18(pretrained=True, efficient=False, use_bn=True)
    model = Net(resnet, size=(512, 1024*4), num_classes=NUM_CLASSES)
  
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()

    def load_my_state_dict(model, state_dict):  
        own_state = model.state_dict()
        
        for a,b in zip(own_state.keys(),state_dict.keys()):
            print(a,'      ',b)
        print('-----------')
        
        for name, param in state_dict.items():
            name = name[7:]
            if name not in own_state:
                 print('{} not in own_state'.format(name))
                 continue
            own_state[name].copy_(param)
        
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights LOADED successfully")

    model.eval()

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")
    
    loader = DataLoader(cityscapes(args.datadir, input_transform_cityscapes, subset=args.subset),
        num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    # For visualizer:
    # must launch in other window "python3.6 -m visdom.server -port 8097"
    # and access localhost:8097 to see it
    if (args.visualize):
        vis = visdom.Visdom()
    
    with torch.no_grad():
        for step, (images, filename) in enumerate(loader):
            
            images = images.cuda()
            outputs = model(images)

            heatmap = outputs[0,26,:,:].cpu().data
            heatmap = (heatmap - heatmap.min().min()) / (heatmap.max().max() - heatmap.min().min() + 1e-6)
            heatmap = heatmap.unsqueeze(0)
        
            filenameSave = "./save_color/" + filename[0].split("leftImg8bit/")[1]
            os.makedirs(os.path.dirname(filenameSave), exist_ok=True)

            label_save = ToPILImage()(heatmap)    
            label_save.save(filenameSave) 
            print (step, filenameSave)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')

    parser.add_argument('--loadDir',default="../trained_models/")
    parser.add_argument('--loadWeights', default="erfnet_pretrained.pth")
    parser.add_argument('--loadModel', default="swaftnet.py")
    parser.add_argument('--subset', default="val")  #can be val, test, train, demoSequence

    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')

    parser.add_argument('--visualize', action='store_true')
    main(parser.parse_args())
