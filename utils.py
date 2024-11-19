import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

def gradient(img):
    height = img.size(2)
    width = img.size(3)
    gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
    gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
    return gradient_h, gradient_w

def tv_loss(illumination):
    gradient_illu_h, gradient_illu_w = gradient(illumination)
    loss_h = gradient_illu_h
    loss_w = gradient_illu_w
    loss = loss_h.mean() + loss_w.mean()  
    return loss

def C_loss(R1, R2):
    loss = torch.nn.MSELoss()(R1, R2)  # 公式21
    return loss

def R_loss(L1, R1, im1, X1):
    max_rgb1, _ = torch.max(im1, 1)
    max_rgb1 = max_rgb1.unsqueeze(1)
    loss1 = torch.nn.MSELoss()(L1*R1, X1) + torch.nn.MSELoss()(R1, X1/L1.detach()) # loss1是式23
    loss2 = torch.nn.MSELoss()(L1, max_rgb1) + tv_loss(L1)  # torch.nn.MSELoss()(L1, max_rgb1)是公式11的第一项，tv_loss(L1)是#式11的第二项
    return loss1 + loss2
    
def P_loss(im1, X1, scale=500):
    loss = torch.nn.MSELoss()(im1, X1)
    return loss * scale

def PairLIELoss(R1, R2, L1, im1, X1):
    loss1 = C_loss(R1, R2)
    loss2 = R_loss(L1, R1, im1, X1)
    loss3 = P_loss(im1, X1)
    return loss1 + loss2 + loss3 
    
