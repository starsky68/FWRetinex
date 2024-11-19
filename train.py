import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import random
import shutil
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from thop import profile
from torchvision import transforms
from data import get_training_set, get_eval_set
from tensorboardX import SummaryWriter
from utils import *
from measure import *
from FWRetinex import *


# Training settings
parser = argparse.ArgumentParser(description='PILIE')
parser.add_argument('--batchSize', type=int, default=1, help='training batch size')  #default=1
parser.add_argument('--nEpochs', type=int, default=230, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=20, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default='100', help='learning rate decay type')
parser.add_argument('--weight_decay', type=float, default=1e-9, help='weight decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123456789, help='random seed to use. Default=123')
parser.add_argument('--data_train', type=str, default='/data/AGroup/2open_dataset/LowlightEnhancDatasets/SICE128/paired_lowlight_datasets/')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
parser.add_argument('--save_folder', default='ckpt/', help='Location to save checkpoint models')
parser.add_argument('--output_folder', default='results/val_results/', help='Location to save checkpoint models')
parser.add_argument('--label_path', default='/data/AGroup/2open_dataset/LowlightEnhancDatasets/SICE128/Eval/GT_128_HQ/', help='Location to save checkpoint models')
parser.add_argument('--data_test', type=str, default='/data/AGroup/2open_dataset/LowlightEnhancDatasets/SICE128/Eval/low_128_HQ')
parser.add_argument('--log_dir', type=str, default='results/log_dir/')
parser.add_argument('--t_model_type', type=str, default='zdce')
parser.add_argument('--lc', type=float, default=0.14)

opt = parser.parse_args()

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
cudnn.benchmark = False

def train(model,training_data_loader,optimizer):
    model.train()
    loss_print = 0
    for iteration, batch in enumerate(training_data_loader, 1):

        im1, im2, file1, file2 = batch[0], batch[1], batch[2], batch[3]
        im1 = im1.cuda()
        im2 = im2.cuda()
        L1, R1, X1 = model(im1)
        L2, R2, X2 = model(im2)
           
        loss = PairLIELoss(R1, R2, L1, im1, X1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_print = loss_print + loss.item()
        if iteration % 10 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Learning rate: lr={}.".format(epoch,
                iteration, len(training_data_loader), loss_print, optimizer.param_groups[0]['lr']))
            loss_print = 0


def val(model,testing_data_loader,lc,output_folder):
    #torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    n=0
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = batch[0], batch[1]
        input = input.cuda()

        with torch.no_grad():
            L, R, X = model(input)
            I = torch.pow(L,lc) * R  # SICE=0.14_ssim, LOL_Real=0.23, LOL_Syn=0.22.Huawei=0.19, V2:SICE=0.01, LOL_Real=0.13, LOL_Syn=
        if not os.path.exists(output_folder):
            os.mkdir(output_folder + 'I/')        
        I = I.cpu()        
        I_img = transforms.ToPILImage()(I.squeeze(0))                       
        I_img.save(output_folder + 'I/'+name[0])                       
        n+=1
    #torch.set_grad_enabled(True)
    print('val img num:',n) 
  
def checkpoint_psnr(epoch,model,model_out_path):
    torch.save(model.state_dict(), model_out_path+"/best_psnr.pth")
    
def checkpoint_ssim(epoch,model,model_out_path):
    torch.save(model.state_dict(), model_out_path+"/best_ssim.pth")
    
def checkpoint_lpips(epoch,model,model_out_path):
    torch.save(model.state_dict(), model_out_path+"/best_lpips.pth")
    
if __name__ == '__main__':
    seed_torch(opt.seed)
    if opt.gpu_mode and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    print('===> Loading train datasets')
    train_set = get_training_set(opt.data_train)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print('===> Loading test datasets')
    test_set = get_eval_set(opt.data_test)
    testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)


    model= FWetinex().cuda()
 
    
    model_out_path = opt.save_folder    
    if not os.path.exists(model_out_path):
        os.makedirs(model_out_path)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    milestones = []
    for i in range(1, opt.nEpochs+1):
        if i % opt.decay == 0:
            milestones.append(i)
    scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

    score_best = 0
    writer = SummaryWriter(opt.log_dir)
    best_psnr=0
    best_ssim=0
    best_lpips=2
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train(model,training_data_loader,optimizer)
        scheduler.step()
        val(model,testing_data_loader,opt.lc,opt.output_folder)
        psnr,ssim,lpips =  metrics(opt.output_folder + 'I/*.png', opt.label_path)
        lpips = lpips.item()
        writer.add_scalar('avg PSNR', psnr, epoch)
        writer.add_scalar('avg SSIM', ssim, epoch)
        writer.add_scalar('avg LPIPS', lpips, epoch)
        print('the psnr value is:', str(psnr)) 
        print('the ssim value is:', str(ssim)) 
        print('the lpips value is:', str(lpips)) 
        
        if psnr > best_psnr:
            best_psnr = psnr
            checkpoint_psnr(epoch,model,model_out_path)
        if ssim > best_ssim:
            best_ssim = ssim
            checkpoint_ssim(epoch,model,model_out_path)
        if lpips < best_lpips:
            best_lpips = lpips
            checkpoint_lpips(epoch,model,model_out_path)
        print('the highest psnr value is:', str(best_psnr))
        print('the highest ssim value is:', str(best_ssim))
        print('the lowest lpips value is:', str(best_lpips))
    writer.close()
