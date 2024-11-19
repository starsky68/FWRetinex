import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import time
import argparse
from thop import profile
# from net.net import 
# from model import PINet
from model import RePINet
from data import get_eval_set
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
from measure import *
import torch


parser = argparse.ArgumentParser(description='PairLIE')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGB')
# parser.add_argument('--data_test', type=str, default='../dataset/LIE/LOL-test/raw')
# parser.add_argument('--data_test', type=str, default='../dataset/LIE/SICE-test/image')
# parser.add_argument('--data_test', type=str, default='/home/jinyutao/RePI/datasets/tests/SICE/Low_256')
# parser.add_argument('--data_test', type=str, default='/home/jinyutao/DCRetinex/datasets/tests/classfication/train/None')
parser.add_argument('--data_test', type=str, default='/home/jinyutao/DCRetinex/datasets/baidu')
# parser.add_argument('--model', default='/home/jinyutao/PairLIE-main/train_weights/best_PairLIE.pth', help='Pretrained base model')  
# parser.add_argument('--model', default='/home/jinyutao/PILIENet/SICE1_weights/PINet/014/best_ssim_PILIELoss.pth', help='Pretrained base model')
parser.add_argument('--model', default='/home/jinyutao/DCRetinex/ckpt/RePINet/best_psnr_RePILossTP_urtx.pth', help='Pretrained base model')    
parser.add_argument('--output_folder', type=str, default='outputs/')
parser.add_argument('--label_folder', type=str, default='/home/jinyutao/RePI/datasets/tests/SICE/GT_256/')
opt = parser.parse_args()


print('===> Loading datasets')
test_set = get_eval_set(opt.data_test)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

print('===> Building model')


# model = PINet()cuda()
model = RePINet().cuda()
model.load_state_dict(torch.load(opt.model))
print('Pre-trained model is loaded.')

def eval():
    torch.set_grad_enabled(False)
    model.eval()
    print('\nEvaluation:')
    for batch in testing_data_loader:
        with torch.no_grad():
            input, name = batch[0], batch[1]
        input = input.cuda()
        #print(name)

        with torch.no_grad():
            #print(input)
            L, R, X, = model(input)
            #print(R)
            D = input - X  
            I = torch.pow(L,0.14) * R  # V1:SICE=0.14, LOL_Real=0.23, LOL_Syn=0.22, Huawei=0.19, Nikon=0.20  V2:SICE=0.01, LOL_Real=0.13, LOL_Syn=0.09, Huawei=0.09, Nikon=0.17. Datasets2:SICE=0.17,Huawei=0.18,Nikon=0.28,v2_SICE=0.10,Huawei=0.18,Nikon=0.29
            # flops, params = profile(model, (input,))
            # print('flops: ', flops, 'params: ', params)

        if not os.path.exists(opt.output_folder):
            os.mkdir(opt.output_folder)
            #os.mkdir(opt.output_folder + 'L/')
            #os.mkdir(opt.output_folder + 'R/')
            os.mkdir(opt.output_folder + 'I/')  
            #os.mkdir(opt.output_folder + 'D/')                       

        L = L.cpu()
        R = R.cpu()
        I = I.cpu()
        D = D.cpu()
        X = X.cpu()
        #i1 = i1.cpu() 
        #i2 = i2.cpu()		

        L_img = transforms.ToPILImage()(L.squeeze(0))
        R_img = transforms.ToPILImage()(R.squeeze(0))
        I_img = transforms.ToPILImage()(I.squeeze(0))                
        D_img = transforms.ToPILImage()(D.squeeze(0))  
        X_img = transforms.ToPILImage()(X.squeeze(0))
        #i1_img = transforms.ToPILImage()(i1.squeeze(0))
        #i2_img = transforms.ToPILImage()(i2.squeeze(0))

        #L_img.save(opt.output_folder + 'L/' + name[0])
        #R_img.save(opt.output_folder + '/R/' + name[0])
        I_img.save(opt.output_folder + 'I/' + name[0])  
        #D_img.save(opt.output_folder + 'D/' + name[0])   
        #X_img.save(opt.output_folder + 'X/' +name[0])
        #i1_img.save(opt.output_folder + 'i1/' + name[0])   
        #i2_img.save(opt.output_folder + 'i2/' +name[0])

    torch.set_grad_enabled(True)
    #print(opt.output_folder + 'I/*.png','11111111111111')
    psnr, ssim, lpips = metrics(opt.output_folder + 'I/*.jpg', opt.label_folder)
    print("===> Avg.PSNR: {:.4f} dB ".format(psnr))
    print("===> Avg.SSIM: {:.4f} ".format(ssim))
    print("===> Avg.LPIPS: {:.4f} ".format(lpips.item()))
eval()


