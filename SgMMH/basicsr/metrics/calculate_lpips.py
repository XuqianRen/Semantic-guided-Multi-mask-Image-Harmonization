import cv2
import glob
import argparse
import numpy as np
import os.path as osp
import os
from torchvision.transforms.functional import normalize
import torch
from basicsr.utils import img2tensor
from tqdm import tqdm
try:
    import lpips
    from skimage.measure import compare_mse as mse
    from skimage.measure import compare_psnr as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    print('Please install lpips: pip install lpips')

from PIL import Image
from basicsr.utils.options import dict2str, parse

def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', type=str, required=True, help='Path to test files.')

    args = parser.parse_args()
    # opt = parse(args.opt, is_train=is_train)
    return args

def main():
    # Configurations
    # -------------------------------------------------------------------------
    opt = parse_options()
    
    image_folder = opt.path
    
    # image_folder = 'results/test_OM_Mask_HScene_20000/visualization/HScene'
    # ------------------------------------------------------------------------- 
    mse_all = []
    psnr_all = []
    ssim_all = []
    lpips_all = []
    img_list = os.listdir(image_folder)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]  

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda() # RGB, normalized to [-1,1] 
    for img_path in tqdm(img_list):
        # print(img_path)
        img = cv2.imread(osp.join(image_folder,img_path))
        try:
            # Image.open(osp.join(image_folder,img_path)).load()
            length = int(img.shape[1]/4)
            real = img[:,:length,:]
            # ham =  img[:,512:768,:]
            ham = img[:,length*2:length*3,:]
            

            mse_score = mse(ham,real)
            psnr_score = 20. * np.log10(255. / np.sqrt(mse_score+1e-9))
            ssim_score_op_a = ssim(real[:,:,0],ham[:,:,0],data_range=ham[:,:,0].max() - ham[:,:,0].min())
            ssim_score_op_b = ssim(real[:,:,1],ham[:,:,1],data_range=ham[:,:,1].max() - ham[:,:,1].min())
            ssim_score_op_c = ssim(real[:,:,2],ham[:,:,2],data_range=ham[:,:,2].max() - ham[:,:,2].min())
            ssim_score = (ssim_score_op_a + ssim_score_op_b + ssim_score_op_c)/3

            mse_all.append(mse_score)
            psnr_all.append(psnr_score)
            ssim_all.append(ssim_score)
            
            real, ham = img2tensor([real, ham])
            
            # norm to [-1, 1]
            normalize(real, mean, std, inplace=True)
            normalize(ham, mean, std, inplace=True)
            # calculate lpips
            lpips_val = loss_fn_vgg(ham.unsqueeze(0).cuda(), real.unsqueeze(0).cuda())
            lpips_all.append(lpips_val.item())
            # print(mse_score,psnr_score,ssim_score, lpips_val)
            del ham, real, mse_score, psnr_score, ssim_score, lpips_val
            torch.cuda.empty_cache()  
        except OSError:
            print(img_path)
            # exit()

    print(f'Average: MSE: {sum(mse_all) / len(mse_all):.4f}')
    print(f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.4f}')
    print(f'Average: SSIM: {sum(ssim_all) / len(ssim_all):.4f}')
    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.4f}')
if __name__ == '__main__':
    main()
