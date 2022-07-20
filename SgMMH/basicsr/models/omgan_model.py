import torch
from collections import OrderedDict
from .base_model import BaseModel
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils import get_root_logger, imwrite, tensor2img
import cv2
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from os import path as osp
from basicsr.metrics import calculate_metric
import imageio
import os
import matplotlib.pyplot as plt
from PIL import Image
import time
@MODEL_REGISTRY.register()
class OMGANModel(BaseModel):
    """OMGAN model for image harmonization."""
    def __init__(self, opt):
        super(OMGANModel,self).__init__(opt)
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_g)
        self.print_network(self.net_d)
        self.load()
        self.net_g.train()
        self.net_d.train()

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None
        
        if train_opt.get('sigmoid_opt'):
            self.cri_pix_boundary = build_loss(train_opt['sigmoid_opt']).to(self.device)
        else:
            self.cri_pix_boundary = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        
        if train_opt.get('tv_opt'):
            self.cri_tv = build_loss(train_opt['tv_opt']).to(self.device)
        else:
            self.cri_tv = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def feed_data(self, data):
        self.mask = data['mask'].to(self.device)
        self.real = data['real'].to(self.device)
        self.comp = data['comp'].to(self.device)
        

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        self.input = self.comp 

        self.attention_mask = self.net_g(self.input)


        self.attention_add_l = self.attention_mask[:,0,:,:].unsqueeze(1)
        self.attention_mul_l = self.attention_mask[:,1,:,:].unsqueeze(1)
        self.attention_add_a = self.attention_mask[:,2,:,:].unsqueeze(1)
        self.attention_mul_a = self.attention_mask[:,3,:,:].unsqueeze(1)
        self.attention_add_b = self.attention_mask[:,4,:,:].unsqueeze(1)
        self.attention_mul_b = self.attention_mask[:,5,:,:].unsqueeze(1)
        
               
        l,a,b = self.comp.split(1,dim=1)
        
        l = l * self.attention_mul_l   + self.attention_add_l
        a = a * self.attention_mul_a   + self.attention_add_a
        b = b * self.attention_mul_b   + self.attention_add_b
        l = torch.clamp(l,-1,1)
        a = torch.clamp(a,-1,1)
        b = torch.clamp(b,-1,1)

        self.harmonized = torch.cat([l,a,b],dim=1)
        

        
        img_mean = [0.5, 0.5, 0.5]
        img_std = [0.5, 0.5, 0.5]
        
        tmp_real = ((self.real.permute(0,2,3,1)[0,:,:,:].cpu().numpy()*img_std+img_mean)*255).astype("uint8")   
        tmp_real = cv2.cvtColor(tmp_real, cv2.COLOR_LAB2BGR).astype("uint8")
        
        tmp_comp = ((self.comp.permute(0,2,3,1)[0,:,:,:].cpu().numpy()*img_std+img_mean)*255).astype("uint8")   
        tmp_comp = cv2.cvtColor(tmp_comp, cv2.COLOR_LAB2BGR).astype("uint8")   
        tmp_harm = ((self.harmonized.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy()*img_std+img_mean)*255).astype("uint8")   
        tmp_harm = cv2.cvtColor(tmp_harm, cv2.COLOR_LAB2BGR).astype("uint8")   
        
        
        tmp_mul =  torch.abs(self.attention_mul_l-1).permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy().repeat(3,axis=2)*255
        tmp_add =  torch.abs(self.attention_add_l).permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy().repeat(3,axis=2)*255

        tmp_mask = self.mask.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy().repeat(3,axis=2)*255

        tmp_img = np.concatenate([tmp_real,tmp_comp, tmp_harm, tmp_mul, tmp_add, tmp_mask],1) 
        cv2.imwrite('./img.png', tmp_img)


        l_g_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_g_pix = self.cri_pix(self.real, self.harmonized) 

            l_g_total += l_g_pix
            loss_dict['l_g_pix'] = l_g_pix

    #   perceptual loss
        if self.cri_perceptual:
            l_g_percep, l_g_style = self.cri_perceptual(self.real, self.harmonized)
            if l_g_percep is not None:
                  l_g_total += l_g_percep
                  loss_dict['l_g_percep'] = l_g_percep
            if l_g_style is not None:
                  l_g_total += l_g_style
                  loss_dict['l_g_style'] = l_g_style

        real_d_pred = self.net_d(self.real).detach()
        fake_g_pred = self.net_d(self.harmonized)

        l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
        l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
        l_g_gan = (l_g_real + l_g_fake) / 2

        l_g_total += l_g_gan
        loss_dict['l_g_gan'] = l_g_gan

        l_g_total.backward()
        self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # gan loss (relativistic gan)

        # In order to avoid the error in distributed training:
        # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # the variables needed for gradient computation has been modified by
        # an inplace operation",
        # we separate the backwards for real and fake, and also detach the
        # tensor for calculating mean.

        # real
        fake_d_pred = self.net_d(self.harmonized).detach()
        real_d_pred = self.net_d(self.real)
        l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.harmonized.detach())
        l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def load(self):
    # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.attention_mask = self.net_g_ema(self.comp)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.input = self.comp
                self.attention_mask = self.net_g(self.input)

                self.attention_add_l = self.attention_mask[:,0,:,:].unsqueeze(1)
                self.attention_mul_l = self.attention_mask[:,1,:,:].unsqueeze(1)
                self.attention_add_a = self.attention_mask[:,2,:,:].unsqueeze(1)
                self.attention_mul_a = self.attention_mask[:,3,:,:].unsqueeze(1)
                self.attention_add_b = self.attention_mask[:,4,:,:].unsqueeze(1)
                self.attention_mul_b = self.attention_mask[:,5,:,:].unsqueeze(1)
                
                    
                l,a,b = self.comp.split(1,dim=1)
                
                l = l * self.attention_mul_l   + (self.attention_add_l)
                a = a * self.attention_mul_a   + (self.attention_add_a)
                b = b * self.attention_mul_b   + (self.attention_add_b)
                l = torch.clamp(l,-1,1)
                a = torch.clamp(a,-1,1)
                b = torch.clamp(b,-1,1)

                self.harmonized = torch.cat([l,a,b],dim=1)

                
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # if self.opt['rank'] == 0:
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        pbar = tqdm(total=len(dataloader), unit='image')
        start = time.time()
        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['mask_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            ham = tensor2img([visuals['ham']])
            real = tensor2img([visuals['real']])
            comp = tensor2img([visuals['comp']])
            
            mask = tensor2img([visuals['mask']])
            

            output = np.concatenate([real, comp, ham, mask],1)


            # tentative for out of GPU memory
            
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        os.makedirs( osp.join(self.opt['path']['visualization'], dataset_name),exist_ok=True)
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}.png')


                imageio.imwrite(save_img_path , output )


            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=ham, img2=real)
                    
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        
        
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        end = time.time()
        print(end-start)
    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}\n'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['real'] = self.real.detach().cpu()
        out_dict['ham'] = self.harmonized.detach().cpu()
        out_dict['comp'] = self.comp.detach().cpu()
        out_dict['mask'] = self.mask.detach().cpu()
        out_dict['om'] = self.attention_mask.detach().cpu()



        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)