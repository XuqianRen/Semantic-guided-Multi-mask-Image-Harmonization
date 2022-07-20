import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
import cv2
import numpy as np
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import cv2
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2imgRGB
from basicsr.utils.registry import MODEL_REGISTRY
import torch.nn.functional as F
from .base_model import BaseModel
import imageio
import os

@MODEL_REGISTRY.register()
class OMModelRGB(BaseModel):
    """OMGAN model for image harmonization."""
    """OM model for image harmonization."""
    def __init__(self, opt):
        super(OMModelRGB,self).__init__(opt)
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.load()
        self.net_g.train()
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
        

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        

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
    def feed_data(self, data):
        self.mask = data['mask'].to(self.device)
        self.real = data['real'].to(self.device)
        self.comp = data['comp'].to(self.device)


    def optimize_parameters(self, current_iter):
        # optimize net_g

        self.optimizer_g.zero_grad()
        self.input = self.comp 

        self.harmonized = self.net_g(self.input)
        r, g, b = self.harmonized.split(1,dim=1)
        r = torch.clamp(r, -1,1)
        g = torch.clamp(g, -1,1)
        b = torch.clamp(b, -1,1)
        self.harmonized = torch.cat([r,g,b],dim=1)

        
        img_mean = [0.5, 0.5, 0.5]
        img_std = [0.5, 0.5, 0.5]
        
        tmp_real = ((self.real.permute(0,2,3,1)[0,:,:,:].cpu().numpy()*img_std+img_mean)*255).astype("uint8")   
        tmp_real = cv2.cvtColor(tmp_real, cv2.COLOR_RGB2BGR).astype("uint8")
        
        tmp_comp = ((self.comp.permute(0,2,3,1)[0,:,:,:].cpu().numpy()*img_std+img_mean)*255).astype("uint8")   
        tmp_comp = cv2.cvtColor(tmp_comp, cv2.COLOR_RGB2BGR).astype("uint8")   
        tmp_harm = ((self.harmonized.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy()*img_std+img_mean)*255).astype("uint8")   
        tmp_harm = cv2.cvtColor(tmp_harm, cv2.COLOR_RGB2BGR).astype("uint8")   
        

        tmp_mask = self.mask.permute(0,2,3,1)[0,:,:,:].detach().cpu().numpy().repeat(3,axis=2)*255

        tmp_img = np.concatenate([tmp_real,tmp_comp, tmp_harm, tmp_mask],1) 
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
            

        l_g_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)





    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.attention_mask = self.net_g_ema(self.comp)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.input = self.comp
                self.harmonized = self.net_g(self.input)
                r, g, b = self.harmonized.split(1,dim=1)
                r = torch.clamp(r, -1,1)
                g = torch.clamp(g, -1,1)
                b = torch.clamp(b, -1,1)
                self.harmonized = torch.cat([r,g,b],dim=1)
                

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

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['mask_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            ham = tensor2imgRGB([visuals['ham']])
            real = tensor2imgRGB([visuals['real']])
            comp = tensor2imgRGB([visuals['comp']])
            
            mask = tensor2imgRGB([visuals['mask']])

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
                # if output is empty:
                #    imwrite(output, save_img_path) 
            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    metric_data = dict(img1=ham, img2=real)
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
                
                # exit()
            pbar.update(1)
            pbar.set_description(f'Test {img_name}')
        pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

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
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
    def load(self):
    # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)
