from torch.utils import data as data
from torchvision.transforms.functional import normalize
import cv2
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file,paired_paths_from_txt
from basicsr.data.transforms import augment, paired_random_crop, img_resize
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.instgram_rgb import Perturb_Simulator
import random



@DATASET_REGISTRY.register()
class MultiMaskRGBImageDataset(data.Dataset):
    """Paired image dataset for image harmonization.

    Read Mask and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(MultiMaskRGBImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.data_root = opt['dataroot_path']
        self.gt_size = self.opt['gt_size']
        self.perturb = Perturb_Simulator(self.gt_size)
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'meta_info_file':
            
            self.paths = paired_paths_from_txt(self.opt['dataroot_path'])
        
    def __getitem__(self, index):
        if self.file_client is None:
            
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load gt and mask images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.


        real_path = self.paths[index]['real_path']
        real_bytes = self.file_client.get(real_path, 'real')
        real = imfrombytes(real_bytes)

        mask_path = self.paths[index]['mask_path']
        mask_bytes = self.file_client.get(mask_path, 'mask')
        mask = imfrombytes(mask_bytes)
        


        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
              
        real, comp, mask = self.perturb.perturb(real, mask)
        real, comp, mask= img2tensor([real, comp, mask])
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(real, self.mean, self.std, inplace=True)
            normalize(comp, self.mean, self.std, inplace=True)
        return {'mask': mask, 'real': real, 'comp': comp, 'mask_path': mask_path, 'real_path': real_path}

    def __len__(self):
        return len(self.paths)
