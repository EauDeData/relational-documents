import os
from collections import OrderedDict

import torch

from utils_jeta.constants import INIT_LR


def auto_load_resume(model, path, status):
    """
    Adopted from MMAL --
    Paper: https://arxiv.org/pdf/2003.09150.pdf
    Code: https://github.com/ZF4444/MMAL-Net
    """
    if status == 'train':
        pth_files = os.listdir(path)
        nums_epoch = [int(name.replace('epoch', '').replace('.pth', '')) for name in pth_files if '.pth' in name]
        if len(nums_epoch) == 0:
            return 0, INIT_LR
        else:
            max_epoch = max(nums_epoch)
            pth_path = os.path.join(path, 'epoch' + str(max_epoch) + '.pth')
            print('Load model from', pth_path)
            checkpoint = torch.load(pth_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            lr = checkpoint['learning_rate']
            print('Resume from %s' % pth_path)
            return epoch, lr
    elif status == 'test':
        print('Load model from', path)
        checkpoint = torch.load(path, map_location='cpu')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            if 'module.' == k[:7]:
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        epoch = checkpoint['epoch']
        print('Resume from %s' % path)
        return epoch
