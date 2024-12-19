import os
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

from data.collate import collate_fn
from models.relational_proxies import RelationalProxies
from networks.encoder import DisjointEncoder
from utils_jeta import constants
from utils_jeta.auto_load_resume import auto_load_resume
from data.boe import BOEDataset, MAX_WORDS
import wandb

class Initializers:
    def __init__(self, args):
        self.args = args
        self.device = None
        self.model = None
        wandb.init(project='relational_documents', config=args)

    def env(self):
        args = self.args
        # Manual seed
        if args.seed >= 0:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("[INFO] Setting SEED: " + str(args.seed))
        else:
            print("[INFO] Setting SEED: None")

        if not torch.cuda.is_available():
            print("[WARNING] CUDA is not available.")
        else:
            print("[INFO] Found " + str(torch.cuda.device_count()) + " GPU(s) available.")
            self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
            print("[INFO] Device type: " + str(self.device))

    def data(self):
        args = self.args
        print('[INFO] Dataset: {}'.format(args.dataset))

        train_set = BOEDataset('train.txt')
        test_set = BOEDataset('test.txt')

        trainloader = DataLoader(train_set, batch_size=constants.TRAIN_BATCH_SIZE,
                                 shuffle=True,
                                 pin_memory=True,
                                 num_workers=args.num_workers,
                                 drop_last=False,
                                 collate_fn=collate_fn)
        testloader = DataLoader(test_set, batch_size=constants.TEST_BATCH_SIZE,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=args.num_workers,
                                 drop_last=True, # Otherwise it can fuck-up acc@{5,10}
                                 collate_fn=collate_fn)

        print('Done', flush=True)

        return args, trainloader, testloader

    def modeltype(self):
        args, device = self.args, self.device
        # Get the pretrained backbone for extracting global-views
        # BERT NUM OF TOKENS
        backbone = DisjointEncoder(vocab_size=30522, device=device)
        backbone.num_local = MAX_WORDS # Hacky but fuck it
        print("[INFO]", str(str(constants.BACKBONE)), "loaded in memory.")

        if args.logdir is None:
            logdir = os.path.join(args.checkpoint, args.dataset, 'logdir')
        else:
            logdir = args.logdir
        model = RelationalProxies(backbone, logdir, logger=wandb)
        print('[INFO] Model: Relational Proxies')
        model.to(device)
        self.model = model

        return model

    def checkpoint(self):
        args, model = self.args, self.model
        save_path = os.path.join(args.checkpoint, args.dataset)
        if args.pretrained and os.path.exists(save_path):
            start_epoch, lr = auto_load_resume(model, save_path, status='train')
            assert start_epoch < constants.END_EPOCH
            model.lr = lr
            model.start_epoch = start_epoch
        else:
            os.makedirs(save_path, exist_ok=True)
            start_epoch = 0
        return save_path, start_epoch


