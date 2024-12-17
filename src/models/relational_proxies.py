import os
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning.losses import ProxyAnchorLoss
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from networks.ast import AST
from networks.relation_net import RelationNet
from utils_jeta import constants


class RelationalProxies(nn.Module):
    def __init__(self, backbone, logdir):
        super(RelationalProxies, self).__init__()
        # self.num_classes = num_classes
        self.feature_dim = constants.FEATURE_DIM
        self.lr = constants.INIT_LR

        self.backbone = backbone
        self.aggregator = AST(num_inputs=backbone.num_local, dim=self.feature_dim, depth=3, heads=3, mlp_dim=256)
        self.relation_net = RelationNet(feature_dim=self.feature_dim)
        self.wtFrac_local, self.wtFrac_global, self.wtFrac_relational = 1, 1, 3

        self.optimizer = torch.optim.SGD(chain(
            backbone.parameters(), self.aggregator.parameters(), self.relation_net.parameters()),
            lr=self.lr, momentum=constants.MOMENTUM, weight_decay=constants.WEIGHT_DECAY)

        self.scheduler = MultiStepLR(self.optimizer, milestones=constants.LR_MILESTONES, gamma=constants.LR_DECAY_RATE)
        self.criterion = nn.CrossEntropyLoss()
        # self.proxy_criterion = ProxyAnchorLoss(num_classes=num_classes, embedding_size=self.feature_dim)
        self.metric_learning_criterion = None # TODO: Aquí afegir el metric learning amb el text encoder

        self.writer = SummaryWriter(logdir)

    def train_one_epoch(self, trainloader, epoch, save_path):
        print('Training %d epoch' % epoch)
        self.train()
        device = self.backbone.DEVICE  # hacky, but keeps the arg list clean
        epoch_state = {'loss': 0, 'correct': 0}
        for i, data in enumerate(tqdm(trainloader)):
            image, crops, query = data
            image, crops, query = image.to(device), crops.to(device), query.to(device)

            self.optimizer.zero_grad()

            global_repr, summary_repr, relation_repr = self.compute_reprs(image, crops, query)
            # Whatever
            # TODO: Triplet loss with mining so we don't have to create negatives
            loss = self.metric_learning_criterion(relation_repr, labels)

            loss.backward()
            self.optimizer.step()

            epoch_state['loss'] += loss.item()
            epoch_state = self.predict(global_repr, summary_repr, relation_repr, labels, epoch_state)

        self.post_epoch('Train', epoch, epoch_state, len(trainloader.dataset), save_path)

    @torch.no_grad()
    def test(self, testloader, epoch):
        if epoch % constants.TEST_EVERY == 0:
            print('Testing %d epoch' % epoch)
            self.eval()

            # TODO: Aquí fer un test de retrieval com deu mana

            self.post_epoch('Test', epoch, None, len(testloader.dataset), None)

    def compute_reprs(self, image, crops, query):
        global_embed, local_embeds, query_embedding = self.backbone(image, crops, query)

        summary_repr = self.aggregator(local_embeds)
        relation_repr = self.relation_net(global_embed, summary_repr)

        return global_embed, summary_repr, relation_repr

    @torch.no_grad()
    def predict(self, global_repr, summary_repr, relation_repr, labels, epoch_state):

        # TODO: Aquí fer un predict de retrieval com deu mana

        return epoch_state

    @torch.no_grad()
    def post_epoch(self, phase, epoch, epoch_state, num_samples, save_path):
        accuracy = epoch_state['correct'] / num_samples
        loss = epoch_state['loss']

        print(f'{phase} Loss: {loss}')
        print(f'{phase} Accuracy: {accuracy * 100}%')
        self.writer.add_scalar(f'Loss/{phase}', loss, epoch)
        self.writer.add_scalar(f'Accuracy/{phase}', accuracy, epoch)

        if (phase == 'Train') and ((epoch % constants.SAVE_EVERY == 0) or (epoch == constants.END_EPOCH)):
            self.scheduler.step()
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.state_dict(),
                'learning_rate': self.lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))

    def post_job(self):
        """Post-job actions"""
        self.writer.flush()
        self.writer.close()
