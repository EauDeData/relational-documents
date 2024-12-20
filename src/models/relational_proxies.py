import os
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import miners, losses
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from pytorch_metric_learning.distances import LpDistance, BatchedDistance
from networks.ast import AST
from networks.relation_net import RelationNet
from utils_jeta import constants


class RelationalProxies(nn.Module):
    def __init__(self, backbone, logdir, logger=None):
        super(RelationalProxies, self).__init__()
        # self.num_classes = num_classes
        self.feature_dim = constants.FEATURE_DIM
        self.lr = constants.INIT_LR
        self.logger = logger

        self.backbone = backbone
        self.aggregator = AST(num_inputs=backbone.num_local, dim=self.feature_dim, depth=3, heads=3, mlp_dim=256)
        self.relation_net = RelationNet(feature_dim=self.feature_dim)
        self.wtFrac_local, self.wtFrac_global, self.wtFrac_relational = 1, 1, 3

        self.optimizer = torch.optim.Adam(chain(
            backbone.parameters(), self.aggregator.parameters(), self.relation_net.parameters()),
            lr=self.lr, weight_decay=constants.WEIGHT_DECAY)

        self.scheduler = MultiStepLR(self.optimizer, milestones=constants.LR_MILESTONES, gamma=constants.LR_DECAY_RATE)
        # self.proxy_criterion = ProxyAnchorLoss(num_classes=num_classes, embedding_size=self.feature_dim)
        self.margin = .02
        self.metric_learning_criterion = losses.TripletMarginLoss(margin = self.margin ) # TODO: Aquí afegir el metric learning amb el text encoder
        self.metric_learning_miner = miners.TripletMarginMiner(margin = self.margin )

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

            global_repr, summary_repr, relation_repr, query_repr = self.compute_reprs(image, crops, query)

            # Whatever
            # TODO: Triplet loss with mining so we don't have to create negatives
            labels = torch.tensor(list(range(query_repr.shape[0])) * 2, dtype = torch.int64, device = query_repr.device)
            total_reprs = torch.cat((relation_repr, query_repr), dim = 0)

            miner_output = self.metric_learning_miner(total_reprs, labels)
            loss = self.metric_learning_criterion(total_reprs, labels, miner_output)

            loss.backward()
            self.optimizer.step()
            if not self.logger is None:
                self.logger.log({'train_loss': loss.item()})
            epoch_state['loss'] += loss.item()
            # epoch_state = self.predict(global_repr, summary_repr, relation_repr, labels, epoch_state)

        self.post_epoch('Train', epoch, epoch_state, len(trainloader.dataset), save_path)

    @torch.no_grad()
    def test(self, testloader, epoch):
        if epoch % constants.TEST_EVERY == 0:
            self.eval()
            device = self.backbone.DEVICE  # hacky, but keeps the arg list clean

            # Same as the loss: https://kevinmusgrave.github.io/pytorch-metric-learning/losses/#tripletmarginloss
            batched_distance_fn = LpDistance(normalize_embeddings=True, p=2, power=1)
            metrics = {'doc2query': {'acc@1': 0, 'acc@5': 0, 'acc@10': 0},
                       'query2doc': {'acc@1': 0, 'acc@5': 0, 'acc@10': 0}}

            # Loop through test data in batches
            with torch.no_grad():
                for data in testloader:
                    image, crops, query = data
                    image, crops, query = image.to(device), crops.to(device), query.to(device)

                    perfect_indices = torch.arange(image.shape[0], device = image.device) # Batch size

                    _, _, relation_repr, query_repr = self.compute_reprs(image, crops, query)

                    # Compute cosine similarity matrix for the current batch
                    similarity_matrix = batched_distance_fn(relation_repr, query_repr)

                    # Doc2Query
                    d2q_acc_at_1, d2q_acc_at_5, d2q_acc_at_10 = self.predict(similarity_matrix, perfect_indices)
                    metrics['doc2query']['acc@1'] += d2q_acc_at_1
                    metrics['doc2query']['acc@5'] += d2q_acc_at_5
                    metrics['doc2query']['acc@10'] += d2q_acc_at_10


                    q2d_acc_at_1, q2d_acc_at_5, q2d_acc_at_10 = self.predict(similarity_matrix.T, perfect_indices)
                    metrics['query2doc']['acc@1'] += q2d_acc_at_1
                    metrics['query2doc']['acc@5'] += q2d_acc_at_5
                    metrics['query2doc']['acc@10'] += q2d_acc_at_10

            metrics['doc2query']['acc@1'] /= (len(testloader) * image.shape[0]) # Number of availible hits
            metrics['doc2query']['acc@5'] /= (len(testloader) * image.shape[0]) # Number of availible hits
            metrics['doc2query']['acc@10'] /= (len(testloader) * image.shape[0]) # Number of availible hits

            metrics['query2doc']['acc@1'] /= (len(testloader) * image.shape[0]) # Number of availible hits
            metrics['query2doc']['acc@5'] /= (len(testloader) * image.shape[0]) # Number of availible hits
            metrics['query2doc']['acc@10'] /= (len(testloader) * image.shape[0]) # Number of availible hits
            if not self.logger is None:
                self.logger.log(metrics)
            print(metrics)
            return metrics

    def compute_reprs(self, image, crops, query):
        global_embed, local_embeds, query_embedding = self.backbone(image, crops, query)

        summary_repr = self.aggregator(local_embeds)
        relation_repr = self.relation_net(global_embed, summary_repr)

        return global_embed, summary_repr, relation_repr, query_embedding

    def predict(self, distance_matrix, perfect_indices):

        ranking = torch.argsort(distance_matrix, dim=1)[:, 0]
        absolute_differences = (ranking - perfect_indices).abs()

        hits_at_1 = torch.sum((absolute_differences == 0).to(torch.int64)).item()
        hits_at_5 = torch.sum((absolute_differences < 5).to(torch.int64)).item()
        hits_at_10 = torch.sum((absolute_differences < 10).to(torch.int64)).item()

        return hits_at_1, hits_at_5, hits_at_10

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

class WordCropsOnly(RelationalProxies):

    def compute_reprs(self, image, crops, query):
        global_embed, local_embeds, query_embedding = self.backbone(image, crops, query)
        summary_repr = self.aggregator(local_embeds)

        # Substitute relation_repr --> summary_repr
        return global_embed, summary_repr, summary_repr, query_embedding

class RelationalWordCropsOnly(RelationalProxies):

    def compute_reprs(self, image, crops, query):
        global_embed, local_embeds, query_embedding = self.backbone(image, crops, query)
        summary_repr = self.aggregator(local_embeds)

        # Substitute relation_repr --> summary_repr
        # Self-Aggregation, should only add depth
        relation_repr = self.relation_net(summary_repr, summary_repr)

        return global_embed, summary_repr, relation_repr, query_embedding

class GLobalReprOnly(RelationalProxies):

    def compute_reprs(self, image, crops, query):
        global_embed, local_embeds, query_embedding = self.backbone(image, crops, query)

        # Substitute relation_repr --> global_embed
        return global_embed, None, global_embed, query_embedding