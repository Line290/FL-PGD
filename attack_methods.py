import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import *
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
import utils
import math

from utils import softCrossEntropy
from utils import one_hot_tensor, label_smoothing
import ot
import pickle

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attack_None(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_None, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.basic_net = basic_net
        print(config)

    def forward(self, inputs, targets, attack=None, batch_idx=-1):
        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        outputs, _ = self.basic_net(inputs)
        return outputs, None


class Attack_PGD(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_PGD, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None

        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net.eval()
        logits_pred_nat = aux_net(inputs)[0]
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        outputs = aux_net(inputs)[0]
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        # x_org = x.detach()
        # loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x)[0]
            loss = self.loss_func(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]

        return logits_pert, targets_prob.detach()


class Attack_nat(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_nat, self).__init__()
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
        ) else config['loss_func']
        self.basic_net = basic_net
        print(config)

    def forward(self, inputs, targets, attack=True, batch_idx=-1):
        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()
        outputs, _ = self.basic_net(inputs)
        self.basic_net.zero_grad()
        adv_loss = self.loss_func(outputs, targets.detach())
        return outputs, adv_loss


class Attack_Madry(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_Madry, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None

        # if self.box_type == 'white':
        #     aux_net = pickle.loads(pickle.dumps(self.basic_net))
        # elif self.box_type == 'black':
        #     assert self.attack_net is not None, "should provide an additional net in black-box case"
        #     aux_net = pickle.loads(pickle.dumps(self.basic_net))
        # aux_net.eval()
        self.basic_net.eval()
        # logits_pred_nat = aux_net(inputs)[0]
        # targets_prob = F.softmax(logits_pred_nat.float(), dim=1)
        #
        # num_classes = targets_prob.size(1)
        #
        # outputs = aux_net(inputs)[0]
        # targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets
        step_sign = 1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        # x_org = x.detach()
        # loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            # aux_net.eval()
            self.basic_net.eval()
            # logits = aux_net(x)[0]
            logits = self.basic_net(x)[0]
            loss = self.loss_func(logits, y_tensor_adv)
            loss = loss.mean()
            # aux_net.zero_grad()
            self.basic_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]
        self.basic_net.zero_grad()
        adv_loss = self.loss_func(logits_pert, targets.detach())
        return logits_pert, adv_loss


class Attack_BAT(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_BAT, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.multi = config['multi']
        self.loss_func = torch.nn.CrossEntropyLoss(
            reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        print(config)

    def adv_labels(self, nat_prob, y_batch, gamma=0.01):
        num_classes = nat_prob.size(1)
        nat_prob = nat_prob.detach().cpu().numpy()
        y_batch = y_batch.detach().cpu().numpy()
        L = -np.log(nat_prob + 1e-8)  # log-likelihood
        LL = np.copy(L)
        LL[np.arange(y_batch.shape[0]), y_batch] = 1e4
        minval = np.min(LL, axis=1)
        LL[np.arange(y_batch.shape[0]), y_batch] = -1e4
        maxval = np.max(LL, axis=1)

        denom = np.sum(L, axis=1) - L[np.arange(y_batch.shape[0]), y_batch] - (
                num_classes - 1) * (minval - gamma)
        delta = 1 / (1 + self.multi * (maxval - minval + gamma) / denom)
        alpha = delta / denom

        y_batch_adv = np.reshape(
            alpha, [-1, 1]) * (L - np.reshape(minval, [-1, 1]) + gamma)
        y_batch_adv[np.arange(y_batch.shape[0]), y_batch] = 1.0 - delta

        return torch.from_numpy(y_batch_adv).cuda()

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs = self.basic_net(inputs)[0]
            return outputs, None

        # if self.box_type == 'white':
        #     aux_net = pickle.loads(pickle.dumps(self.basic_net))
        # elif self.box_type == 'black':
        #     assert self.attack_net is not None, "should provide an additional net in black-box case"
        #     aux_net = pickle.loads(pickle.dumps(self.basic_net))
        # aux_net.eval()
        loss_ce = softCrossEntropy()
        self.basic_net.eval()
        logits_pred_nat = self.basic_net(inputs)[0]
        prob_pred_nat = F.softmax(logits_pred_nat, dim=1)
        y_batch_adv = self.adv_labels(prob_pred_nat, targets)
        # print(y_batch_adv, y_batch_adv.sum(1))
        logits_pred_nat[torch.arange(0, targets.size(0)), targets] = -1e4
        y_tensor_adv = torch.argmax(logits_pred_nat, axis=1)
        # targets_prob = F.softmax(logits_pred_nat.float(), dim=1)
        #
        # num_classes = targets_prob.size(1)
        #
        # outputs = aux_net(inputs)[0]
        # targets_prob = F.softmax(outputs.float(), dim=1)
        # y_tensor_adv = targets
        step_sign = -1.0

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        # x_org = x.detach()
        # loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)
            # aux_net.eval()
            self.basic_net.eval()
            # logits = aux_net(x)[0]
            logits = self.basic_net(x)[0]
            loss = self.loss_func(logits, y_tensor_adv)
            loss = loss.mean()
            # aux_net.zero_grad()
            self.basic_net.zero_grad()
            loss.backward()

            x_adv = x.data + step_sign * self.step_size * torch.sign(
                x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach())[0]
        self.basic_net.zero_grad()
        adv_loss = loss_ce(logits_pert, y_batch_adv.detach())
        return logits_pert, adv_loss

class Attack_FeaScatter(nn.Module):
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_FeaScatter, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']
        self.ls_factor = 0.1 if 'ls_factor' not in config.keys(
        ) else config['ls_factor']

        print(config)

    def forward(self,
                inputs,
                targets,
                attack=True,
                targeted_label=-1,
                batch_idx=0):

        if not attack:
            outputs, _ = self.basic_net(inputs)
            return outputs, None
        if self.box_type == 'white':
            aux_net = pickle.loads(pickle.dumps(self.basic_net))
        elif self.box_type == 'black':
            assert self.attack_net is not None, "should provide an additional net in black-box case"
            aux_net = pickle.loads(pickle.dumps(self.basic_net))

        aux_net.eval()
        batch_size = inputs.size(0)
        m = batch_size
        n = batch_size

        # logits = aux_net(inputs)[0]
        # num_classes = logits.size(1)

        # outputs = aux_net(inputs)[0]
        # targets_prob = F.softmax(outputs.float(), dim=1)
        # y_tensor_adv = targets
        # step_sign = 1.0

        x = inputs.detach()

        # x_org = x.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pred_nat, fea_nat = aux_net(inputs)

        num_classes = logits_pred_nat.size(1)
        y_gt = one_hot_tensor(targets, num_classes, device)

        loss_ce = softCrossEntropy()

        iter_num = self.num_steps

        for i in range(iter_num):
            x.requires_grad_()
            zero_gradients(x)
            if x.grad is not None:
                x.grad.data.fill_(0)

            logits_pred, fea = aux_net(x)

            ot_loss = ot.sinkhorn_loss_joint_IPOT(1, 0.00, logits_pred_nat,
                                                  logits_pred, None, None,
                                                  0.01, m, n)

            aux_net.zero_grad()
            adv_loss = ot_loss
            adv_loss.backward(retain_graph=True)
            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)
            x = Variable(x_adv)

            logits_pred, fea = self.basic_net(x)
            self.basic_net.zero_grad()

            y_sm = utils.label_smoothing(y_gt, y_gt.size(1), self.ls_factor)

            adv_loss = loss_ce(logits_pred, y_sm.detach())

        return logits_pred, adv_loss
