import torch
import torchvision
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


class Loss(object):
    """
    Co-Pencil Loss Class
    Co-Pencil 알고리즘에서 사용하는 다양한 손실 함수들을 정의한 클래스입니다.
    - PENCIL Loss: Label Correction(라벨 수정)을 위한 손실 함수 (KL Div + Entropy + Compatibility)
    - Co-teaching Loss: 두 모델이 서로 선택한 샘플로 학습하는 손실 함수
    """

    def __init__(self, args):
        self.args = args
        self.logsoftmax = nn.LogSoftmax(dim=1).to(self.args.device)
        self.softmax = nn.Softmax(dim=1).to(self.args.device)

    # PENCIL Loss Functions: Label Correction (라벨 수정) 관련 함수들
    
    # KL Divergence 손실 (분류 손실)
    def _pencil_loss_kl(self, X, Y, reduction='mean'):
        # X: 모델 예측값 (Logits), Y: 라벨 분포 (y_tilde)
        if reduction == 'mean':
            return torch.mean(self.softmax(X) * (self.logsoftmax(X) - torch.log((Y))))
        elif reduction == 'none':
            return torch.mean(self.softmax(X) * (self.logsoftmax(X) - torch.log((Y))), dim=1)
        elif reduction == 'sum':
            return torch.sum(self.softmax(X) * (self.logsoftmax(X) - torch.log((Y))))
        else:
            return torch.mean(self.softmax(X) * (self.logsoftmax(X) - torch.log((Y))))

    # Entropy 손실 (불확실성 낮추기, 예측 확신 높이기)
    def _pencil_loss_entropy(self, X, reduction='mean'):
        if reduction == 'mean':
            return - torch.mean(torch.mul(self.softmax(X), self.logsoftmax(X)))
        elif reduction == 'none':
            return - torch.mean(torch.mul(self.softmax(X), self.logsoftmax(X)), dim=1)
        elif reduction == 'sum':
            return - torch.sum(torch.mul(self.softmax(X), self.logsoftmax(X)))
        else:
            return - torch.mean(torch.mul(self.softmax(X), self.logsoftmax(X)))

    # Compatibility 손실 (현재 라벨 분포와 초기 라벨 또는 타겟과의 호환성)
    def _pencil_loss_compatibility(self, Y, T, reduction='mean'):
        return F.cross_entropy(Y, T, reduction=reduction)

    def _pencil_loss(self, X, last_y_var_A, alpha, beta, reduction='mean', target_var=None):
        # 최종 PENCIL 손실 계산: KL Div + alpha * Compatibility + beta * Entropy
        assert not target_var == None
        # lc: Classification Loss (KL Divergence)
        lc = self._pencil_loss_kl(X, last_y_var_A, reduction=reduction)
        # le: Entropy Loss
        le = self._pencil_loss_entropy(X, reduction=reduction)
        # lo: Compatibility Loss
        lo = self._pencil_loss_compatibility(last_y_var_A, target_var, reduction=reduction)
        return lc + alpha * lo + beta * le

    def _get_loss(self, X, Y, loss_type='CE', reduction='mean', **kwargs):
        if loss_type == 'CE':
            loss = F.cross_entropy(X, Y, reduction=reduction)
        elif loss_type == 'KL':
            loss = F.kl_div(X, Y, reduction=reduction)
        elif loss_type == "PENCIL_KL":
            loss = self._pencil_loss_kl(X, Y, reduction=reduction)
        elif loss_type == 'PENCIL':
            loss = self._pencil_loss(X, Y, alpha=self.args.alpha, beta=self.args.beta, reduction=reduction, **kwargs)
        else:
            loss = F.cross_entropy(X, Y, reduction=reduction)
        return loss

    def _sort_by_loss(self, predict, target, loss_type='CE', index=True, **kwargs):
        loss = self._get_loss(predict, target, loss_type=loss_type, reduction='none', **kwargs)
        index_sorted = torch.argsort(loss.data.cpu()).numpy()
        return index_sorted if index else predict[index_sorted]

    # Loss functions
    def loss_coteaching(self, y_1, y_2, t_1, t_2, forget_rate, loss_type='CE', ind=[], noise_or_not=[],
                        target_var=None, parallel=False, softmax=False):
        """
        Co-teaching Loss 함수
        두 네트워크(NetA, NetB)가 서로 Small Loss 샘플을 선택하여 교차 학습합니다.
        
        작동 원리:
        1. 각 네트워크가 배치 내 샘플들의 손실(Loss)을 계산
        2. 손실이 작은 순서대로 정렬 (Small Loss Trick)
        3. 전체 데이터 중 (1 - forget_rate) 비율만큼의 '깨끗한(Clean)' 것으로 추정되는 샘플 선택
        4. 서로 선택한 샘플에 대해서 상대방 네트워크를 업데이트 (Exchange)
        
        :param y_1, y_2: 모델 1, 2의 예측값
        :param t_1, t_2: 타겟 라벨
        :param forget_rate: 현재 에폭에서의 망각 비율 (노이즈로 간주하여 버릴 비율)
        """
        if softmax:
            y_1 = self.softmax(y_1)
            y_2 = self.softmax(y_2)

        # compute NetA prediction loss
        ind_1_sorted = self._sort_by_loss(y_1, t_1, loss_type=loss_type, index=True, target_var=target_var)

        # compute NetB prediction loss
        ind_2_sorted = self._sort_by_loss(y_2, t_2, loss_type=loss_type, index=True, target_var=target_var)

        # catch R(t)% samples
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(ind_1_sorted))

        # caculate how many pure sample in selected batch
        pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(num_remember)
        pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(num_remember)
        pure_ratio_discard_1 = np.sum(noise_or_not[ind[ind_1_sorted[num_remember:]]]) / float(num_remember)
        pure_ratio_discard_2 = np.sum(noise_or_not[ind[ind_2_sorted[num_remember:]]]) / float(num_remember)

        ind_1_update = ind_1_sorted[:num_remember]
        ind_2_update = ind_2_sorted[:num_remember]
        ind_1_discard = ind_1_sorted[num_remember:]
        ind_2_discard = ind_2_sorted[num_remember:]

        # exchange
        if parallel:
            loss_1_update = self._get_loss(y_1[ind_2_update], t_1[ind_2_update], loss_type=loss_type, reduction='none',
                                           target_var=None if target_var == None else target_var[ind_2_update])
            loss_2_update = self._get_loss(y_2[ind_1_update], t_2[ind_1_update], loss_type=loss_type, reduction='none',
                                           target_var=None if target_var == None else target_var[ind_1_update])
        else:
            loss_1_update = self._get_loss(y_1[ind_2_update], t_2[ind_2_update], loss_type=loss_type, reduction='none',
                                           target_var=None if target_var == None else target_var[ind_2_update])
            loss_2_update = self._get_loss(y_2[ind_1_update], t_1[ind_1_update], loss_type=loss_type, reduction='none',
                                           target_var=None if target_var == None else target_var[ind_1_update])


        return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember, ind_1_update, ind_2_update,\
               ind_1_discard, ind_2_discard, pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2

    def loss_coteaching_plus(self, y_1, y_2, t_1, t_2, forget_rate, step, ind=[], loss_type='CE',
                             noise_or_not=[], target_var=None, parallel=False, softmax=True):
        """
        Co-teaching+ Loss 함수
        Co-teaching의 개선판으로, 두 모델의 예측이 '불일치(Disagree)'하는 샘플 중에서
        Small Loss 샘플을 선택하여 학습합니다.
        """
        if softmax:
            outputs = F.softmax(y_1, dim=1)
            outputs2 = F.softmax(y_2, dim=1)
        else:
            outputs = y_1
            outputs2 = y_2

        _, pred1 = torch.max(y_1.data, 1)
        _, pred2 = torch.max(y_2.data, 1)

        disagree_id = torch.where(pred1 == pred2)[0].cpu().numpy()
        ind_disagree = ind[disagree_id]

        if len(disagree_id)*(1-forget_rate) >= 1:
            update_label_1 = t_1[disagree_id]
            update_label_2 = t_2[disagree_id]
            update_outputs = outputs[disagree_id]
            update_outputs2 = outputs2[disagree_id]

            # if not target_var == None:
            update_target_var = target_var[disagree_id] if not target_var == None else None

            loss_1, loss_2, _ind_1_update, _ind_2_update, _ind_1_discard, _ind_2_discard, \
            pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching(
                update_outputs, update_outputs2, update_label_1, update_label_2, forget_rate, loss_type, ind_disagree,
                noise_or_not, target_var=update_target_var, parallel=parallel)

            # predict same sample will be discard
            ind_1_update = disagree_id[_ind_1_update]
            ind_2_update = disagree_id[_ind_2_update]
            ind_1_discard = disagree_id[_ind_1_discard]
            ind_2_discard = disagree_id[_ind_2_discard]
        else:
            update_label_1 = t_1
            update_label_2 = t_2
            update_outputs = outputs
            update_outputs2 = outputs2

            logical_disagree_id = torch.zeros(t_1.shape[0], dtype=torch.bool)
            logical_disagree_id[disagree_id] = True
            update_step = logical_disagree_id | (step < 5000)
            update_step = update_step.type(torch.float32).to(self.args.device)

            l1 = self._get_loss(update_outputs, update_label_1, loss_type=loss_type, reduction='mean', target_var=target_var)
            l2 = self._get_loss(update_outputs2, update_label_2, loss_type=loss_type, reduction='mean', target_var=target_var)

            loss_1 = torch.sum(update_step * l1) / t_1.shape[0]
            loss_2 = torch.sum(update_step * l2) / t_2.shape[0]

            ones = torch.ones(update_step.shape, dtype=torch.float32, device=self.args.device)
            zeros = torch.zeros(update_step.shape, dtype=torch.float32, device=self.args.device)
            ind_1_update  = ind_2_update = torch.where(update_step == ones)[0].cpu().numpy()
            ind_1_discard = ind_2_discard = torch.where(update_step == zeros)[0].cpu().numpy()

            pure_ratio_1 = np.sum(noise_or_not[ind]) / ind.shape[0]
            pure_ratio_2 = np.sum(noise_or_not[ind]) / ind.shape[0]
            pure_ratio_discard_1 = -1
            pure_ratio_discard_2 = -1

        # return loss_1, loss_2, pure_ratio_1, pure_ratio_2
        return loss_1, loss_2, ind_1_update, ind_2_update, ind_1_discard, ind_2_discard,\
               pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2

def np_softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s