import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
import numpy as np
import torch.nn as nn

def class_projection(self, output, target):
    ds_level = len(output)
    batch_size = output[0].shape[0]
    new_output = []
    new_target = []
    multi_dataset_class_ind_start_end = self.plans['multi_dataset_class_ind_start_end']

    for b in range(batch_size):
        for i in range(ds_level):
            class_value = target[i][b:b + 1].unique()
            if (class_value >= multi_dataset_class_ind_start_end[0][1]).any():
                class_value = class_value[class_value != 0]
            for class_ind_start_end in multi_dataset_class_ind_start_end:
                if ((class_ind_start_end[0] <= class_value).all() and (class_value < class_ind_start_end[1]).all()):
                    new_output.append(output[i][b:b + 1][:, class_ind_start_end[0]:class_ind_start_end[1]])
                    new_target.append(torch.clamp(target[i][b:b + 1] - class_ind_start_end[0], 0))
                    continue

    return new_output, new_target

class MultiClassDeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        Wraps a loss function so that it can be applied to multiple outputs. Forward accepts an arbitrary number of
        inputs. Each input is expected to be a tuple/list. Each tuple/list must have the same length. The loss is then
        applied to each entry like this:
        l = w0 * loss(input0[0], input1[0], ...) +  w1 * loss(input0[1], input1[1], ...) + ...
        If weights are None, all w will be 1.
        """
        super(MultiClassDeepSupervisionWrapper, self).__init__()
        assert any([x != 0 for x in weight_factors]), "At least one weight factor should be != 0.0"
        self.weight_factors = tuple(weight_factors)
        self.loss = loss
        self.dataset_labels_description = [[0, 2], [2, 8], [8,23], [23,28], [28, 30], [30,33], [33, 35], [35,38], [38, 40],[40,56]]

    def forward(self, *args):
        assert all([isinstance(i, (tuple, list)) for i in args]), \
            f"all args must be either tuple or list, got {[type(i) for i in args]}"
        # we could check for equal lengths here as well, but we really shouldn't overdo it with checks because
        # this code is executed a lot of times!

        if self.weight_factors is None:
            weights = (1, ) * len(args[0])
        else:
            weights = self.weight_factors

        # we mask the output and target to only include the classes of the current dataset
        input = args[0]
        target = args[1]

        labels_per_image_list = [target[0][i:i+1].unique() for i in range(target[0].shape[0])]
        for i in range(len(labels_per_image_list)):
            if (labels_per_image_list[i] >= self.dataset_labels_description[0][1]).any():
                labels_per_image_list[i] = labels_per_image_list[i][labels_per_image_list[i] != 0]

        dataset_ind_list = []
        for labels_per_image in labels_per_image_list:
            for dataset_ind, labels_description in enumerate( self.dataset_labels_description):
                if ((labels_description[0] <= labels_per_image).all() and (labels_per_image < labels_description[1]).all()):
                    dataset_ind_list.append(dataset_ind)

        for image_ind,dataset_ind in enumerate( dataset_ind_list):
            dataset_ind_first_label_ind = self.dataset_labels_description[dataset_ind][0]
            dataset_ind_last_label_ind = self.dataset_labels_description[dataset_ind][1]

            for i in range(len(input)):
                output[i][image_ind:image_ind+1] = output[i][image_ind:image_ind+1, dataset_ind_first_label_ind:dataset_ind_last_label_ind]
                target[i][image_ind:image_ind+1] = torch.clamp(target[i][image_ind:image_ind+1] - dataset_ind_first_label_ind, 0,dataset_ind_last_label_ind)

        loss = sum([weights[i] * self.loss(*inputs) for i, inputs in enumerate(zip(*args)) if weights[i] != 0.0])
        return loss

class mcUNetTrainer(nnUNetTrainer):
    def _build_loss(self):
        assert not self.label_manager.has_regions, "regions not supported by this trainer"

        loss = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice,
                               'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                              ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)


        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                # very strange and stupid interaction. DDP crashes and complains about unused parameters due to
                # weights[-1] = 0. Interestingly this crash doesn't happen with torch.compile enabled. Strange stuff.
                # Anywho, the simple fix is to set a very low weight to this.
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = MultiClassDeepSupervisionWrapper(loss, weights)

        return loss


