"""
This file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/tree/master/models

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).
"""

import numpy as np
import torch
from torch.nn import functional as F

from models.cl.continual_learner import ContinualLearner
from models.fc.layers import fc_layer
from models.fc.nets import MLP
from models.utils import loss_functions as lf, modules


class Classifier(ContinualLearner):
    '''Model for encoding (i.e., feature extraction) and classifying images, enriched as "ContinualLearner"--object.'''


    def __init__(self, classes, classes_main=None,
             fc_layers=3, fc_units=1000, h_dim=400, sub_to_main=None,lambda_main = 0.5,fc_drop=0, fc_bn=True, fc_nl="relu", fc_gated=False,
                 bias=True, excitability=False, excit_buffer=False,
                 # -training-specific settings (can be changed after setting up model)
                 hidden=False):

        # model configurations
        super().__init__()


        self.classes_sub = classes
        self.classes_main = classes_main
        self.lambda_main = lambda_main   # weight for main-class loss

        if sub_to_main is not None:
            self.register_buffer("sub_to_main", torch.LongTensor(sub_to_main))
        else:
            self.sub_to_main = None

        # self.classes = classes
        self.label = "Classifier"
        self.fc_layers = fc_layers
        self.fc_drop = fc_drop

        # settings for training
        self.hidden = hidden
        # --> if True, [self.classify] & replayed data of [self.train_a_batch] expected to be "hidden data"

        # optimizer (needs to be set before training starts))
        self.optimizer = None
        self.optim_list = []

        # check whether there is at least 1 fc-layer
        if fc_layers < 1:
            raise ValueError("The classifier needs to have at least 1 fully-connected layer.")

        ######------SPECIFY MODEL------######
        self.flatten = modules.Flatten()
        # ------------------------------calculate input/output-sizes--------------------------------#
        self.conv_out_units = 1536  # TODO
        self.conv_out_size = 1536  # TODO
        if fc_layers < 2:
            self.fc_layer_sizes = [self.conv_out_units]  # --> this results in self.fcE = modules.Identity()
        elif fc_layers == 2:
            self.fc_layer_sizes = [self.conv_out_units, h_dim]
        else:
            self.fc_layer_sizes = [self.conv_out_units] + [int(x) for x in
                                                           np.linspace(fc_units, h_dim, num=fc_layers - 1)]
        self.units_before_classifier = h_dim if fc_layers > 1 else self.conv_out_units
        # ------------------------------------------------------------------------------------------#
        # --> fully connected layers
        self.fcE = MLP(size_per_layer=self.fc_layer_sizes, drop=fc_drop, batch_norm=fc_bn, nl=fc_nl, bias=bias,
                       excitability=excitability, excit_buffer=excit_buffer,
                       gated=fc_gated)  # , output="none") ## NOTE: temporary change!!!
        # --> classifier
        self.classifier_sub = fc_layer(
            self.units_before_classifier, self.classes_sub,
            excit_buffer=True, nl='none', drop=fc_drop
        )

        if self.classes_main is not None:
            self.classifier_main = fc_layer(
                self.units_before_classifier, self.classes_main,
                excit_buffer=True, nl='none', drop=fc_drop
            )

        # backward compatibility
        self.classifier = self.classifier_sub


    def list_init_layers(self):
        '''Return list of modules whose parameters could be initialized differently (i.e., conv- or fc-layers).'''
        list = self.fcE.list_init_layers()
        list += self.classifier.list_init_layers()
        return list

    @property
    def name(self):
        return "{}_c{}".format(self.fcE.name, self.classes)

    def forward(self, x):
        final_features = self.fcE(self.flatten(x))
        return self.classifier_sub(final_features)


    def input_to_hidden(self, x):
        '''Get [hidden_rep]s (inputs to final fully-connected layers) for images [x].'''
        return x

    def hidden_to_output(self, hidden_rep):
        '''Map [hidden_rep]s to outputs (i.e., logits for all possible output-classes).'''
        return self.classifier(self.fcE(self.flatten(hidden_rep)))

    def feature_extractor(self, images, from_hidden=False):
        return self.fcE(self.flatten(images))

    def classify(self, x, return_both=False):

        '''For input [x] (image or extracted "intermediate" image features), return all predicted "scores"/"logits".'''

        image_features = self.flatten(x)
        hE = self.fcE(image_features)

        logits_sub = self.classifier_sub(hE)
        if (not return_both) or (self.classes_main is None):
            return logits_sub
        logits_main = self.classifier_main(hE)
        return logits_sub, logits_main



    def train_a_batch(self, x, y=None, y_main=None, x_=None, y_=None, scores_=None, rnt=0.5, active_classes=None,
                      task=1, replay_not_hidden=False, **kwargs):
        '''Train model for one batch ([x],[y]), possibly supplemented with replayed data ([x_],[y_]).

        [x]                 <tensor> batch of inputs (could be None, in which case only 'replayed' data is used)
        [y]                 <tensor> batch of corresponding labels
        [x_]                None or (<list> of) <tensor> batch of replayed inputs
                              NOTE: expected to be as [self.hidden] or [replay_up_to], unless [replay_not_hidden]==True
        [y_]                None or (<list> of) <tensor> batch of corresponding "replayed" labels
        [scores_]           None or (<list> of) <tensor> 2Dtensor:[batch]x[classes] predicted "scores"/"logits" for [x_]
        [rnt]               <number> in [0,1], relative importance of new task
        [active_classes]    None or (<list> of) <list> with "active" classes
        [task]              <int>, for setting task-specific mask
        [replay_not_hidden] <bool> provided [x_] are original images, even though other level might be expected'''

        # Set model to training-mode
        self.train()

        # Reset optimizer
        self.optimizer.zero_grad()

        ##--(1)-- CURRENT DATA --##

        if x is not None:
            # If requested, apply correct task-specific mask
            if self.mask_dict is not None:
                self.apply_XdGmask(task=task)

            # Run model
            y_hat_sub, y_hat_main = self.classify(x, return_both=True)

            predL_sub = F.cross_entropy(y_hat_sub, y)
            loss_cur = predL_sub

            predL_main = None
            if y_main is not None and self.classes_main is not None:
                predL_main = F.cross_entropy(y_hat_main, y_main)
                loss_cur = loss_cur + self.lambda_main * predL_main

            # y_hat = self(x)
            # if active_classes is not None:
            #     class_entries = active_classes[-1] if type(active_classes[0]) == list else active_classes
            #     y_hat = y_hat[:, class_entries]

            # # Calculate multiclass prediction loss
            # if y is not None and len(y.size()) == 0:
            #     y = y.expand(1)  # --> hack to make it work if batch-size is 1
            # predL = None if y is None else F.cross_entropy(input=y_hat, target=y, reduction='none')
            # # --> no reduction needed, summing over classes is "implicit"
            # predL = None if y is None else lf.weighted_average(predL, weights=None, dim=0)  # -> average over batch

            # # Weigh losses
            # loss_cur = predL

            # Calculate training-precision
            precision = (y == y_hat_sub.argmax(1)).sum().item() / x.size(0)

            precision_main = None
            if y_main is not None and self.classes_main is not None:
                precision_main = (y_main == y_hat_main.argmax(1)).sum().item() / x.size(0)

            # precision = None if y is None else (y == y_hat.max(1)[1]).sum().item() / x.size(0)

            # If XdG is combined with replay, backward-pass needs to be done before new task-mask is applied
            if (self.mask_dict is not None) and (x_ is not None):
                weighted_current_loss = rnt * loss_cur
                weighted_current_loss.backward()
        else:
            precision = predL = None

        ##--(2)-- REPLAYED DATA --##

        if x_ is not None:
            TaskIL = (type(y_) == list) if (y_ is not None) else (type(scores_) == list)
            if not TaskIL:
                y_ = [y_]
                scores_ = [scores_]
                active_classes = [active_classes] if (active_classes is not None) else None
            n_replays = len(y_) if (y_ is not None) else len(scores_)

            # Prepare lists to store losses for each replay
            loss_replay = [torch.tensor(0., device=self._device())] * n_replays
            predL_r = [torch.tensor(0., device=self._device())] * n_replays
            distilL_r = [torch.tensor(0., device=self._device())] * n_replays

            # Run model (if [x_] is not a list with separate replay per task and there is no task-specific mask)
            if (not type(x_) == list) and (self.mask_dict is None):
                y_hat_sub, y_hat_main = self.classify(x_, return_both=True)
                # y_hat_all = self.classify(x_, not_hidden=replay_not_hidden)

            # Loop to perform each replay
            for replay_id in range(n_replays):

                # -if [x_] is a list with separate replay per task, evaluate model on this task's replay
                if (type(x_) == list) or (self.mask_dict is not None):
                    x_temp_ = x_[replay_id] if type(x_) == list else x_
                    if self.mask_dict is not None:
                        self.apply_XdGmask(task=replay_id + 1)
                    y_hat_sub, y_hat_main = self.classify(x_temp_, return_both=True)

                # y_hat = y_hat_all if (active_classes is None) else y_hat_all[:, active_classes[replay_id]]
                y_hat_sub = y_hat_sub if (active_classes is None) else y_hat_sub[:, active_classes[replay_id]]
                y_hat_main = y_hat_main  # do NOT slice by active_classes (those are subclass IDs)
                
                                # y_sub for replay can be local indices if we sliced by active_classes
                if (y_ is not None) and (y_[replay_id] is not None) and (self.sub_to_main is not None):
                    y_sub_r = y_[replay_id]  # these labels correspond to y_hat_sub's class indexing

                    if active_classes is None:
                        # labels already global subclass IDs
                        y_sub_global = y_sub_r
                    else:
                        # labels are indices into active_classes[replay_id] → map to global subclass IDs
                        ac = torch.tensor(active_classes[replay_id], device=self._device(), dtype=torch.long)
                        y_sub_global = ac[y_sub_r]

                    # map subclass → main
                    y_main_r = self.sub_to_main[y_sub_global]
                else:
                    y_main_r = None


                # Calculate losses
                if (y_ is not None) and (y_[replay_id] is not None):

                    predL_r[replay_id] = F.cross_entropy(y_hat_sub, y_[replay_id], reduction='none')
                    predL_r[replay_id] = lf.weighted_average(predL_r[replay_id], dim=0)

                # main CE (NEW, only if we can map)
                predL_main_r_this = torch.tensor(0., device=self._device())
                if y_main_r is not None and (self.classes_main is not None):
                    predL_main_r_this = F.cross_entropy(y_hat_main, y_main_r, reduction='none')
                    predL_main_r_this = lf.weighted_average(predL_main_r_this, dim=0)

                    
                    # -> average over batch
                if (scores_ is not None) and (scores_[replay_id] is not None):
                    # n_classes_to_consider = scores.size(1) #--> with this version, no zeroes are added to [scores]!
                    n_classes_to_consider = y_hat.size(1)  # --> zeros will be added to [scores] to make it this size!
                    distilL_r[replay_id] = lf.loss_fn_kd(
                        scores=y_hat[:, :n_classes_to_consider], target_scores=scores_[replay_id], T=self.KD_temp,
                    )  # --> summing over classes & averaging over batch within this function
                # Weigh losses
                if self.replay_targets == "hard":
                    loss_replay[replay_id] = predL_r[replay_id] + self.lambda_main * predL_main_r_this
                elif self.replay_targets == "soft":
                    loss_replay[replay_id] = distilL_r[replay_id]


                # If task-specific mask, backward pass needs to be performed before next task-mask is applied
                if self.mask_dict is not None:
                    weighted_replay_loss_this_task = (1 - rnt) * loss_replay[replay_id] / n_replays
                    weighted_replay_loss_this_task.backward()

        # Calculate total loss
        loss_replay = None if (x_ is None) else sum(loss_replay) / n_replays
        loss_total = loss_replay if (x is None) else (
            loss_cur if x_ is None else rnt * loss_cur + (1 - rnt) * loss_replay)

        ##--(3)-- ALLOCATION LOSSES --##

        # Add SI-loss (Zenke et al., 2017)
        surrogate_loss = self.surrogate_loss()
        if self.si_c > 0:
            loss_total += self.si_c * surrogate_loss

        # Backpropagate errors (if not yet done)
        if (self.mask_dict is None) or (x_ is None):
            loss_total.backward()
        # Take optimization-step
        self.optimizer.step()

        # Return the dictionary with different training-loss split in categories
        return {
            'loss_total': loss_total.item(),
            'loss_current': loss_cur.item() if x is not None else 0,
            'loss_replay': loss_replay.item() if (loss_replay is not None) and (x is not None) else 0,
            'pred': predL.item() if predL is not None else 0,
            'pred_r': sum(predL_r).item() / n_replays if (x_ is not None and predL_r[0] is not None) else 0,
            'distil_r': sum(distilL_r).item() / n_replays if (x_ is not None and distilL_r[0] is not None) else 0,
            'precision': precision if precision is not None else 0.,
        }
