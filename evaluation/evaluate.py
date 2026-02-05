"""
This file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/tree/master/eval

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).

"""

import torch
from sklearn import manifold

import utils
import visual.plt

def _unwrap_dataset(ds):
    # SubDataset has attribute `.dataset`
    while hasattr(ds, "dataset"):
        ds = ds.dataset
    return ds

def _get_sub_to_main(ds):
    base = _unwrap_dataset(ds)
    if hasattr(base, "sub_to_main"):
        return base.sub_to_main  # dict: sub_id -> main_id
    return None

####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####

def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
             no_task_mask=False, task=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Get device-type / using cuda?
    device = model._device()
    cuda = model._is_on_cuda()

    # Set model to eval()-mode
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda)
    total_tested = total_correct = 0
    # for data, labels in data_loader:
        # # -break on [test_size] (if "None", full dataset is used)
        # if test_size:
        #     if total_tested >= test_size:
        #         break
        # # -evaluate model (if requested, only on [allowed_classes])
        # data, labels = data.to(device), labels.to(device)
        # labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        # with torch.no_grad():
        #     scores = model.classify(data, not_hidden=True)
        #     scores = scores if (allowed_classes is None) else scores[:, allowed_classes]
        #     _, predicted = torch.max(scores, 1)
        # # -update statistics
        # total_correct += (predicted == labels).sum().item()
        # total_tested += len(data)
    for batch in data_loader:

    # --- unpack batch (flat vs hierarchical)
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            data, y_sub, y_main = batch
            labels = y_sub                 # TEMP: evaluate on subclass labels
            # y_main kept for later hierarchical metrics
        else:
            data, labels = batch

        # --- break on test_size (if None, full dataset is used)
        if test_size is not None and total_tested >= test_size:
            break

        # --- move to device
        data = data.to(device)
        labels = labels.to(device)

        # --- adjust labels to active range if evaluating only a slice of classes
        labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels

        # --- forward + restrict to allowed classes
        with torch.no_grad():
            scores = model.classify(data, not_hidden=True)
            scores = scores if (allowed_classes is None) else scores[:, allowed_classes]
            _, predicted = torch.max(scores, 1)

        # --- update stats
        total_correct += (predicted == labels).sum().item()
        total_tested += labels.size(0)   # <-- critical (don’t use len(data) if data isn't first dim guaranteed)


    precision = total_correct / total_tested

    # Print result on screen (if requested) and return it
    if verbose:
        print('=> precision: {:.3f}'.format(precision))
    return precision
def validate_hier(model, dataset, batch_size=128, test_size=1024, verbose=False,
                  allowed_classes=None, no_task_mask=False, task=None):
    """
    Hierarchical evaluation:
    - acc_sub  : subclass accuracy
    - acc_main : main-class accuracy (derived from predicted subclass)
    """

    device = model._device()
    cuda = model._is_on_cuda()
    model.eval()

    # task-gating (same behavior as validate)
    if model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    sub_to_main = _get_sub_to_main(dataset)
    if sub_to_main is None:
        raise ValueError("Dataset does not expose sub_to_main mapping.")

    data_loader = utils.get_data_loader(dataset, batch_size, cuda=cuda, drop_last=False)

    total = 0
    correct_sub = 0
    correct_main = 0

    for batch in data_loader:
        if test_size and total >= test_size:
            break

        # hierarchical batch
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            data, y_sub, y_main = batch
        else:
            raise ValueError("validate_hier expects (x, y_sub, y_main).")

        data = data.to(device)
        y_sub = y_sub.to(device)
        y_main = y_main.to(device)

        # align subclass labels with allowed_classes (same logic as validate)
        y_sub_eval = y_sub - allowed_classes[0] if allowed_classes is not None else y_sub

        with torch.no_grad():
            scores = model.classify(data, not_hidden=True)
            scores = scores if allowed_classes is None else scores[:, allowed_classes]
            pred_sub = torch.argmax(scores, dim=1)

        # subclass accuracy
        correct_sub += (pred_sub == y_sub_eval).sum().item()

        # map predicted subclass → global subclass id
        if allowed_classes is not None:
            pred_sub_global = pred_sub + allowed_classes[0]
        else:
            pred_sub_global = pred_sub

        # map subclass → main
        pred_main = torch.tensor(
            [sub_to_main[int(s)] for s in pred_sub_global.cpu()],
            device=device
        )

        correct_main += (pred_main == y_main).sum().item()
        total += y_sub.size(0)

    if total == 0:
        return {"acc_sub": 0.0, "acc_main": 0.0, "n": 0}

    return {
        "acc_sub": correct_sub / total,
        "acc_main": correct_main / total,
        "n": total
    }



def initiate_precision_dict(n_tasks):
    '''Initiate <dict> with all precision-measures to keep track of.'''
    precision = {}
    precision["all_tasks"] = [[] for _ in range(n_tasks)]
    precision["average"] = []
    precision["x_iteration"] = []
    precision["x_task"] = []
    return precision


def precision(model, datasets, current_task, iteration, classes_per_task=None,
              precision_dict=None, test_size=None, visdom=None, verbose=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [precision_dict]    None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task
    [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

    # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
    n_tasks = len(datasets)
    precs = []
    main_precs = []

    for i in range(n_tasks):
        if i + 1 <= current_task:
            allowed_classes = list(range(classes_per_task * current_task))

            # --- subclass accuracy (original behavior)
            acc_sub = validate(
                model, datasets[i], test_size=test_size, verbose=verbose,
                allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1
            )
            precs.append(acc_sub)

            # --- hierarchical main-class accuracy (new)
            if _get_sub_to_main(datasets[i]) is not None:
                hier = validate_hier(
                    model, datasets[i], test_size=test_size,
                    allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1
                )
                main_precs.append(hier["acc_main"])
            else:
                main_precs.append(0)

        else:
            precs.append(0)
            main_precs.append(0)

    average_precs = sum(
        [precs[task_id] if task_id == 0 else precs[task_id] for task_id in range(current_task)]
    ) / (current_task)

    average_main_precs = sum(
    [main_precs[task_id] for task_id in range(current_task)]) / (current_task)

    # Print results on screen
    if verbose:
        print(' => ave subclass precision: {:.3f}'.format(average_precs))
        print(' => ave main precision: {:.3f}'.format(average_main_precs))
    # Send results to visdom server
    names = ['task {}'.format(i + 1) for i in range(n_tasks)]
    if visdom is not None:
        visual.visdom.visualize_scalars(
            scalars=precs, names=names, iteration=iteration,
            title="Accuracy per task ({})".format(visdom["graph"]), env=visdom["env"], ylabel="precision"
        )
        if n_tasks > 1:
            visual.visdom.visualize_scalars(
                scalars=[average_precs], names=["ave precision"], iteration=iteration,
                title="Average accuracy ({})".format(visdom["graph"]), env=visdom["env"], ylabel="precision"
            )

    # Append results to [progress]-dictionary and return
    if precision_dict is not None:
        # subclass accuracy (existing)
        for task_id, _ in enumerate(names):
            precision_dict["all_tasks"][task_id].append(precs[task_id])
        precision_dict["average"].append(average_precs)

        # --- hierarchical main accuracy (new, non-breaking)
        precision_dict.setdefault("all_tasks_main", [[] for _ in range(n_tasks)])
        precision_dict.setdefault("average_main", [])

        for task_id in range(n_tasks):
            precision_dict["all_tasks_main"][task_id].append(main_precs[task_id])
        precision_dict["average_main"].append(average_main_precs)

        precision_dict["x_iteration"].append(iteration)
        precision_dict["x_task"].append(current_task)

    return precision_dict


####--------------------------------------------------------------------------------------------------------------####

####------------------------------------------####
####----VISUALIZE EXTRACTED REPRESENTATION----####
####------------------------------------------####

def visualize_latent_space(model, X, y=None, visdom=None, verbose=False, file_path=None):
    '''Show T-sne projection of feature representation used to classify from (with each class in different color).'''

    # Set model to eval()-mode
    model.eval()

    # Compute the representation used for classification
    if verbose:
        print("Computing feature space...")
    with torch.no_grad():
        z_mean = model.feature_extractor(X)

    # Compute t-SNE embedding of latent space (unless z has 2 dimensions!)
    if z_mean.size()[1] == 2:
        z_tsne = z_mean.cpu().numpy()
    else:
        if verbose:
            print("Computing t-SNE embedding...")
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        z_tsne = tsne.fit_transform(z_mean.cpu())
    # Plot images according to t-sne embedding
    visual.plt.plot_scatter(z_tsne[:, 0], z_tsne[:, 1], colors=y.cpu().numpy(), file_path=file_path)
    if visdom is not None:
        message = ("Visualization of extracted representation")
        visual.visdom.scatter_plot(z_tsne, title="{} ({})".format(message, visdom["graph"]),
                                   colors=y + 1 if y is not None else y, env=visdom["env"])
