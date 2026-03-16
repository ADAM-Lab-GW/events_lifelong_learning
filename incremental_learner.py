#!/usr/bin/env python3

"""
The code in this file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/blob/master/main_cl.py

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).
"""

import os

import numpy as np
import torch
from torch import optim

import define_models as define
# -custom-written libraries
import options
import utils
from data_provider.data_loader import get_data_incremental_strategy
from evaluation import callbacks as cb
from evaluation import evaluate
from models.cl.continual_learner import ContinualLearner
from param_stamp import get_param_stamp
from train import train_cl


## Function for specifying input-options and organizing / checking them
def handle_inputs():
    # Define input options
    parser = options.define_args(filename="main_cl", description='Compare & combine continual learning approaches.')
    parser = options.add_general_options(parser)
    parser = options.add_eval_options(parser)
    parser = options.add_task_options(parser)
    parser = options.add_model_options(parser)
    parser = options.add_train_options(parser)
    parser = options.add_replay_options(parser)
    parser = options.add_bir_options(parser)
    parser = options.add_allocation_options(parser)
    # Parse, process (i.e., set defaults for unselected options) and check chosen options
    args = parser.parse_args()
    options.set_defaults(args)
    options.check_for_errors(args)
    return args


## Function for running one continual learning experiment
def run(args, verbose=False):
    # Create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
    if args.pdf and not os.path.isdir(args.p_dir):
        os.mkdir(args.p_dir)

    # If only want param-stamp, get it and exit
    if args.get_stamp:
        from param_stamp import get_param_stamp_from_args
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")

    # Report whether cuda is used
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    # -------------------------------------------------------------------------------------------------#

    # ----------------#
    # ----- DATA -----#
    # ----------------#

    # Prepare data for chosen experiment
    if verbose:
        print("\nPreparing the data...")
    (train_datasets, test_datasets), config, classes_per_task = get_data_incremental_strategy(
        name=args.dataset, tasks=args.tasks, data_dir=args.d_dir,
        verbose=verbose, only_test=(not args.train)
    )

    # -------------------------------------------------------------------------------------------------#

    # ----------------------#
    # ----- MAIN MODEL -----#
    # ----------------------#

    # Define main model (i.e., classifier, if requested with feedback connections)
    if utils.checkattr(args, 'feedback'):
        model = define.define_autoencoder(args=args, config=config, device=device)
    else:
        model = define.define_classifier(args=args, config=config, device=device)

    # - initialize (pre-trained) parameters
    model = define.init_params(model, args)

    # Define optimizer (only optimize parameters that "requires_grad")
    model.optim_list = [
        {'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr},
    ]
    model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

    # -------------------------------------------------------------------------------------------------#

    # ----------------------------------------------------#
    # ----- CL-STRATEGY: REGULARIZATION / ALLOCATION -----#
    # ----------------------------------------------------#

    # Synpatic Intelligence (SI)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'si'):
        model.si_c = args.si_c if args.si else 0
        model.epsilon = args.epsilon

    # Habituation (H)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'habituation'):
        model.habituation = args.habituation

    # Slowness (ST)
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'slowness'):
        model.slowness = args.slowness

    # XdG: create for every task a "mask" for each hidden fully connected layer
    if isinstance(model, ContinualLearner) and utils.checkattr(args, 'xdg') and args.xdg_prop > 0:
        model.define_XdGmask(gating_prop=args.xdg_prop, n_tasks=args.tasks)

    # -------------------------------------------------------------------------------------------------#

    # -------------------------------#
    # ----- CL-STRATEGY: REPLAY -----#
    # -------------------------------#

    # Use distillation loss (i.e., soft targets) for replayed data? (and set temperature)
    if isinstance(model, ContinualLearner) and hasattr(args, 'replay') and not args.replay == "none":
        model.replay_targets = "soft" if args.distill else "hard"
        model.KD_temp = args.temp

    # If needed, specify separate model for the generator
    train_gen = (hasattr(args, 'replay') and args.replay == "generative" and not utils.checkattr(args, 'feedback'))
    if train_gen:
        # Specify architecture
        generator = define.define_autoencoder(args, config, device, generator=True)

        # Initialize parameters
        generator = define.init_params(generator, args)

        # Set optimizer(s)
        generator.optim_list = [
            {'params': filter(lambda p: p.requires_grad, generator.parameters()),
             'lr': args.lr_gen if hasattr(args, 'lr_gen') else args.lr},
        ]
        generator.optimizer = optim.Adam(generator.optim_list, betas=(0.9, 0.999))
    else:
        generator = None

    # -------------------------------------------------------------------------------------------------#

    # ---------------------#
    # ----- REPORTING -----#
    # ---------------------#

    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp = get_param_stamp(
        args, model.name, verbose=verbose,
        replay=True if (hasattr(args, 'replay') and not args.replay == "none") else False,
        replay_model_name=generator.name if (
                hasattr(args, 'replay') and args.replay in ("generative") and not utils.checkattr(args, 'feedback')
        ) else None,
    )

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        utils.print_model_info(model, title="MAIN MODEL")
        # -generator
        if generator is not None:
            utils.print_model_info(generator, title="GENERATOR")

    # Define [progress_dicts] to keep track of performance during training for storing and for later plotting in pdf
    precision_dict = evaluate.initiate_precision_dict(args.tasks)

    # Prepare for plotting in visdom
    visdom = None

    # -------------------------------------------------------------------------------------------------#

    # ---------------------#
    # ----- CALLBACKS -----#
    # ---------------------#

    g_iters = args.g_iters if hasattr(args, 'g_iters') else args.iters

    # Callbacks for reporting on and visualizing loss
    generator_loss_cbs = [
        cb._VAE_loss_cb(log=args.loss_log, visdom=visdom,
                        replay=(hasattr(args, "replay") and not args.replay == "none"),
                        model=model if utils.checkattr(args, 'feedback') else generator, tasks=args.tasks,
                        iters_per_task=args.iters if utils.checkattr(args, 'feedback') else g_iters)
    ] if (train_gen or utils.checkattr(args, 'feedback')) else [None]
    solver_loss_cbs = [
        cb._solver_loss_cb(log=args.loss_log, visdom=visdom, model=model, iters_per_task=args.iters, tasks=args.tasks,
                           replay=(hasattr(args, "replay") and not args.replay == "none"))
    ] if (not utils.checkattr(args, 'feedback')) else [None]

    sample_cbs = [None]

    # Callbacks for reporting and visualizing accuracy, and visualizing representation extracted by main model
    # -visdom (i.e., after each [prec_log]
    eval_cb = cb._eval_cb(
        log=args.prec_log, test_datasets=test_datasets, visdom=visdom, precision_dict=None, iters_per_task=args.iters,
        test_size=args.prec_n, classes_per_task=classes_per_task,
    )
    # -pdf / reporting: summary plots (i.e, only after each task)
    eval_cb_full = cb._eval_cb(
        log=args.iters, test_datasets=test_datasets, precision_dict=precision_dict,
        iters_per_task=args.iters, classes_per_task=classes_per_task,
    )
    # -visualize feature space
    latent_space_cb = cb._latent_space_cb(
        log=args.iters,
        datasets=test_datasets,
        iters_per_task=args.iters,
        sample_size=80,
        pdf=args.plot_tsne,
        file_path=args.p_dir,
    )
    # -collect them in <lists>
    eval_cbs = [eval_cb, eval_cb_full, latent_space_cb]

    # -------------------------------------------------------------------------------------------------#

    # --------------------#
    # ----- TRAINING -----#
    # --------------------#

    if args.train:
        if verbose:
            print("\nTraining...")
        # Train model
        train_cl(
            model, train_datasets, replay_mode=args.replay if hasattr(args, 'replay') else "none",
            classes_per_task=classes_per_task, iters=args.iters,
            batch_size=args.batch, batch_size_replay=args.batch_replay if hasattr(args, 'batch_replay') else None,
            generator=generator, gen_iters=g_iters, gen_loss_cbs=generator_loss_cbs,
            feedback=utils.checkattr(args, 'feedback'), sample_cbs=sample_cbs, eval_cbs=eval_cbs,
            loss_cbs=generator_loss_cbs if utils.checkattr(args, 'feedback') else solver_loss_cbs,
            args=args, reinit=utils.checkattr(args, 'reinit'), only_last=utils.checkattr(args, 'only_last')
        )
        # Save evaluation metrics measured throughout training
        file_name = "{}/dict-{}".format(args.r_dir, param_stamp)
        utils.save_object(precision_dict, file_name)
        # Save trained model(s), if requested
        if args.save:
            save_name = "mM-{}".format(param_stamp) if (
                    not hasattr(args, 'full_stag') or args.full_stag == "none"
            ) else "{}-{}".format(model.name, args.full_stag)
            utils.save_checkpoint(model, args.m_dir, name=save_name, verbose=verbose)
            if generator is not None:
                save_name = "gM-{}".format(param_stamp) if (
                        not hasattr(args, 'full_stag') or args.full_stag == "none"
                ) else "{}-{}".format(generator.name, args.full_stag)
                utils.save_checkpoint(generator, args.m_dir, name=save_name, verbose=verbose)

    else:
        # Load previously trained model(s) (if goal is to only evaluate previously trained model)
        if verbose:
            print("\nLoading parameters of the previously trained models...")
        load_name = "mM-{}".format(param_stamp) if (
                not hasattr(args, 'full_ltag') or args.full_ltag == "none"
        ) else "{}-{}".format(model.name, args.full_ltag)
        utils.load_checkpoint(model, args.m_dir, name=load_name, verbose=verbose,
                              add_si_buffers=(isinstance(model, ContinualLearner) and utils.checkattr(args, 'si')))
        if generator is not None:
            load_name = "gM-{}".format(param_stamp) if (
                    not hasattr(args, 'full_ltag') or args.full_ltag == "none"
            ) else "{}-{}".format(generator.name, args.full_ltag)
            utils.load_checkpoint(generator, args.m_dir, name=load_name, verbose=verbose)

    # -------------------------------------------------------------------------------------------------#

    # -----------------------------------#
    # ----- EVALUATION of CLASSIFIER-----#
    # -----------------------------------#

    if verbose:
        print("\n\nEVALUATION RESULTS:")

    # Evaluate precision of final model on full test-set
    precs = [evaluate.validate(
        model, test_datasets[i], verbose=False, test_size=None, task=i + 1,
        allowed_classes=None
    ) for i in range(args.tasks)]
    average_precs = sum(precs) / args.tasks
    # -print on screen
    if verbose:
        print("\n Accuracy of final model on test-set:")
        for i in range(args.tasks):
            print(" - {} {}: {:.4f}".format("For classes from task",
                                            i + 1, precs[i]))
        print('=> Average accuracy over all {} {}: {:.4f}\n'.format(
            args.tasks * classes_per_task,
            "classes", average_precs
        ))
    # -write out to text file
    output_file = open("{}/prec-{}.txt".format(args.r_dir, param_stamp), 'w')
    output_file.write('{}\n'.format(average_precs))
    output_file.close()


if __name__ == '__main__':
    args = handle_inputs()
    run(args, verbose=True)
