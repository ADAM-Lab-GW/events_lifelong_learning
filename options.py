"""
The code in this file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/blob/master/options.py

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).
"""

import argparse

from utils import checkattr


##-------------------------------------------------------------------------------------------------------------------##

####################
## Define options ##
####################

def define_args(filename, description):
    parser = argparse.ArgumentParser('./{}.py'.format(filename), description=description)
    return parser


def add_general_options(parser):
    parser.add_argument('--no-save', action='store_false', dest='save', help="don't save trained models")
    parser.add_argument('--full-stag', type=str, metavar='STAG', default='none', help="tag for saving full model")
    parser.add_argument('--full-ltag', type=str, metavar='LTAG', default='none', help="tag for loading full model")
    parser.add_argument('--test', action='store_false', dest='train', help='evaluate previously saved model')
    parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
    parser.add_argument('--seed', type=int, default=0, help='[first] random seed (for each random-module used)')
    parser.add_argument('--n-seeds', type=int, default=1, help='how often to repeat?')
    parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
    parser.add_argument('--data-dir', type=str, default='./store/datasets', dest='d_dir', help="default: %(default)s")
    parser.add_argument('--model-dir', type=str, default='./store/models', dest='m_dir', help="default: %(default)s")
    parser.add_argument('--plot-dir', type=str, default='./store/plots', dest='p_dir', help="default: %(default)s")
    parser.add_argument('--results-dir', type=str, default='./store/results', dest='r_dir',
                        help="default: %(default)s")
    parser.add_argument('--settings_file', help='Path to settings yaml', required=False, default='./settings.yaml')

    return parser


def add_eval_options(parser):
    # evaluation parameters
    eval = parser.add_argument_group('Evaluation Parameters')
    eval.add_argument('--pdf', action='store_true', help="generate pdf with plots for individual experiment(s)")
    eval.add_argument('--plot_tsne', action='store_true', help="generate pdf with tsne plots for a generator")
    eval.add_argument('--prec-n', type=int, default=1024, help="# samples for evaluating accuracy (visdom-plots)")
    eval.add_argument('--sample-n', type=int, default=64, help="# images to show")
    eval.add_argument('--no-samples', action='store_true', help="don't plot generated/reconstructed images")
    eval.add_argument('--eval-tag', type=str, metavar="ETAG", default="none", help="tag for evaluation model")

    return parser


def add_task_options(parser):
    task_params = parser.add_argument_group('Task Parameters')
    tasks = ['NCALTECH12', 'NCALTECH256', 'NCALTECH101', 'NMNIST', 'eventSym']
    task_default = 'eventSym'
    task_params.add_argument('--experiment', type=str, default=task_default, choices=tasks)
    task_params.add_argument('--tasks', type=int, help='number of tasks')

    return parser


def add_model_options(parser):
    # model architecture parameters
    model = parser.add_argument_group('Parameters Main Model')
    # -fully-connected-layers
    model.add_argument('--fc-layers', type=int, default=3, dest='fc_lay', help="# of fully-connected layers")
    model.add_argument('--fc-units', type=int, default=None, metavar="N", help="# of units in first fc-layers")
    model.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
    model.add_argument('--fc-bn', type=str, default="no", help="use batch-norm in the fc-layers (no|yes)")
    model.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu", "none"])
    model.add_argument('--h-dim', type=int, metavar="N", help='# of hidden units final layer (default: fc-units)')
    # NOTE: number of units per fc-layer linearly declinces from [fc_units] to [h_dim].
    model.add_argument('--z-dim', type=int, default=100, help='size of latent representation (if feedback, def=100)')

    return parser


def add_train_options(parser):
    # training hyperparameters / initialization
    train_params = parser.add_argument_group('Training Parameters')
    train_params.add_argument('--iters', type=int, help="# batches to optimize main model")
    train_params.add_argument('--lr', type=float, default=None, help="learning rate")
    train_params.add_argument('--batch', type=int, default=None, help="batch-size")
    train_params.add_argument('--init-weight', type=str, default='standard', choices=['standard', 'xavier'])
    train_params.add_argument('--init-bias', type=str, default='standard', choices=['standard', 'constant'])
    train_params.add_argument('--reinit', action='store_true', help='reinitialize networks before each new task')
    train_params.add_argument('--recon-loss', type=str, choices=['MSE', 'BCE'])
    return parser


def add_VAE_options(parser):
    VAE = parser.add_argument_group('VAE-specific Parameters')
    # -how to weigh components of the loss-function?
    VAE.add_argument('--recon-weight', type=float, default=1., dest='rcl', help="weight of recon-loss (def=1)")
    VAE.add_argument('--variat-weight', type=float, default=1., dest='vl', help="weight of KLD-loss (def=1)")
    return parser


def add_replay_options(parser):
    replay = parser.add_argument_group('Replay Parameters')
    replay.add_argument('--distill', action='store_true', help="use distillation for replay")
    replay.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    # - generative model parameters (only if separate generator)
    replay.add_argument('--g-fc-lay', type=int, help='[fc_layers] in generator (default: same as classifier)')
    replay.add_argument('--g-fc-uni', type=int, help='[fc_units] in generator (default: same as classifier)')
    replay.add_argument('--g-h-dim', type=int, help='[h_dim] in generator (default: same as classifier)')
    replay.add_argument('--g-z-dim', type=int, default=100, help="size of generator's latent representation (def=100)")
    # - hyper-parameters (again only if separate generator)
    replay.add_argument('--gen-iters', type=int, dest="g_iters", help="# batches to optimize generator (def=[iters])")
    replay.add_argument('--lr-gen', type=float, help="learning rate (separate) generator (default: lr)")

    return parser


def add_bir_options(parser):
    BIR = parser.add_argument_group('Brain-inspired Replay Parameters')
    BIR.add_argument('--pred-weight', type=float, default=1., dest='pl', help="(FB) weight of prediction loss (def=1)")
    # -where on the VAE should the softmax-layer be placed?
    BIR.add_argument('--classify', type=str, default="beforeZ", choices=['beforeZ', 'fromZ'])
    BIR.add_argument('--n-modes', type=int, default=1, help="how many modes for prior (per class)? (def=1)")
    BIR.add_argument('--dg-type', type=str, metavar="TYPE", help="decoder-gates: based on tasks or classes?")
    BIR.add_argument('--dg-prop', type=float, help="decoder-gates: masking-prop")
    BIR.add_argument('--dg-si-prop', type=float, metavar="PROP", help="decoder-gates: masking-prop for BI-R + SI")
    BIR.add_argument('--dg-c', type=float, metavar="C", help="SI hyperparameter for BI-R + SI")

    return parser


def add_allocation_options(parser):
    cl = parser.add_argument_group('Memory Allocation Parameters')
    cl.add_argument('--c', type=float, dest="si_c", help="-->  SI: regularisation strength")
    cl.add_argument('--habituation', type=bool, default=False, dest="habituation", help="-->  Habituation")
    cl.add_argument('--habituation_decay_rate', type=int, default=0, dest="habituation_decay_rate",
                    help="-->  Habituation_decay_rate")
    cl.add_argument('--slowness', type=bool, default=False, dest="slowness", help="-->  Slowness term")
    cl.add_argument('--epsilon', type=float, default=0.1, dest="epsilon", help="-->  SI: dampening parameter")
    cl.add_argument('--xdg-prop', type=float, dest='xdg_prop', help="--> XdG: prop neurons per layer to gate")

    return parser


##-------------------------------------------------------------------------------------------------------------------##

############################
## Check / modify options ##
############################

def set_defaults(args):
    # -if 'brain-inspired' is selected, select corresponding defaults
    args.recon_loss = "MSE"
    args.dg_type = ("task" if args.experiment == 'permMNIST' else "class") if args.dg_type is None else args.dg_type
    args.tasks = (6 if args.experiment == 'NCALTECH12' else 64) if args.tasks is None else args.tasks
    args.iters = (100 if args.experiment == 'NCALTECH12' else 100) if args.iters is None else args.iters
    args.lr = 0.0001 if args.lr is None else args.lr
    args.batch = (64 if args.experiment == 'NCALTECH12' else 256) if args.batch is None else args.batch
    args.fc_units = 2000 if args.fc_units is None else args.fc_units

    args.si_c = 1. if args.si_c is None else args.si_c
    args.dg_prop = 0.7 if args.dg_prop is None else args.dg_prop
    args.dg_si_prop = 0.6 if args.dg_si_prop is None else args.dg_si_prop
    args.dg_c = 100000. if args.dg_c is None else args.dg_c
    # -for other unselected options, set default values (not specific to chosen experiment)
    args.h_dim = args.fc_units if args.h_dim is None else args.h_dim
    args.lr_gen = args.lr if args.lr_gen is None else args.lr_gen
    args.g_iters = args.iters if args.g_iters is None else args.g_iters
    args.g_fc_lay = args.fc_lay if args.g_fc_lay is None else args.g_fc_lay
    args.g_fc_uni = args.fc_units if args.g_fc_uni is None else args.g_fc_uni
    args.g_h_dim = args.g_fc_uni if args.g_h_dim is None else args.g_h_dim
    # -if [log_per_task] (which is default for comparison-scripts), reset all logs
    args.log_per_task = True
    args.prec_log = args.iters
    args.loss_log = args.iters
    args.sample_log = args.iters

    return args


def check_for_errors(args):
    if checkattr(args, 'xdg') and args.xdg_prop > 0:
        raise ValueError("'XdG' does not make sense")
    # -if XdG is selected together with replay of any kind, give error
    if checkattr(args, 'xdg') and args.xdg_prop > 0 and (not args.replay == "none"):
        raise NotImplementedError("XdG is not supported with '{}' replay.".format(args.replay))
        # --> problem is that applying different task-masks interferes with gradient calculation
        #    (should be possible to overcome by calculating each gradient before applying next mask)
    # -if 'only_last' is selected with replay, EWC or SI, give error
    if checkattr(args, 'only_last') and (not args.replay == "none"):
        raise NotImplementedError("Option 'only_last' is not supported with '{}' replay.".format(args.replay))
    if checkattr(args, 'only_last') and (checkattr(args, 'si') and args.si_c > 0):
        raise NotImplementedError("Option 'only_last' is not supported with SI.")
    # -error in type of reconstruction loss
    if checkattr(args, "normalize") and hasattr(args, "recon_los") and args.recon_loss == "BCE":
        raise ValueError("'BCE' is not a valid reconstruction loss with normalized images")
