"""
The code in this file is copied from
https://github.com/GMvandeVen/brain-inspired-replay/blob/master/param_stamp.py

Adaptations of the code are performed by Vadym Gryshchuk (vadym.gryshchuk@protonmail.com).
"""

from data_provider.data_loader import get_data_incremental_strategy
from utils import checkattr


def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''
    from define_models import define_autoencoder, define_classifier

    # -get configurations of experiment
    config = get_data_incremental_strategy(
        name=args.dataset, tasks=args.tasks, data_dir=args.d_dir, only_config=True,
        verbose=False,
    )

    # -get model architectures
    model = define_autoencoder(args=args, config=config, device='cpu') if checkattr(
        args, 'feedback'
    ) else define_classifier(args=args, config=config, device='cpu')
    if checkattr(args, 'feedback'):
        model.lamda_pl = 1. if not hasattr(args, 'pl') else args.pl
    train_gen = (hasattr(args, 'replay') and args.replay == "generative" and not checkattr(args, 'feedback'))
    if train_gen:
        generator = define_autoencoder(args=args, config=config, device='cpu', generator=True)

    # -extract and return param-stamp
    model_name = model.name
    replay_model_name = generator.name if train_gen else None
    param_stamp = get_param_stamp(args, model_name, replay=(hasattr(args, "replay") and not args.replay == "none"),
                                  replay_model_name=replay_model_name, verbose=False)
    return param_stamp


def get_param_stamp(args, model_name, verbose=True, replay=False, replay_model_name=None):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    multi_n_stamp = "{n}-{of}".format(
        n=args.tasks, of="OL" if checkattr(args, 'only_last') else ""
    ) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{multi_n}".format(
        exp=args.dataset,
        multi_n=multi_n_stamp
    )
    if verbose:
        print(" --> task:          " + task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         " + model_stamp)

    # -for hyper-parameters
    hyper_stamp = "{i_e}{num}-lr{lr}{lrg}-b{bsz}{reinit}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        lrg=("" if args.lr == args.lr_gen else "-lrG{}".format(args.lr_gen)) if (
                hasattr(args, "lr_gen") and hasattr(args, "replay") and args.replay == "generative" and
                (not checkattr(args, "feedback"))
        ) else "",
        bsz=args.batch, reinit="-R" if checkattr(args, 'reinit') else ""
    )
    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)

    # for habituation
    h_stamp = ""
    if checkattr(args, 'habituation') and args.habituation:
        h_stamp = "Habituation{h}".format(h=args.habituation) if (
                checkattr(args, 'habituation') and args.habituation) else ""
        if verbose and checkattr(args, 'habituation') and args.habituation:
            print(" --> Habituation:            " + h_stamp)
    h_stamp = "--{}".format(h_stamp) if (
        (checkattr(args, 'habituation') and args.habituation)
    ) else ""

    # for decay rate (habituation)
    decay_stamp = ""
    if args.habituation_decay_rate:
        decay_stamp = "DecayRate{dc}".format(dc=args.habituation_decay_rate) if (
                args.habituation_decay_rate) else ""
        if verbose and args.habituation_decay_rate:
            print(" --> Decay rate:            " + decay_stamp)
    decay_stamp = "--{}".format(decay_stamp) if (
        (args.habituation_decay_rate)
    ) else ""

    # for top_neurons_stamp (habituation)
    top_neurons_stamp = ""
    if args.top_hab_neurons:
        top_neurons_stamp = "TopNeurons{tn}".format(tn=args.top_hab_neurons) if (
            args.top_hab_neurons) else ""
        if verbose and args.top_hab_neurons:
            print(" --> Top Neurons:            " + top_neurons_stamp)
    top_neurons_stamp = "--{}".format(top_neurons_stamp) if (
        (args.top_hab_neurons)
    ) else ""

    # for slowness
    st_stamp = ""
    if checkattr(args, 'slowness') and args.slowness:
        st_stamp = "Slowness{st}".format(st=args.slowness) if (
                checkattr(args, 'slowness') and args.slowness) else ""
        if verbose and checkattr(args, 'slowness') and args.slowness:
            print(" --> Slowness:            " + st_stamp)
    st_stamp = "--{}".format(st_stamp) if (
        (checkattr(args, 'slowness') and args.slowness)
    ) else ""

    # -for SI
    si_stamp = ""
    if checkattr(args, 'si') and args.si_c > 0:
        si_stamp = "SI{c}-{eps}".format(c=args.si_c, eps=args.epsilon) if (
                    checkattr(args, 'si') and args.si_c > 0) else ""
        if verbose and checkattr(args, 'si') and args.si_c > 0:
            print(" --> SI:            " + si_stamp)
    si_stamp = "--{}".format(si_stamp) if (
        (checkattr(args, 'si') and args.si_c > 0)
    ) else ""

    # -for XdG
    xdg_stamp = ""
    if (checkattr(args, "xdg") and args.xdg_prop > 0):
        xdg_stamp = "--XdG{}".format(args.xdg_prop)
        if verbose:
            print(" --> XdG:           " + "gating = {}".format(args.xdg_prop))

    # -for replay
    if replay:
        replay_stamp = "{rep}{bat}{distil}{model}{gi}".format(
            rep="gen" if args.replay == "generative" else args.replay,
            bat="" if (
                    (not hasattr(args, 'batch_replay')) or (
                        args.batch_replay is None) or args.batch_replay == args.batch
            ) else "-br{}".format(args.batch_replay),
            distil="-Di{}".format(args.temp) if args.distill else "",
            model="" if (replay_model_name is None) else "-{}".format(replay_model_name),
            gi="-gi{}".format(args.g_iters) if (
                    hasattr(args, "g_iters") and (replay_model_name is not None) and (not args.iters == args.g_iters)
            ) else "",
        )
        if verbose:
            print(" --> replay:        " + replay_stamp)
    replay_stamp = "--{}".format(replay_stamp) if replay else ""

    # -for choices regarding reconstruction loss
    if checkattr(args, "feedback"):
        recon_stamp = "--{}".format(args.recon_loss)
    elif hasattr(args, "replay") and args.replay == "generative":
        recon_stamp = "--{}".format(args.recon_loss)
    else:
        recon_stamp = ""

    # --> combine
    param_stamp = "{}--{}--{}{}{}{}{}{}{}{}{}{}".format(
        task_stamp, model_stamp, hyper_stamp, si_stamp, h_stamp, decay_stamp, top_neurons_stamp, st_stamp,
        xdg_stamp, replay_stamp, recon_stamp, "-s{}".format(args.seed) if not args.seed == 0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp
