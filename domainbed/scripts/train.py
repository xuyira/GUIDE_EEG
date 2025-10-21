# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import (
    InfiniteDataLoader,
    FastDataLoader,
    FastDataLoader_no_shuffle,
)

from .helpers import *
from .clustering import Faiss_Clustering
from .preprocess import *

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str, default="RotatedMNIST")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--task",
        type=str,
        default="domain_generalization",
        choices=["domain_generalization", "domain_adaptation"],
    )
    parser.add_argument("--hparams", type=str, help="JSON-serialized hparams dict")
    parser.add_argument(
        "--hparams_seed",
        type=int,
        default=0,
        help='Seed for random hparams (0 means "default hparams")',
    )
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and " "random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of steps. Default is dataset-dependent.",
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=[0])
    parser.add_argument("--output_dir", type=str, default="train_output")
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument(
        "--uda_holdout_fraction",
        type=float,
        default=0,
        help="For domain adaptation, % of test to use unlabeled for training.",
    )
    parser.add_argument("--skip_model_save", action="store_true")
    parser.add_argument("--save_model_every_checkpoint", action="store_true")

    parser.add_argument(
        "--no_pca", action="store_true", help="Clustering without SVD + Truncation step"
    )
    parser.add_argument(
        "--clust_step", type=int, default=None, help="step to perform clustering"
    )
    parser.add_argument(
        "--num_clusters", type=int, default=None, help="Number of clusters"
    )
    parser.add_argument(
        "--clustering_batch_size", type=int, default=8, help="Batch size for clustering"
    )
    parser.add_argument(
        "--random_centroids",
        action="store_true",
        help="Random centroids for clustering",
    )
    parser.add_argument(
        "--cluster_space", type=str, choices=["cluster", "org"], default="org"
    )
    parser.add_argument(
        "--cosine_sched", action="store_true", help="Use cosine scheduler"
    )

    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, "out.txt"))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, "err.txt"))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print("Args:")
    for k, v in sorted(vars(args).items()):
        print("\t{}: {}".format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(
            args.algorithm,
            args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed),
        )
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print("HParams:")
    for k, v in sorted(hparams.items()):
        print("\t{}: {}".format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    if "GUIDE" in args.algorithm:
        num_clusters = (
            args.num_clusters or hparams["num_clusters"] * dataset.num_classes
        )  # Set number of clusters
        print("NUM CLUSTERS: ", num_clusters)

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    test_data_sep = []
    train_data_sep = []
    # train_data_sep_diff = []
    # test_data_sep_diff = []
    eval_loader_names = []
    in_splits = []
    out_splits = []
    train_domain_labels = []
    for env_i, (env) in enumerate(dataset):
        print(f"Processing env {env_i}")
        uda = []
        out, in_ = misc.split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        # out_diff, in_diff = misc.split_dataset(
        #     diff_env,
        #     int(len(diff_env) * args.holdout_fraction),
        #     misc.seed_hash(args.trial_seed, env_i),
        # )
        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(
                in_,
                int(len(in_) * args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i),
            )
            # uda_diff, in_diff = misc.split_dataset(
            #     in_diff,
            #     int(len(in_diff) * args.uda_holdout_fraction),
            #     misc.seed_hash(args.trial_seed, env_i),
            # )
        test_data_sep.append(in_)
        # test_data_sep_diff.append(in_diff)
        eval_loader_names += ["env{}_in".format(env_i)]
        test_data_sep.append(out)
        # test_data_sep_diff.append(out_diff)
        eval_loader_names += ["env{}_out".format(env_i)]
        if env_i not in args.test_envs:
            train_data_sep.append(in_)
            # train_data_sep_diff.append(in_diff)
            train_domain_labels.extend([env_i] * len(in_))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_data = MyDataloader(train_data_sep)  # Concat train data
    test_data = MyDataloader(test_data_sep)  # Concat test data

    # train_data_diff = MyDataloader(train_data_sep_diff)  # Concat train data
    # test_data_diff = MyDataloader(test_data_sep_diff)  # Concat test data

    print(len(train_data), len(test_data))

    # print image sizes
    img = train_data[0][0]
    print("Image size: ", img[0].size())
    print("Image size: ", img[1].size())

    # img, _ = train_data_diff[0][0]
    # print("Image size: ", img.size())

    len_train_data = len(train_data)
    len_test_data = len(test_data)

    # Loaders to perform clustering
    if "GUIDE" in args.algorithm:
        train_loader = FastDataLoader_no_shuffle(
            dataset=train_data,
            batch_size=args.clustering_batch_size,
            num_workers=4,
        )
        test_loader = FastDataLoader_no_shuffle(
            dataset=test_data,
            batch_size=args.clustering_batch_size,
            num_workers=4,
        )

    # DomainBed dataloaders
    train_idx_split = get_data_split_idx(train_data_sep)
    train_loaders = [
        InfiniteDataLoader(
            dataset=torch.utils.data.Subset(train_data, idx),
            weights=None,
            batch_size=hparams["batch_size"],
            num_workers=0,
        )
        for idx in train_idx_split
    ]
    train_minibatches_iterator = zip(*train_loaders)

    test_idx_split = get_data_split_idx(test_data_sep)
    eval_loaders = [
        FastDataLoader(
            dataset=torch.utils.data.Subset(test_data, idx),
            batch_size=64,
            num_workers=0,
        )
        for idx in test_idx_split
    ]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    # Set flag for GUIDE specific operations
    if "GUIDE" in args.algorithm:
        cluster = True
    else:
        cluster = False
    
    # uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    # steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])
    steps_per_epoch = int(len(train_data) / hparams["batch_size"])

    n_steps = args.steps or dataset.N_STEPS
    print(f"Number of steps per epoch: {steps_per_epoch}")
    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    epochs = int(n_steps / steps_per_epoch)

    if args.cosine_sched:
        algorithm.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(algorithm.optimizer, T_max=n_steps) # Initialize scheduler

    # Set clustering schedule
    if cluster:
        cluster_step = args.clust_step
        if args.clust_step is None:
            cluster_step = steps_per_epoch * hparams["clust_epoch"] # Cluster every hparams["clust_epoch"] epochs
        cluster_step = [(x*cluster_step) for x in range(n_steps) if (x*cluster_step)<= n_steps]
        if hparams["clust_epoch"]==0: # cluster every 2**n epochs (0, 1, 2, 4, 8, 16, ...)
            cluster_step = [((2 ** x)*steps_per_epoch) for x in range(epochs) if (2**x)<= epochs] # store the steps at which clustering take place
        print(f"Cluster every {cluster_step} steps")
    else:
        cluster_step = [-1] # dummy value

    # cluster_step = [0]

    # from range 0 to 100 timesteps, assign a timestep to a cluster step
    num_cluster_steps = len(cluster_step)
    print(f"Number of cluster steps: {num_cluster_steps}")

    print("Cluster step: ", cluster_step)

    time_step_per_cluster_dict = {}
    MAX_TIME_STEP = 200

    # for idx, step in enumerate(cluster_step):
    #     reversed_idx = num_cluster_steps - idx  # Reverse the index
    #     # time_step_per_cluster_dict[step] = int((reversed_idx / num_cluster_steps) * MAX_TIME_STEP)
    #     time_step_per_cluster_dict[step] = 50

    # if timestep does not exist in hparams, set it to default value
    if "timestep" not in hparams:
        hparams["timestep"] = -1

    time_step_per_cluster_dict[0] = hparams["timestep"]
    for idx, step in enumerate(cluster_step):
        time_step_per_cluster_dict[step] = hparams["timestep"]

    print(f"Time step per cluster step: {time_step_per_cluster_dict}")

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict(),
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    def return_centroids(algorithm, step, timestep):

        with torch.no_grad():
            # Get features and labels
            org_train_features, train_features, train_labels = get_features_V2(train_loader, algorithm, len_train_data, 8, device, timestep)
            org_test_features, test_features, test_labels = get_features_V2(test_loader, algorithm, len_test_data, 8, device, timestep)
            train_labels = torch.Tensor(train_labels).type(torch.LongTensor)
            test_labels = torch.Tensor(test_labels).type(torch.LongTensor)

        print("Train Features: ", train_features.shape)
        print("Org Train Features: ", org_train_features.shape)
        print("Test Features: ", test_features.shape)
        print("Org Test Features: ", org_test_features.shape)
            
        exp_var = 1
        # Clustering on Train Data
        if args.no_pca:
            train_features2 = train_features # if no PCA
            row_sums = np.linalg.norm(train_features2, axis=1)
            train_features2 = train_features2 / row_sums[:, np.newaxis]
        else:
            print("Doing Truncated PCA, with PCA dim: ", hparams["pca_dim"], "Offset: ", hparams["offset"])
            train_features_pca = np.asarray(train_features)
            pca = PCA(hparams["pca_dim"])
            exp_var = pca.fit(train_features, hparams["offset"])
            train_features2 = pca.apply(torch.from_numpy(train_features_pca)).detach().numpy()
            row_sums = np.linalg.norm(train_features2, axis=1)
            train_features2 = train_features2 / row_sums[:, np.newaxis]

        print("Clutsering at step: ", step)
        clustering = Faiss_Clustering(train_features2.copy(order="C"), num_clusters)
        clustering.fit()
        print("Clustering Done")
        cluster_labels_train = get_cluster_labels(clustering, train_features2)
        if args.random_centroids:
            print("Using Random Centroids")
            cluster_labels_train = np.random.randint(0, num_clusters, size=len_test_data)
        images_lists = get_images_list(num_clusters, len_train_data, cluster_labels_train)

        if args.cluster_space == "org":
            print("Using Original Space", org_train_features.shape)
            train_centroids = torch.empty((len_train_data, org_train_features.shape[1]))
            # Get the centroid of the images that share the same cluster in PCA space
            for i, indx in enumerate(images_lists):
                if len(indx) > 0:
                    train_centroids[indx] = torch.Tensor(org_train_features[indx].mean(axis=0))
        elif args.cluster_space == "cluster":
            print("Using Cluster Space", train_features.shape)
            train_centroids = torch.empty((len_train_data, train_features.shape[1]))
            for i, indx in enumerate(images_lists):
                if len(indx) > 0:
                    train_centroids[indx] = torch.Tensor(train_features[indx].mean(axis=0))

        # Clustering on Test Data
        if args.no_pca:
            test_features2 = test_features
            row_sums = np.linalg.norm(test_features2, axis=1)
            test_features2 = test_features2 / row_sums[:, np.newaxis]
        else:
            test_features_pca = np.asarray(test_features)
            test_features2 = pca.apply(torch.from_numpy(test_features_pca)).detach().numpy()
            row_sums = np.linalg.norm(test_features2, axis=1)
            test_features2 = test_features2 / row_sums[:, np.newaxis]

        cluster_labels_test = get_cluster_labels(clustering, test_features2)
        if args.random_centroids:
            print("Using Random Centroids at Test")
            cluster_labels_test = np.random.randint(0, num_clusters, size=len_test_data)
        images_lists = get_images_list(num_clusters, len_test_data, cluster_labels_test)

        if args.cluster_space == "org":
            print("Using Original Space: ", org_test_features.shape)
            test_centroids = torch.empty((len_test_data, org_test_features.shape[1]))
            for i, indx in enumerate(images_lists):
                if len(indx) > 0:
                    test_centroids[indx] = torch.Tensor(org_test_features[indx].mean(axis=0))
        elif args.cluster_space == "cluster":
            print("Using Cluster Space", test_features.shape)
            test_centroids = torch.empty((len_test_data, test_features.shape[1]))
            for i, indx in enumerate(images_lists):
                if len(indx) > 0:
                    test_centroids[indx] = torch.Tensor(test_features[indx].mean(axis=0))

        print("Train Centroids: ", train_centroids.shape)
        print("Test Centroids: ", test_centroids.shape)

        return train_centroids, test_centroids, exp_var

    last_results_keys = None
    train_centroids = None
    test_centroids = None
    # for step in range(start_step, n_steps):
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        if step == 0 or (step in cluster_step):
            if cluster:
                train_centroids, test_centroids, exp_var = return_centroids(algorithm, step, timestep = time_step_per_cluster_dict[step])
                # # # save centroids as .pt file
                # # cleaned model name, replace slapshes with underscores
                model_name = hparams["model_name"].replace("/", "_")
                # torch.save(train_centroids, f"train_centroids_{model_name}_{time_step_per_cluster_dict[step]}_no_pca_{args.no_pca}.pt")
                # torch.save(test_centroids, f"test_centroids_{model_name}_{time_step_per_cluster_dict[step]}_no_pca_{args.no_pca}.pt")

                # # # Load centroids
                # train_centroids = torch.load(f"train_centroids_{model_name}_{time_step_per_cluster_dict[step]}_no_pca_{args.no_pca}.pt")
                # test_centroids = torch.load(f"test_centroids_{model_name}_{time_step_per_cluster_dict[step]}_no_pca_{args.no_pca}.pt")
                # exp_var = 0
                # exp_var = 1
            else:
                test_centroids = None

        if cluster:
            minibatches_device = [
                (x[0].to(device), train_centroids[idx].to(device), x[2].to(device))
                for (x, idx) in next(train_minibatches_iterator)
            ]
        else:
            minibatches_device = [
                (x[0].to(device), x[2].to(device))
                for (x, idx) in next(train_minibatches_iterator)
            ]
        # if args.task == "domain_adaptation":
        #     uda_device = [x.to(device)
        #         for x,_ in next(uda_minibatches_iterator)]
        # else:
        uda_device = None
        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals["step_time"].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                "step": step,
                "epoch": step / steps_per_epoch,
            }
            if args.cosine_sched:
                results['lr'] = algorithm.scheduler.get_last_lr()[0]

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # evals = zip(eval_loader_names, eval_loaders, eval_weights)
            # for name, loader, weights in evals:
            #     acc = misc.accuracy(algorithm, loader, weights, device)
            #     results[name+'_acc'] = acc

            evals = zip(eval_loader_names, eval_loaders)
            for i, (name, loader) in enumerate(evals):
                acc = misc.accuracy(algorithm, loader, None, device, test_centroids)
                results[name + "_acc"] = acc

            results["mem_gb"] = torch.cuda.max_memory_allocated() / (
                1024.0 * 1024.0 * 1024.0
            )

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({"hparams": hparams, "args": vars(args)})

            epochs_path = os.path.join(args.output_dir, "results.jsonl")
            with open(epochs_path, "a") as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f"model_step{step}.pkl")

    save_checkpoint("model.pkl")

    with open(os.path.join(args.output_dir, "done"), "w") as f:
        f.write("done")
