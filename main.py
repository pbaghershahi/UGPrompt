from train_utils import *
from utils import *
from data_utils import *
import yaml
import argparse
import ipdb


class PromptTrialResults(object):
    def __init__(self) -> None:
        self.results = dict()

    def update(self, results_dict):
        if self.results is None:
            self.results = {key:[] for key in results_dict.keys()}
        for key, value in results_dict.items():
            if key in self.results:
                self.results[key].append(value)
            else:
                self.results[key] = []
                self.results[key].append(value)

    def average_results(self,):
        for key, value in self.results.items():
            self.results[key] = torch.as_tensor(value).mean()
    
    def report_results(self, logger,):
        report = "Final results:"
        for key, value in self.results.items():
            key_str = " ".join([s.capitalize() for s in key.split("_")])
            report += f" -- {key_str}: {value:.3f}"
        logger.info(report)


def main(args) -> None:
    os.makedirs('./log', exist_ok=True)
    os.makedirs('./config', exist_ok=True)
    exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    log_file_path = "./log/"+exec_name+".log"
    logger = setup_logger(name=exec_name, level=logging.INFO, log_file=log_file_path)
    logger.info(f"Logging to: {log_file_path}")
    if args.config_from_file != "":
        logger.info(f"Reading config from: {args.config_from_file}")
        with open(args.config_from_file, 'r') as infile:
            all_args = vars(args)
            input_args = []
            for key, value in all_args.items():
                if value is not None:
                    input_args.append(key)
            file_args = yaml.safe_load(infile)
            args = {key:file_args[key] if (key in file_args and key not in input_args) else value for key, value in all_args.items()}
            args = argparse.Namespace(**args)
    arg_seeds = np.random.randint(1000, 5000, (args.total_iters,)) if len(args.seed) == 0 else args.seed
    total_iters = len(arg_seeds)
    t_ds_name = args.t_dataset if args.t_dataset != args.s_dataset else args.s_dataset
    args.s_split = args.s_split if len(args.s_split) > 0 else args.t_split
    logger.info(args)
    logger.info("#"*100)
    logger.info(f"Args seeds: {arg_seeds}")
    logger.info(f"Source Dataset: {args.s_dataset}, Target Dataset: {t_ds_name}. Training, Test: Source: {args.s_split}, Target: {args.t_split} -- Batch size: {args.batch_size}")

    results = PromptTrialResults()
    pretrained_paths = args.pretrained_path if len(args.pretrained_path) > 0 else []
    logger.info(f"All input pretrained paths: {pretrained_paths}")
    for i in range(total_iters):
        logger.info(f"Started round {i}/{total_iters} of experiments!")
        gen_ds = GenDataset(logger)
        if args.s_dataset in ["Cora", "CiteSeer", "PubMed"]:
            s_dataset, t_dataset = gen_ds.get_node_dataset(
                args.s_dataset,
                shift_type = args.shift_type,
                p_intra = args.p_shift_intra,
                p_inter = args.p_shift_inter,
                cov_scale = args.noise_cov_scale,
                mean_shift = args.noise_mean_shift,
                shift_mode = args.noise_shift_mode,
                s_split = args.s_split,
                t_split = args.t_split,
                batch_size = args.batch_size,
                n_hopes = 2,
                norm_mode = "normal",
                node_attributes = True,
                label_reduction = args.label_reduction,
                seed = arg_seeds[i],
                select_mode = args.noise_select_mode
            )
        elif args.s_dataset in ["ENZYMES", "PROTEINS", "Mutagenicity", "AIDS", "NCI1", "DHFR", "COX2"]:
            s_dataset, t_dataset = gen_ds.get_graph_dataset(
                args.s_dataset,
                shift_type = args.shift_type,
                p_intra = args.p_shift_intra,
                p_inter = args.p_shift_inter,
                cov_scale = args.noise_cov_scale,
                mean_shift = args.noise_mean_shift,
                shift_mode = args.noise_shift_mode,
                store_to_path = "./data/TUDataset",
                s_split = args.s_split,
                t_split = args.t_split,
                src_ratio = args.src_ratio,
                batch_size = args.batch_size,
                norm_mode = "normal",
                node_attributes = True,
                label_reduction = args.label_reduction,
                seed = arg_seeds[i],
                select_mode = args.noise_select_mode
            )
        elif args.s_dataset in ["Letter-high", "Letter-low", "Letter-med"]:
            s_dataset, t_dataset = gen_ds.get_pyggda_dataset(
                args.s_dataset,
                t_ds_name,
                store_to_path = "./data/TUDataset",
                s_split = args.s_split,
                t_split = args.t_split,
                batch_size = args.batch_size,
                norm_mode = "normal",
                node_attributes = True,
                label_reduction = args.label_reduction,
                seed = arg_seeds[i]
            )
        elif args.s_dataset in ["digg", "oag", "twitter", "weibo"]:
            s_dataset, t_dataset = gen_ds.get_gda_dataset(
                ds_dir = "./data/ego_network/",
                s_ds_name = args.s_dataset,
                t_ds_name = t_ds_name,
                s_split = args.s_split,
                t_split = args.t_split,
                batch_size = args.batch_size,
                get_s_dataset = True,
                get_t_dataset = True,
                label_reduction = args.label_reduction,
                seed = arg_seeds[i]
            )

        model_name = "GCN"
        model_config = dict(
            gnn_type = args.gnn_type,
            in_channels = s_dataset.n_feats,
            hidden_channels = args.gnn_h_dim,
            out_channels = s_dataset.num_gclass,
            num_layers = args.gnn_num_layers, 
            dropout = args.gnn_dropout,
            with_bn = False,
            with_head = True,
        )
        optimizer_config = dict(
            lr = args.gnn_lr,
            scheduler_step_size = args.gnn_step_size,
            scheduler_gamma = args.gnn_gamma,
            weight_decay = args.gnn_weight_decay
        )
        training_config = dict(
            n_epochs = args.gnn_n_epochs
        )
        if i % 5 == 0:
            logger.info(f"Setting for pretraining: Model: {model_config} -- Optimizer: {optimizer_config} -- Training: {training_config}")

        if len(args.pretrained_path) == 0:
            logger.info(f"Pretraining {model_name} on {args.s_dataset} started for {args.gnn_n_epochs} epochs")
            _, p_path = pretrain_model(
                s_dataset,
                model_name,
                model_config,
                optimizer_config,
                training_config,
                logger,
                eval_step = args.gnn_eval_step,
                save_model = True,
                pretext_task = "classification",
                model_dir = "./pretrained",
                empty_pretrained_dir = args.empty_pretrained_dir
            )
            pretrained_paths.append(p_path)
        else:
            logger.info(f"Loading previous pretrained model at {pretrained_paths[i]}")

        pretrained_config = model_config
        num_tokens = int(np.ceil(cal_avg_num_nodes(t_dataset))) if args.num_tokens == -1 else args.num_tokens
        logger.info(f"Total number of tokens: {num_tokens}")
        optimizer_config = dict(
            lr = args.lr,
            scheduler_step_size = args.step_size,
            scheduler_gamma = args.gamma,
            weight_decay = args.weight_decay
        )
        # ipdb.set_trace()
        if args.prompt_method == "all_in_one_original":
            prompt_config = dict(
                token_dim = t_dataset.n_feats,
                token_num = num_tokens,
                cross_prune = args.cross_prune,
                inner_prune = args.inner_prune,
            )
            training_config = dict(
                n_epochs = args.n_epochs,
                r_reg = args.r_reg
            )
        elif args.prompt_method == "all_in_one_modified":
            prompt_config = dict(
                token_dim = t_dataset.n_feats,
                token_num = num_tokens,
                cross_prune = args.cross_prune,
                inner_prune = args.inner_prune,
            )
            training_config = dict(
                n_epochs = args.n_epochs,
                r_reg = args.r_reg
            )
        elif args.prompt_method == "gpf_plus":
            prompt_config = dict(
                token_dim = t_dataset.n_feats,
                token_num = num_tokens,
            )
            training_config = dict(
                n_epochs = args.n_epochs,
                r_reg = args.r_reg
            )
        elif args.prompt_method == "contrastive":
            prompt_config = dict(
                emb_dim = t_dataset.n_feats,
                h_dim = args.h_dim,
                output_dim = t_dataset.n_feats,
                prompt_fn = args.prompt_fn,
                token_num = num_tokens
            )
            training_config = dict(
                aug_type = args.aug_type,
                pos_aug_mode = args.pos_aug_mode,
                neg_aug_mode = args.neg_aug_mode,
                p_raug = args.p_raug,
                n_raug = args.n_raug,
                add_link_loss = args.add_link_loss,
                n_epochs = args.n_epochs,
                r_reg = args.r_reg
            )
        elif args.prompt_method == "pseudo_labeling":
            prompt_config = dict(
                emb_dim = t_dataset.n_feats,
                h_dim = args.h_dim,
                output_dim = t_dataset.n_feats,
                prompt_fn = args.prompt_fn,
                token_num = num_tokens,
                cross_prune = args.cross_prune,
                inner_prune = args.inner_prune,
            )
            training_config = dict(
                aug_type = args.aug_type,
                pos_aug_mode = args.pos_aug_mode,
                p_raug = args.p_raug,
                n_epochs = args.n_epochs,
                r_reg = args.r_reg,
                soft_label = args.soft_label,
                clutering_iters = args.clutering_iters,
                iterative_clustering = args.iterative_clustering,
                entropy_div_ratio = args.entropy_div_ratio,
                w_entropy_loss = args.w_entropy_loss,
                w_softmax_loss = args.w_softmax_loss,
                w_domain_loss = args.w_domain_loss,
            )
        elif args.prompt_method in ["fix_match", "flex_match"]:
            prompt_config = dict(
                emb_dim = t_dataset.n_feats,
                h_dim = args.h_dim,
                output_dim = t_dataset.n_feats,
                prompt_fn = args.prompt_fn,
                token_num = num_tokens,
                cross_prune = args.cross_prune,
                inner_prune = args.inner_prune,
                attn_with_param = args.attn_with_param,
                attn_dropout = args.dropout
            )
            training_config = dict(
                aug_type = args.aug_type,
                pos_aug_mode = args.pos_aug_mode,
                p_raug = args.p_raug,
                n_epochs = args.n_epochs,
                r_reg = args.r_reg,
                soft_label = args.soft_label,
                clutering_iters = args.clutering_iters,
                iterative_clustering = args.iterative_clustering,
                entropy_div_ratio = args.entropy_div_ratio,
                w_entropy_loss = args.w_entropy_loss,
                w_softmax_loss = args.w_softmax_loss,
                w_domain_loss = args.w_domain_loss,
                light_aug_prob = args.light_aug_prob,
                light_aug_mode = args.light_aug_mode,
            )
        else:
            raise Exception("The chosen method is not valid!")

        training_config.update(num_classes = t_dataset.num_gclass, cut_off = args.cut_off)
        if i % 5 == 0:
            logger.info(f"Prompting method: {args.prompt_method} -- Setting: Prompting function: {args.prompt_fn} -- Target Dataset: {t_ds_name}")
            logger.info(f"Setting for prompt tuning: Prompt: {prompt_config} -- Pretrained Model: {pretrained_config} -- Optimizer: {optimizer_config} -- Training: {training_config}")
            logger.info(f"Prompt tuning started: Num runs: {args.num_runs} -- Eval step: {args.eval_step}")
        pmodel, temp_results = prompting(
            t_dataset,
            args.prompt_method,
            prompt_config,
            pretrained_config,
            optimizer_config,
            pretrained_paths[i],
            training_config,
            logger,
            s_dataset,
            num_runs = args.num_runs,
            eval_step = args.eval_step
        )
        results.update(temp_results)
        logger.info(f"Finished round {i}/{total_iters} of experiments!")

    results.average_results()
    results.report_results(logger)

    logger.info("List of important configs: ")
    for akey, avalue in vars(args).items():
        if akey in "prompt-method, num-tokens, shift-type, noise-shift-mode, attn-with-param, w-entropy-loss, w-softmax-loss, w-domain-loss, weight-decay, n-epochs, lr, batch-size, dropout, cut-off, light-aug-prob, light-aug-mode, pos-aug-mode, p-raug, label-reduction".replace(" ", "").replace("-", "_").split(","):
            logger.info(f"{akey}: {avalue}")
            
    if args.config_to_file != "":
        with open(args.config_to_file, 'w') as outfile:
            yaml.dump(vars(args), outfile, default_flow_style=False)
        logger.info(f"Config saved to file: {args.config_to_file}")
    logger.info(f"Pretrained modela are saved to: {' '.join(pretrained_paths)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UPrompt')
    parser.add_argument("-itrs", "--total-iters", type=int, default=2, help="Total number of trials with random initialization of datasets")
    parser.add_argument("-pm", "--prompt-method", type=str, 
                        help="Prompting method: [contrastive, pseudo_labeling, all_in_one_original, all_in_one_modified, gpf_plus]")
    parser.add_argument("-sds", "--s-dataset", type=str, 
                        help="Source dataset: [Cora, CiteSeer, PubMed, ENZYMES, PROTEINS_full, AIDS, Letter-high, Letter-low, Letter-med, digg, oag, twitter, weibo]")
    parser.add_argument("-tds", "--t-dataset", type=str, 
                        help="Target dataset: [Cora, CiteSeer, PubMed, ENZYMES, PROTEINS_full, AIDS, Letter-high, Letter-low, Letter-med, digg, oag, twitter, weibo]")
    parser.add_argument("-pf", "--prompt-fn", type=str, 
                        help="Prompting function for pseudo_labeling method: [add_tokens, gpf_plus, tucker]")
    parser.add_argument("-shift", "--shift-type", type=str, default="structural", help="Shift Type: [feature, structural]")
    parser.add_argument("-p-intra", "--p-shift-intra", type=float, default=0.0, help="Probability of intra class structural shift")
    parser.add_argument("-p-inter", "--p-shift-inter", type=float, default=0.0, help="Probability of inter class structural shift")
    parser.add_argument("--pretrain", action='store_true')
    parser.add_argument("--save-pretrained", action='store_true')
    parser.add_argument("--soft-label", action='store_true')
    parser.add_argument("--add-link-loss", action='store_true')
    parser.add_argument("--attn-with-param", action='store_true')
    parser.add_argument("--empty-pretrained-dir", action='store_true')
    parser.add_argument("--iterative-clustering", action='store_true')
    parser.add_argument("--clutering-iters", type=int, help="Total rounds of updating cluster centers")
    parser.add_argument("--entropy-div-ratio", type=int, help="Ratio of deviding samples based on entropy for clustering")
    parser.add_argument("--w-entropy-loss", type=float, help="Weight of entropy loss")
    parser.add_argument("--w-softmax-loss", type=float, help="Weight of softmax loss")
    parser.add_argument("--w-domain-loss", type=float, help="Weight of domain loss")
    parser.add_argument("--r-reg", type=float, help="Rate of regularization")
    parser.add_argument("--gnn-weight-decay", type=float, default=0.0, help="Rate of regularization")
    parser.add_argument("--weight-decay", type=float, help="Rate of regularization")
    parser.add_argument("-gnn", "--gnn-type", type=str, default="gcn", help="Type of base GNN: [gcn, gat, gin, sage]")
    parser.add_argument("--gnn-num-layers", type=int, help="Number of layers of the base GNN")
    parser.add_argument("--pretrained-path", nargs='*', default=[], type=str, help="Paths to the pretrained model")
    parser.add_argument("--gnn-n-epochs", type=int, help="Number of epochs for pretraining")
    parser.add_argument("--gnn-eval-step", type=int, default=10, help="Evaluation step for pretrained model")
    parser.add_argument("--gnn-h-dim", type=int, help="Hidden dim of the GNN")
    parser.add_argument("--gnn-lr", type=float, help="Learning rate for pretraining")
    parser.add_argument("--gnn-step-size", type=int, help="Learning rate step size for pretraining")
    parser.add_argument("--gnn-gamma", type=float, help="Learning rate gamma for pretraining")
    parser.add_argument("--gnn-batch-size", type=int, help="Batch size for pretraining")
    parser.add_argument("--gnn-dropout", type=float, help="Dropout for GNN")
    parser.add_argument("--n-epochs", type=int, help="Number of epochs for prompt tuning")
    parser.add_argument("--eval-step", type=int, help="Evaluation step for prompt tuning")
    parser.add_argument("--h-dim", type=int, help="Hidden dim for prompt tuning")
    parser.add_argument("-lr", "--lr", type=float, help="Learning rate for prompt tuning")
    parser.add_argument("--step-size", type=int, help="Learning rate step size for prompt tuning")
    parser.add_argument("--gamma", type=float, help="Learning rate gamma for prompt tuning")
    parser.add_argument("--batch-size", type=int, help="Batch size for prompt tuning")
    parser.add_argument("--num-tokens", type=int, help="Number of Tokens")
    parser.add_argument("--src-ratio", type=float, help="Split ratio for the source dataset")
    parser.add_argument("--dropout", type=float, help="Dropout for prompt tuning")
    parser.add_argument("--cut-off", type=float, help="Cut-off threshold for pseudo labels")
    parser.add_argument("--aug-type", type=str, default="feature", help="Augmentation Type: [feature, structural]")
    parser.add_argument("--light-aug-prob", type=float, help="Probability of light augmentation augmentation")
    parser.add_argument("--light-aug-mode", type=str, default="mask", help="Light augmentation mode: [mask, arbitrary]")
    parser.add_argument("--pos-aug-mode", type=str, default="mask", help="Positive augmentation mode: [mask, arbitrary]")
    parser.add_argument("--neg-aug-mode", type=str, default="arbitrary", help="Negative augmentation mode: [mask, arbitrary]")
    parser.add_argument("--noise-cov-scale", type=float, help="Noise covariance scale")
    parser.add_argument("--noise-mean-shift", type=float, help="Noise mean")
    parser.add_argument("--noise-shift-mode", type=str, help="Noise shift mode: [homophily]")
    parser.add_argument("--noise-select-mode", type=str, default="soft", help="Shift selection mode for spliting dataset: [soft, hard]")
    parser.add_argument("--cross-prune", type=float, help="Cross prune threshold")
    parser.add_argument("--inner-prune", type=float, help="Inner prune threshold")
    parser.add_argument("--num-runs", type=int, help="Numer of runs per trial")
    parser.add_argument("--p-raug", type=float, help="Positive augmentation rate")
    parser.add_argument("--n-raug", type=float, help="Negative augmentation rate")
    parser.add_argument("--s-split", nargs='*', default=[], type=float, help="source dataset split ratio: [train_percentage, test_percentage]")
    parser.add_argument("--t-split", nargs='*', type=float, help="target dataset split ratio: [train_percentage, test_percentage]")
    parser.add_argument("--seed", nargs='*', default=[], type=int, help="Seed for random")
    parser.add_argument("--config-from-file", type=str, default="", help="Config file to read from")
    parser.add_argument("--config-to-file", type=str, default="", help="Config file to save to")
    parser.add_argument("--label-reduction", type=float, help="Percentage of label reduction")
    args = parser.parse_args()
    main(args)