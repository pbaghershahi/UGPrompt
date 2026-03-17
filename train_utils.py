import torch, random, os, ipdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from utils import *
from data_utils import *
from prompt_func import *
from model import *
import ipdb


def pretrain_model(
    s_dataset,
    model_config,
    optimizer_config,
    training_config,
    logger,
    eval_step = 1,
    save_model = True, 
    pretext_task = "classification",
    model_dir = "./pretrained",
    empty_pretrained_dir = False,
    tunning = False,
):
    binary_task = False if s_dataset.num_gclass > 2 else True
    if model_config["gnn_type"] in ["graphgps"]:
        model = GPS(**model_config)
    else:
        model = PretrainedModel(**model_config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if pretext_task == "classification":
        obj_fun = nn.CrossEntropyLoss()
    else:
        raise Exception("Pretext task is not implemented yet!")
    optimizer = Adam(model.parameters(), lr = optimizer_config["lr"], weight_decay = optimizer_config["weight_decay"])
    scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])
    
    test_resutls = test(model, s_dataset, device, binary_task = binary_task, mode = "pretrain", validation = False)
    logger.info(f'GNN Before Pretraining: -- Test Loss: {test_resutls["loss"]:.3f} -- Test ACC: {test_resutls["acc"]:.3f} -- Test F1-score: {test_resutls["f1"]:.3f} -- Test ECE: {test_resutls["ece"]:.3f}')
    # ipdb.set_trace()
    n_epochs = training_config["n_epochs"]
    for epoch in range(n_epochs):
        model.train()
        for i, (batch, idxs) in enumerate(s_dataset.train_loader):
            optimizer.zero_grad()
            scores, _ = model(
                batch,
                decoder = True,
                device = device
            )
            # ipdb.set_trace()
            loss = obj_fun(scores, batch.y.to(device))
            loss.backward()
            optimizer.step()
            # if i % max(1, int((s_dataset.n_train//scores.size(0))*0.5)) == 0:
            #     logger.info(f"Train batch: {i}/{np.ceil(s_dataset.n_train//scores.size(0))} -- Train Loss: {loss.item()}")
        scheduler.step()
        optimizer.zero_grad()

        if epoch % eval_step == 0 and epoch > 0:
            valid_results = test(model, s_dataset, device, binary_task = binary_task, mode = "pretrain", validation = True)
            logger.info(
                f"Epoch: {epoch}/{n_epochs} -- Train Loss: {loss:.3f} -- " +
                f"Validation Loss: {valid_results['loss']:.3f} -- Validation ACC: {valid_results['acc']:.3f} -- Validation F1: {valid_results['f1']:.3f} -- Validation ECE: {valid_results['ece']:.3f}"
            )

    test_results = test(model, s_dataset, device, binary_task = binary_task, mode = "pretrain", validation = False)
    logger.info(
        f"GNN After Pretraining: -- Train Loss: {loss:.3f} -- Test Loss: {test_results['loss']:.3f} -- Test ACC: {test_results['acc']:.3f} -- Test F1: {test_results['f1']:.3f} -- Test ECE: {test_results['ece']:.3f}"
    )
    if empty_pretrained_dir:
        empty_directory(model_dir)
    if save_model:
        os.makedirs(model_dir, exist_ok=True)
        exec_name = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        model_path = os.path.join(model_dir, f"{model_config['gnn_type']}_Pretrained_{exec_name}.pth")
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path
        )
        logger.info(f"Model saved to: {model_path}")
    else:
        model_path = None
        print("Model is not saved!")
    return model, model_path
    

def prompting(
    t_dataset,
    prompt_method, 
    prompt_config,
    pretrained_config,
    optimizer_config,
    pretrained_path,
    training_config,
    logger,
    s_dataset = None,
    num_runs = 5,
    eval_step = 1
):
    binary_task = False if s_dataset.num_gclass > 2 else True
    training_config["binary_task"] = binary_task
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_mode = "pretrain" if prompt_method == "head_tuning" else "prompt"

    if pretrained_config["gnn_type"] in ["graphgps"]:
        main_model = GPS(**pretrained_config)
    else:
        main_model = PretrainedModel(**pretrained_config)
        
    discr_config = {
        "in_channels":pretrained_config["hidden_channels"], 
        "hidden_channels":pretrained_config["hidden_channels"], 
        "out_channels":1, "num_layers":2, "dropout":pretrained_config["dropout"]
    }
    main_model.to(device)
    load_model(main_model, read_checkpoint=True, pretrained_path=pretrained_path)
    for name, param in main_model.named_parameters():
        if training_config["tune_decoder"] and name.startswith("linear_decoder"):
            continue
        param.requires_grad = False
        
    results = dict()
    # ipdb.set_trace()
    if s_dataset is not None:
        test_results = test(main_model, s_dataset, device, binary_task = binary_task, mode = "pretrain", validation = False)
        logger.info(f'Pretrained GNN on Source Dataset -- Test Loss: {test_results["loss"]:.3f} -- Test ACC: {test_results["acc"]:.3f} -- Test F1-score: {test_results["f1"]:.3f} -- Test ECE: {test_results["ece"]:.3f}')
        results |= dict(
            source_test_acc = test_results["acc"],
            source_test_f1 = test_results["f1"],
            source_test_ece = test_results["ece"],
        )
        
    valid_results = test(main_model, t_dataset, device, binary_task = binary_task, mode = "pretrain", validation = True)
    results |= dict(
        target_valid_acc = valid_results["acc"],
        target_valid_f1 = valid_results["f1"],
        target_valid_ece = valid_results["ece"],
    )
    
    test_results = test(main_model, t_dataset, device, binary_task = binary_task, mode = "pretrain", validation = False, cut_off = training_config["cut_off"])
    logger.info(
        f"Pretrained GNN on Target Dataset Without Prompting: -- " +\
        f"Validation Loss: {valid_results['loss']:.3f} -- Validation ACC: {valid_results['acc']:.3f} -- Validation F1-score: {valid_results['f1']:.3f} -- Validation ECE: {valid_results['ece']:.3f}" +\
        f"Test Loss: {test_results['loss']:.3f} -- Test ACC: {test_results['acc']:.3f} -- Test F1-score: {test_results['f1']:.3f} -- Test ECE: {test_results['ece']:.3f} -- Test Pseudo-ACC: {test_results['pseudo_acc']:.3f} -- Test Pseudo-F1: {test_results['pseudo_f1']:.3f}"
    )

    results |= dict(
        target_test_acc = test_results["acc"],
        target_test_f1 = test_results["f1"],
        target_test_ece = test_results["ece"],
        target_test_pseudo_acc = test_results["pseudo_acc"],
        target_test_pseudo_f1 = test_results["pseudo_f1"],
        prompt_test_acc = [],
        prompt_test_ece = [],
        prompt_test_f1 = [],
        prompt_valid_acc = [],
        prompt_valid_ece = [],
        prompt_valid_f1 = [],
    )
    
    for k in range(num_runs):

        if prompt_method == "all_in_one":
            pmodel = AllInOneOrginal(**prompt_config)
        elif prompt_method == "gpf_plus":
            pmodel = GPFPlus(**prompt_config)
        elif prompt_method == "graph_prompt":
            pmodel = GraphPrompt(**prompt_config)
        elif prompt_method == "graph_prompt_plus":
            pmodel = GraphPromptPlus(**prompt_config)
        elif prompt_method == "gppt":
            pmodel = GPPT(**prompt_config)
        elif prompt_method == "fix_match":
            pmodel = BasePrompt(**prompt_config)
        elif prompt_method == "flex_match":
            pmodel = BasePrompt(**prompt_config)
        elif prompt_method == "head_tuning":
            main_model = PretrainedModel(**pretrained_config)
            main_model.to(device)
            load_model(main_model, read_checkpoint=True, pretrained_path=pretrained_path)
            for name, param in main_model.named_parameters():
                if name.startswith("linear_decoder"):
                    continue
                param.requires_grad = False
            pmodel = HeadTuning(**prompt_config)
        else:
            raise NotImplementedError
        
        pmodel.to(device)
        main_model.eval()

        discriminator = Discriminator(**discr_config)
        discriminator.to(device)

        if prompt_method == "head_tuning":
            opt_params = main_model.parameters()
        elif training_config["tune_decoder"]:
            opt_params = list(pmodel.parameters()) + list(main_model.parameters())
        else:
            opt_params = pmodel.parameters()
        optimizer = Adam(opt_params, lr = optimizer_config["lr"], weight_decay = optimizer_config["weight_decay"])
        optimizer_d = Adam(discriminator.parameters(), lr = optimizer_config["lr"], weight_decay = optimizer_config["weight_decay"])
        scheduler = StepLR(optimizer, step_size = optimizer_config["scheduler_step_size"], gamma = optimizer_config["scheduler_gamma"])
        Trainer = PromptTrainer(prompt_method, training_config, device)
    
        valid_average_acc = []
        valid_average_f1 = []
        valid_average_ece = []
        n_epochs = training_config["n_epochs"]
        for epoch in range(n_epochs):
            pmodel.train()
            main_model.eval()
            loss = Trainer.train(
                t_dataset, main_model, pmodel, optimizer, logger, 
                discriminator = discriminator, optimizer_d = optimizer_d,
                n_shot_ratio = training_config["n_shot_ratio"]
            )
            scheduler.step()        
            optimizer.zero_grad()
            
            if epoch % eval_step == 0 or epoch >= n_epochs - 6:
                pmodel.eval()
                main_model.eval()
                valid_results = test(main_model, t_dataset, device, binary_task = binary_task, mode = test_mode, pmodel = pmodel, validation = True, cut_off = training_config["cut_off"])
                logger.info(f"Epoch: {epoch}/{n_epochs} -- Train Loss: {loss:.3f} -- Validation Loss: {valid_results['loss']:.3f} -- Validation ACC: {valid_results['acc']:.3f} -- Validation F1: {valid_results['f1']:.3f} -- Validation ECE: {valid_results['ece']:.3f}")
                if epoch >= n_epochs - 6:
                    valid_average_acc.append(valid_results["acc"])
                    valid_average_f1.append(valid_results["f1"])
                    valid_average_ece.append(valid_results["ece"])

        n_evali_valid = len(valid_average_f1)
        valid_average_acc = np.array(valid_average_acc).mean()
        valid_average_f1 = np.array(valid_average_f1).mean()
        valid_average_ece = np.array(valid_average_ece).mean()
        logger.info(f"Run {k}/{num_runs}: Average Over Last {n_evali_valid} epochs -- Valid ACC: {valid_average_acc} -- Valid F1-score: {valid_average_f1}")
        results["prompt_valid_acc"].append(valid_average_acc)
        results["prompt_valid_f1"].append(valid_average_f1)
        results["prompt_valid_ece"].append(valid_average_f1)

        test_results = test(main_model, t_dataset, device, binary_task = binary_task, mode = test_mode, pmodel = pmodel, validation = False, cut_off = training_config["cut_off"])
        logger.info(f"Test Results of Run {k}/{num_runs}: -- Test Loss: {test_results['loss']:.3f} -- Test ACC: {test_results['acc']:.3f} -- Test F1: {test_results['f1']:.3f} -- Test ECE: {test_results['ece']:.3f}")
        results["prompt_test_acc"].append(test_results["acc"])
        results["prompt_test_f1"].append(test_results["f1"])
        results["prompt_test_ece"].append(test_results["ece"])

        t_dataset.reset_preds()

    results.update(
        dict(
            prompt_valid_acc = np.array(results["prompt_valid_acc"]).mean(),
            prompt_valid_f1 = np.array(results["prompt_valid_f1"]).mean(),
            prompt_valid_ece = np.array(results["prompt_valid_ece"]).mean(),
            prompt_test_acc = np.array(results["prompt_test_acc"]).mean(),
            prompt_test_f1 = np.array(results["prompt_test_f1"]).mean(),
            prompt_test_ece = np.array(results["prompt_test_ece"]).mean(),
        )
    )
    logger.info(f"Validation average after {num_runs} runs -- ACC: {results['prompt_valid_acc']:.3f} -- F1-score: {results['prompt_valid_f1']:.3f} -- ECE: {results['prompt_valid_ece']:.3f}")
    logger.info(f"Test average after {num_runs} runs -- ACC: {results['prompt_test_acc']:.3f} -- F1-score: {results['prompt_test_f1']:.3f} -- ECE: {results['prompt_test_ece']:.3f}")
    return pmodel, results