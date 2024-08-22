import torch, random, os, ipdb
import torch.nn.functional as F
import numpy as np
from utils import *
from data_utils import *
from model import *
from scipy.spatial.distance import cdist
from scipy.special import softmax


class PromptTrainer():
    def __init__(self, training_method, training_config, device, *args, **kwargs) -> None:
        super(PromptTrainer, self).__init__()
        self.training_config = training_config
        if training_method == "supervised":
            self.train_func = self.supervised
        elif training_method == "fix_match":
            self.train_func = self.consistency
            self.get_mask= self.fix_match_mask
        elif training_method == "flex_match":
            self.train_func = self.consistency
            self.get_mask = self.flex_match_mask
        self.classwise_beta = torch.zeros((training_config["num_classes"],), dtype=torch.float, device=device)
        self.cut_off = training_config["cut_off"]
        self.device = device

    def consistency(self, pretrained_model, prompt_model, batch, idxs, t_dataset, **kwargs):
        batch = batch.to_data_list()
        idxs = idxs.to(self.device)
        pos_batch = [
            aug_graph(
                Data(x = g.x, edge_index = g.edge_index, y = g.y), 
                aug_prob = self.training_config["light_aug_prob"], 
                aug_type = self.training_config["aug_type"], 
                mode = self.training_config["light_aug_mode"]
            ).to(self.device)
            for g in batch
        ]
        prompt_batch = [
            aug_graph(
                Data(x = g.x, edge_index = g.edge_index, y = g.y), 
                aug_prob = self.training_config["p_raug"], 
                aug_type = self.training_config["aug_type"], 
                mode = self.training_config["pos_aug_mode"]
            ).to(self.device)
            for g in batch
        ]
        prompt_batch = prompt_model(prompt_batch)
        prompt_out, prompt_embed = pretrained_model(
            prompt_batch,
            decoder = True,
            device = self.device
            )
        pos_out, pos_embed = pretrained_model(
            pos_batch,
            decoder = True,
            device = self.device
            )
        pos_out = F.softmax(pos_out, dim=1)
        # ipdb.set_trace()
        thresh_mask = self.get_mask(pos_out, idxs_ulb=idxs, t_dataset=t_dataset)
        pos_out = pos_out[thresh_mask, :]
        prompt_out = prompt_out[thresh_mask, :]
        pseudo_labels = torch.zeros_like(pos_out).scatter_(1, pos_out.argmax(dim=1)[:, None], 1.)
        ce_loss, ent_loss, softmax_loss, domain_loss = 0.0, 0.0, 0.0, 0.0
        if self.training_config["binary_task"]:
            ce_loss = F.binary_cross_entropy_with_logits(prompt_out, pseudo_labels)
        else:
            ce_loss = F.cross_entropy(prompt_out, pseudo_labels)
        softmax_out = F.softmax(prompt_out, dim=1)
        if self.training_config["w_entropy_loss"] > 0.0:
            ent_loss = entropy_loss(softmax_out).mean()
        if self.training_config["w_softmax_loss"] > 0.0:
            b_softmax = softmax_out.mean(dim=0)
            softmax_loss = -torch.sum(-b_softmax * torch.log(b_softmax + 1e-5))
        if self.training_config["w_domain_loss"] > 0.0:
            domain_loss = self.domain_loss(kwargs["discriminator"], kwargs["optimizer_d"], pos_embed, prompt_embed)
        loss = ce_loss + ent_loss * self.training_config["w_entropy_loss"] + \
        softmax_loss * self.training_config["w_softmax_loss"] + domain_loss * self.training_config["w_domain_loss"]
        return loss

    @torch.no_grad()
    def fix_match_mask(self, output, **kwargs):
        mask = (output.max(dim=1).values > self.cut_off).nonzero().T[0]
        return mask

    @torch.no_grad()
    def flex_match_mask(self, output, idxs_ulb, t_dataset):
        max_probs, max_idxs = output.max(dim=-1)
        mask = (max_probs >= self.cut_off * self.classwise_beta[max_idxs]).int()
        # print(self.classwise_beta[max_idxs])
        select = (max_probs >= self.cut_off).int()
        if select.sum() > 0:
            t_dataset.update_preds(idxs_ulb[select == 1].cpu(), max_idxs[select == 1].cpu())
        
        ds_preds = t_dataset.get_preds().to(self.device)
        pseudo_counter = ds_preds[ds_preds != -1].bincount(minlength = t_dataset.num_gclass)
        max_label = pseudo_counter.max()
        n_samples = len(t_dataset.train_ds)
        if max_label < n_samples:
            self.classwise_beta = pseudo_counter / max(max_label, (ds_preds == -1).sum())
        else:
            self.classwise_beta = pseudo_counter / max_label
        return mask

    def supervised(self, pretrained_model, prompt_model, batch, **kwargs):
        labels = batch.y.to(self.device)
        prompt_batch = prompt_model(batch, self.device)
        prompt_out, _ = pretrained_model(
            prompt_batch,
            decoder = True,
            )
        loss = F.cross_entropy(prompt_out, labels, reduction="mean")
        if self.training_config["r_reg"] > 0.0 and not prompt_model.trans_x:
            loss += self.training_config["r_reg"] * prompt_model.token_embeds.pow(2).mean()
        return loss

    def domain_loss(self, discriminator, optimizer_d, pos_embed, prompt_embed):

        real_labels = torch.ones((prompt_embed.size(0), 1), device = self.device)
        fake_labels = torch.zeros((prompt_embed.size(0), 1), device = self.device)

        optimizer_d.zero_grad()
        real_out = discriminator(pos_embed)
        fake_out = discriminator(prompt_embed.detach())
        real_loss = F.binary_cross_entropy_with_logits(real_out, real_labels)
        fake_loss = F.binary_cross_entropy_with_logits(fake_out, fake_labels)
        disc_loss = (real_loss + fake_loss) / 2
        disc_loss.backward()
        optimizer_d.step()

        prompt_out = discriminator(prompt_embed)
        prompt_loss = F.binary_cross_entropy_with_logits(prompt_out, real_labels)
        return prompt_loss
        
    def train(self, t_dataset, pretrained_model, prompt_model, optimizer, logger, **kwargs) -> None:
        for i, (batch, idxs) in enumerate(t_dataset.train_loader):
            optimizer.zero_grad()
            loss = self.train_func(pretrained_model, prompt_model, batch, idxs=idxs, t_dataset=t_dataset, **kwargs)
            loss.backward()
            optimizer.step()
            total_grad_norm = 0
            # if i % max(1, int((t_dataset.n_train//batch.y.size(0))*0.5)) == 0:
            #     logger.info(f"Train batch: {i}/{np.ceil(t_dataset.n_train//batch.y.size(0))}, Train Loss: {loss.data}")
        return loss