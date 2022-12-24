# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ run_pretrain.py ]
#   Synopsis     [ scripts for running the pre-training of upstream models ]
#   Author       [ Andy T. Liu (Andi611) ]
#   Copyright    [ Copyleft(c), Speech Lab, NTU, Taiwan ]
"""*********************************************************************************************"""


###############
# IMPORTATION #
###############
import os
import math
import glob
import random
import importlib
from tqdm import tqdm
from collections import defaultdict
#-------------#
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import is_initialized
from tensorboardX import SummaryWriter
import numpy as np
#-------------#
from optimizers import get_optimizer, get_grouped_parameters
from schedulers import get_scheduler
#-------------#
from s3prl import discriminator
from s3prl.utility.helper import get_model_state

class ModelEntry:
    def __init__(self, model, name, trainable, interfaces):
        self.model = model
        self.name = name
        self.trainable = trainable
        self.interfaces = interfaces

##########
# RUNNER #
##########
class Runner():
    """
    Used to handle high-level concepts of a ML experiment
    eg. training loop, evaluation loop, upstream propagation, optimization, tensorboard logging, checkpoint saving
    """
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.logger = SummaryWriter(args.expdir)                                                 

        self.init_ckpt = torch.load(self.args.init_ckpt, map_location='cpu') if self.args.init_ckpt else {}
        self.upstream = self._get_upstream()
        self.discriminator = self._get_discriminator()

        self.lamb = self.config['runner']['lamb']
        self.dis_step = self.config['runner']['dis_step']

        self.devDisLoss = 100000
        self.devAdvLoss = 0

    def _load_weight(self, model, name):
        init_weight = self.init_ckpt.get(name)
        if init_weight:
            show(f'[Runner] - Loading {name} weights from the previous experiment')
            model.load_state_dict(init_weight)

    def _init_model(self, model, name, trainable, interfaces=None):
        for interface in interfaces or []:
            assert hasattr(model, interface), interface

        self._load_weight(model, name)

        if is_initialized() and trainable and any((p.requires_grad for p in model.parameters())):
            model = DDP(model, device_ids=[self.args.local_rank], find_unused_parameters=True)
            for interface in interfaces or []:
                setattr(model, interface, getattr(model.module, interface))

        return ModelEntry(model, name, trainable, interfaces)

    def _get_upstream(self):
        init_upstream = self.init_ckpt.get('Upstream_Config')
        if init_upstream:
            self.args.upstream_config = init_upstream
        module_path = f'pretrain.{self.args.upstream}.pretrain_expert'
        Upstream = getattr(importlib.import_module(module_path), 'UpstreamPretrainExpert')
        upstream = Upstream(self.config['pretrain_expert']['datarc'], 
                            self.args.upstream_config,
                            self.args.device,
                            self.args.multi_gpu).to(self.args.device)

        assert hasattr(upstream, 'device')
        assert hasattr(upstream, 'forward')
        assert hasattr(upstream, 'load_model')
        assert hasattr(upstream, 'add_state_to_save')
        assert hasattr(upstream, 'on_before_zero_grad')
        assert hasattr(upstream, 'get_train_dataloader')

        if self.init_ckpt != {}:
            print('[Runner] - Loading upstream weights from the previous experiment')
            upstream.load_model(self.init_ckpt)
        if hasattr(upstream, 'loss_to_device'):
            print('[Runner] - Loss to device')
            upstream.loss_to_device()
        return upstream

    
    def _get_discriminator(self):
        Discriminator = getattr(discriminator.expert, 'DicriminatorExpert')
        model = Discriminator(
            upstream_dim = self.upstream.output_dim,
            dicriminator_expert = self.config["pretrain_expert"],
            **vars(self.args)
        ).to(self.args.device)

        return self._init_model(
            model = model,
            name = 'Discriminator',
            trainable = True,
            interfaces = []
        )
    

    def _get_optimizer(self, model_params, isUpstream=False):
        optimizer = get_optimizer(
            model_params, 
            self.config['runner']['total_steps'],
            self.config['optimizer'] if isUpstream else self.config['dc_optimizer']
        )

        if self.init_ckpt != {}:
            init_optimizer = self.init_ckpt.get('Optimizer' if isUpstream else 'DC_Optimizer')
            assert init_optimizer
            print('[Runner] - Loading optimizer weights from the previous experiment')
            optimizer.load_state_dict(init_optimizer)
        return optimizer


    def _get_scheduler(self, optimizer):
        scheduler = get_scheduler(
            optimizer,
            self.config['runner']['total_steps'],
            self.config['scheduler']
        )

        if self.init_ckpt != {}:
            init_scheduler = self.init_ckpt.get('Scheduler')
            assert init_scheduler
            print('[Runner] - Loading scheduler weights from the previous experiment')
            scheduler.load_state_dict(init_scheduler)
        return scheduler


    def train(self):
        # set model train mode
        self.upstream.train()

        # prepare data
        # gradient_accumulate_steps = self.config['runner']['gradient_accumulate_steps']
        train_batch_size = self.config['pretrain_expert']['datarc']['train_batch_size']
        #print('[Runner] - Accumulated batch size:', train_batch_size * gradient_accumulate_steps)
        dataloader = self.upstream.get_train_dataloader()
        devloader = self.upstream.get_dev_dataloader()
        # set epoch
        n_epochs = self.config['runner']['n_epochs']
        if n_epochs > 0: 
            # total_steps = int(n_epochs * len(dataloader.dataset) / gradient_accumulate_steps)
            total_steps = int(n_epochs * len(dataloader.dataset))
            print(f'[Runner] - Training for {n_epochs} epochs, which is equivalent to {total_steps} steps')
        else:
            total_steps = self.config['runner']['total_steps']
            # n_epochs = int(total_steps * gradient_accumulate_steps / len(dataloader.dataset))
            n_epochs = int(total_steps / len(dataloader.dataset))
            print(f'[Runner] - Training for {total_steps} steps, which is approximately {n_epochs} epochs')

        assert total_steps > self.config['runner']['log_step']
        assert total_steps > self.config['runner']['save_step']

        # set amp
        amp = self.config['runner'].get('fp16', False)
        if amp:
            print('[Runner] - Enabled fp16 training')
            scaler = torch.cuda.amp.GradScaler()

        # set optimizer
        model_params = [self.upstream.model]
        optimizer = self._get_optimizer(model_params, isUpstream=True)
        dc_optimizer = self._get_optimizer([self.discriminator.model])

        # set scheduler
        scheduler = None
        if self.config.get('scheduler'):
            scheduler = self._get_scheduler(optimizer)

        # set progress bar
        pbar = tqdm(total=total_steps, dynamic_ncols=True, desc='overall')
        init_step = self.init_ckpt.get('Step')
        if init_step:
            pbar.n = init_step

        all_step1_dc_loss = 0
        all_step2_dc_loss = 0
        all_loss = 0
        step1_dc_records = defaultdict(list)
        upstream_records = defaultdict(list)
        dev_records = defaultdict(list)
        step2_dc_records = defaultdict(list)
        prefix = f'{self.args.upstream}/train-'

        while pbar.n < pbar.total:
            for teacher_data, student_data in tqdm(dataloader, dynamic_ncols=True, desc='train'):
                # try/except block for forward/backward
                
                if pbar.n >= pbar.total:
                    break
                global_step = pbar.n + 1

                with torch.cuda.amp.autocast(enabled=amp):
                    dis_loss, upstream_records = self.upstream(
                        teacher_data,
                        student_data,
                        records=upstream_records,
                        global_step=global_step,
                        log_step=self.config['runner']['log_step'],
                        return_other=True
                    )
                
                student_features = upstream_records["student_features"]
                teacher_features = upstream_records["teacher_features"]
                if global_step % self.dis_step == 0 or pbar.n == pbar.total -1:
                    try:    
                        # Step 1 : train discriminator
                        dc_loss = self.discriminator.model(
                            student_features, 
                            teacher_features,
                            step1_dc_records,
                            detach=True
                        )

                        if self.args.multi_gpu:
                            dc_loss = dc_loss.sum()
                        if amp:
                            scaler.scale(dc_loss).backward()
                        else:
                            dc_loss.backward()

                    except RuntimeError as e:
                        if 'CUDA out of memory' in str(e):
                            print(f'[Runner] - CUDA out of memory at step {global_step}')
                            torch.cuda.empty_cache()
                            dc_optimizer.zero_grad()
                            continue
                        else:
                            raise

                    # record loss
                    all_step1_dc_loss += dc_loss.item()
                    del dc_loss
                        
                    # unscale
                    if amp:
                        scaler.unscale_(dc_optimizer)

                    # gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.model.parameters(), self.config['runner']['gradient_clipping'])
                    if math.isnan(grad_norm):
                        print(f'[Runner] - Error : grad norm is NaN at global step {global_step}')

                    # optimize
                    if amp:
                        scaler.step(dc_optimizer)
                        scaler.update()
                    elif not math.isnan(grad_norm):
                        dc_optimizer.step()
                    
                    dc_optimizer.zero_grad()

                # Step 2: train distillation
                try:
                    step2_dc_loss = self.discriminator.model(
                        student_features, 
                        teacher_features,
                        step2_dc_records,
                        detach=False
                    )
                    loss = dis_loss - self.lamb*step2_dc_loss

                    if self.args.multi_gpu:
                        loss = loss.sum()
                    if amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print(f'[Runner] - CUDA out of memory at step {global_step}')
                        torch.cuda.empty_cache()
                        optimizer.zero_grad()
                        continue
                    else:
                        raise
                
                all_step2_dc_loss += step2_dc_loss.item()
                all_loss += loss
                del step2_dc_loss
                del loss

                # unscale
                if amp:
                    scaler.unscale_(optimizer)

                # gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.upstream.model.parameters(), self.config['runner']['gradient_clipping'])
                if math.isnan(grad_norm):
                    print(f'[Runner] - Error : grad norm is NaN at global step {global_step}')

                # optimize
                if amp:
                    scaler.step(optimizer)
                    scaler.update()
                elif not math.isnan(grad_norm):
                    optimizer.step()

                self.upstream.on_before_zero_grad()
                optimizer.zero_grad()
                dc_optimizer.zero_grad()

                # adjust learning rate
                if scheduler:
                    scheduler.step()


                # logging
                if global_step % self.config['runner']['log_step'] == 0 or pbar.n == pbar.total -1:
                    # log step 1 dc loss
                    self.logger.add_scalar(f'{prefix}step1 dc loss', all_step1_dc_loss, global_step=global_step)
                    # log distil loss
                    self.logger.add_scalar(f'{prefix}loss', all_loss, global_step=global_step)
                    # log step 1 dc loss
                    self.logger.add_scalar(f'{prefix}step2 dc loss', all_step2_dc_loss, global_step=global_step)

                    # log lr
                    if hasattr(optimizer, 'get_lr'):
                        self.logger.add_scalar(f'{prefix}lr', optimizer.get_lr()[0], global_step=global_step)
                    else:
                        self.logger.add_scalar(f'{prefix}lr', self.config['optimizer']['lr'], global_step=global_step)
                    # log norm
                    self.logger.add_scalar(f'{prefix}gradient-norm', grad_norm, global_step=global_step)

                    # log customized contents
                    self.discriminator.model.log_records(
                        'dicriminator step1', 
                        records=step1_dc_records,
                        logger=self.logger,
                        prefix=prefix,
                        global_step=global_step,
                    )
                    self.upstream.log_records(
                        records=upstream_records,
                        logger=self.logger,
                        prefix=prefix,
                        global_step=global_step,
                    )
                    self.discriminator.model.log_records(
                        'dicriminator step2', 
                        records=step2_dc_records,
                        logger=self.logger,
                        prefix=prefix,
                        global_step=global_step,
                    )
                    step1_dc_records = defaultdict(list)
                    upstream_records = defaultdict(list)
                    step2_dc_records = defaultdict(list)

                if global_step % self.config['runner']['save_step'] == 0 or pbar.n == pbar.total -1:
                    def check_ckpt_num(directory):
                        max_keep = self.config['runner']['max_keep']
                        ckpt_pths = glob.glob(f'{directory}/states-*.ckpt')
                        if len(ckpt_pths) >= max_keep:
                            ckpt_pths = sorted(ckpt_pths, key=lambda pth: int(pth.split('-')[-1].split('.')[0]))
                            for ckpt_pth in ckpt_pths[:len(ckpt_pths) - max_keep + 1]:
                                os.remove(ckpt_pth)
                    check_ckpt_num(self.args.expdir)

                    all_states = {
                        'Optimizer': optimizer.state_dict(),
                        'DC_Optimizer': dc_optimizer.state_dict(),
                        'Step': pbar.n,
                        'Args': self.args,
                        'Config': self.config,
                    }
                    all_states[self.discriminator.name] = get_model_state(self.discriminator.model)
                    all_states = self.upstream.add_state_to_save(all_states)
                    

                    if scheduler:
                        all_states['Scheduler'] = scheduler.state_dict()
                    
                    name = f'states-epoch-{n_epochs}.ckpt' if pbar.n == pbar.total -1 and n_epochs > 0 else \
                           f'states-{global_step}.ckpt'
                    save_path = os.path.join(self.args.expdir, name)
                    tqdm.write(f'[Runner] - Save the checkpoint to: {save_path}')
                    torch.save(all_states, save_path)
                
                if global_step % self.config['runner']['eval_step'] == 0:
                    with torch.no_grad():
                        self.upstream.eval()
                        self.discriminator.model.eval()
                        avg_dis = 0
                        avg_adv = 0
                        for teacher_data, student_data in tqdm(devloader, dynamic_ncols=True, desc='dev'):
                            dev_dis, dev_records = self.upstream(
                                teacher_data,
                                student_data,
                                records=dev_records,
                                global_step=-1,
                                log_step=self.config['runner']['log_step'],
                                return_other=True
                            )
                            student_features = dev_records["student_features"]
                            teacher_features = dev_records["teacher_features"]
                            dev_adv = self.discriminator.model(
                                student_features, 
                                teacher_features,
                                step2_dc_records,
                                detach=False
                            )
                            avg_dis += dev_dis.item()
                            avg_adv += dev_adv.item()
                    avg_dis /= len(devloader.dataset)
                    avg_adv /= len(devloader.dataset)
                    self.upstream.train()
                    self.discriminator.model.train()

                    self.logger.add_scalar(f'{self.args.upstream}/dev-dis-loss', avg_dis, global_step=global_step)
                    self.logger.add_scalar(f'{self.args.upstream}/dev-adv-loss', avg_adv, global_step=global_step)
                    tqdm.write(f"[Runner] - Distil loss: {avg_dis}")
                    tqdm.write(f"[Runner] - Discriminator loss: {avg_adv}")
                    if avg_dis < self.devDisLoss:
                        self.devDisLoss = avg_dis
                        all_states = {
                            'Optimizer': optimizer.state_dict(),
                            'Step': pbar.n,
                            'Args': self.args,
                            'Config': self.config,
                        }
                        all_states = self.upstream.add_state_to_save(all_states)

                        if scheduler:
                            all_states['Scheduler'] = scheduler.state_dict()
                        
                        name = 'dev-dis-best.ckpt'
                        save_path = os.path.join(self.args.expdir, name)
                        tqdm.write(f'[Runner] - Better dis score, save the checkpoint to: {save_path}')
                        torch.save(all_states, save_path)
                    if self.config["runner"]["save_adv"] and avg_adv > self.devAdvLoss:
                        self.devAdvLoss = avg_adv
                        all_states = {
                            'Optimizer': optimizer.state_dict(),
                            'Step': pbar.n,
                            'Args': self.args,
                            'Config': self.config,
                        }
                        all_states = self.upstream.add_state_to_save(all_states)

                        if scheduler:
                            all_states['Scheduler'] = scheduler.state_dict()
                        
                        name = 'dev-adv-best.ckpt'
                        save_path = os.path.join(self.args.expdir, name)
                        tqdm.write(f'[Runner] - Better adv score, save the checkpoint to: {save_path}')
                        torch.save(all_states, save_path)
                all_loss = 0
                all_step1_dc_loss = 0
                all_step2_dc_loss = 0      
                pbar.update(1)

        pbar.close()
