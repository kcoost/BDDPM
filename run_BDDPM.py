# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from pretty_midi import PrettyMIDI
import pypianoroll
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import copy
import glob
import numpy as np
import json
import random
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from functools import partial
from PIL import Image

from pytorch_lightning.core.lightning import LightningModule
import pytorch_lightning as pl
import wandb


class AdapterLayer(nn.Module):
    def __init__(self, input_size, reduction_factor):
        super(AdapterLayer, self).__init__()
        self.adapter = nn.Sequential(nn.Linear(input_size, input_size//reduction_factor),
                                     nn.ReLU(),
                                     nn.Linear(input_size//reduction_factor, input_size))
        self.adapter.apply(self.init_weights)

    def init_weights(self, m, std = 1e-2):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, std = std)
            torch.nn.init.normal_(m.bias, std = std)
            m.weight.data = torch.clamp(m.weight.data, min = -2*std, max = 2*std)
            m.bias.data = torch.clamp(m.bias.data, min = -2*std, max = 2*std)
    
    def forward(self, X):
        return self.adapter(X) + X


class ParamGenerator(nn.Module):
    def __init__(self, args):
        super(ParamGenerator, self).__init__()
        self.T = args.T
        GPT = AutoModelForCausalLM.from_pretrained('gpt2')#distil
        self.layers = GPT.transformer.h
        self.wpe = GPT.transformer.wpe

        hidden_size = GPT.config.hidden_size
        self.W_in = nn.Linear(args.midi_size, hidden_size)
        self.W_out = nn.Linear(hidden_size, args.midi_size)

        self.proj1 = nn.Sequential(nn.Linear(2*self.T + 2, self.T), nn.ReLU(),
                                   nn.Linear(self.T, self.T), nn.ReLU(),
                                   nn.Linear(self.T, hidden_size))
    def t_embedding(self, t):
        freq = torch.arange(self.T+1).to(t.device)/(self.T+1)
        ts = t[:, None].repeat(1, self.T+1)*freq[None, :].repeat(t.shape[0], 1)
        t = torch.cat((torch.sin(ts), torch.cos(ts)), 1)
        return t # batch_size, 128
    
    def forward(self, x, t):
        batch_size, seq_len, _ = x.shape
        t_proj = self.t_embedding(t)
        t_proj = self.proj1(t_proj).unsqueeze(1).repeat(1, seq_len, 1)

        inputs_embeds = self.W_in(x)

        #attention_mask = torch.tensor([seq_len*[1] + (max(seq_lens) - seq_len)*[0] for seq_len in seq_lens])
        #position_ids = attention_mask.long().cumsum(-1) - 1
        #position_ids.masked_fill_(attention_mask == 0, 1)
        #position_embeds = self.wpe(position_ids)

        position_embeds = self.wpe(torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(t_proj.device))
        hidden_states = inputs_embeds + position_embeds + t_proj

        for layer in self.layers:
            hidden_states = layer(hidden_states + t_proj)[0]
        
        hidden_states = self.W_out(hidden_states)
        return nn.Sigmoid()(hidden_states)


class ParamGenerator(nn.Module):
    def __init__(self, args):
        super(ParamGenerator, self).__init__()
        self.T = args.T
        GPT = AutoModelForCausalLM.from_pretrained('distilgpt2')#
        self.layers = GPT.transformer.h
        self.wpe = GPT.transformer.wpe
        hidden_size = GPT.config.hidden_size
        reduction_factor = 12

        n_layers = len(self.layers)
        self.adapters = nn.ModuleList([AdapterLayer(hidden_size, reduction_factor) for n in range(n_layers)])

        self.W_in1 = nn.Linear(args.midi_size, hidden_size)
        self.W_in2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.W_out = nn.Linear(hidden_size, args.midi_size)

        self.proj1 = nn.Sequential(nn.Linear(2*self.T + 2, self.T), nn.ReLU(),
                                   nn.Linear(self.T, self.T//2), nn.ReLU())
        self.poss = nn.ModuleList([nn.Linear(self.T//2, hidden_size) for n in range(n_layers)])

    def t_embedding(self, t):
        freq = torch.arange(self.T+1).to(t.device)/(self.T+1)
        ts = t[:, None].repeat(1, self.T+1)*freq[None, :].repeat(t.shape[0], 1)
        ts = torch.cat((torch.sin(ts), torch.cos(ts)), 1)
        return ts # self.T//2, 128
    
    def forward(self, x, t):
        batch_size, seq_len, _ = x.shape
        t_proj = self.t_embedding(t)
        t_proj = self.proj1(t_proj).unsqueeze(1).repeat(1, seq_len, 1)

        x = self.W_in1(x)
        inputs_embeds = torch.transpose(self.W_in2(torch.transpose(nn.ReLU()(x), 1, 2)), 1, 2) + x

        #attention_mask = torch.tensor([seq_len*[1] + (max(seq_lens) - seq_len)*[0] for seq_len in seq_lens])
        #position_ids = attention_mask.long().cumsum(-1) - 1
        #position_ids.masked_fill_(attention_mask == 0, 1)
        #position_embeds = self.wpe(position_ids)

        position_embeds = self.wpe(torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(t_proj.device))
        hidden_states = inputs_embeds + position_embeds# + self.pos[0](t_proj)

        for pos, adapter, layer in zip(self.poss, self.adapters, self.layers):
            hidden_states = hidden_states + pos(t_proj)
            hidden_states = adapter(hidden_states)
            hidden_states = layer(hidden_states)[0]
        
        hidden_states = self.W_out(hidden_states)
        return nn.Sigmoid()(hidden_states)



from torch.utils.data import Dataset, DataLoader
class MidiDataset(Dataset):
    """
    Heart beat: every ten steps a row with ones
    """
    def __init__(self, args, heart_beat = False):
        self.heart_beat = heart_beat
        self.min_seq_len = args.min_seq_len
        self.max_seq_len = args.max_seq_len
        if not self.heart_beat:
            self.file_names = glob.glob(os.path.join(args.tensor_path, '*.npy'))
        else:
            self.midi_size = args.midi_size
            self.batch_size = args.batch_size
            self.freq = 3

    def __len__(self):
        if self.heart_beat:
            return 100*self.batch_size
        else:
            return len(self.file_names)

    def __getitem__(self, idx):
        if self.heart_beat:
            seq_len = self.min_seq_len #random.randint(self.min_seq_len, self.max_seq_len)
            data = torch.cat((torch.zeros(self.freq-1, self.midi_size),
                              torch.ones(1, self.midi_size)), 0)
            data = data.repeat(seq_len//self.freq + 1, 1)
            start_idx = random.randint(0, data.shape[0] - seq_len)
            data = data[start_idx:start_idx + seq_len]
            return data
        else:
            pianoroll = np.load(self.file_names[idx])
            return pianoroll

def collate_fn(batch, min_seq_len, max_seq_len):
    lens = [pianoroll.shape[0] for pianoroll in batch]
    max_seq_len = min(max_seq_len, min(lens))
    min_seq_len = min(min_seq_len, max_seq_len)
    if min_seq_len == max_seq_len:
        seq_len = max_seq_len
    else:
        seq_len = random.randint(min_seq_len, max_seq_len) # yes all samples have same length

    pianorolls = []
    for pianoroll in batch:
        start_idx = 0 #random.randint(0, pianoroll.shape[0] - seq_len)
        pianoroll_cut = pianoroll[start_idx:start_idx + seq_len]
        pianoroll_cut = np.unpackbits(pianoroll_cut, 1)
        pianorolls.append(torch.tensor(pianoroll_cut, dtype=torch.float32).clone().detach().unsqueeze(0))
    return torch.cat(pianorolls, dim = 0)

class MidiDataModule(pl.LightningDataModule):
    def __init__(self, args, heart_beat = False):
        super().__init__()
        self.heart_beat = heart_beat
        self.args = args
        self.min_seq_len = args.min_seq_len
        self.max_seq_len = args.max_seq_len
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_data = MidiDataset(self.args)

    def train_dataloader(self):
        if self.heart_beat:
            return DataLoader(self.train_data, batch_size=self.batch_size,
                              shuffle=True, num_workers=self.num_workers, pin_memory=True)
        else:
            return DataLoader(self.train_data, batch_size=self.batch_size,
                              collate_fn=partial(collate_fn, min_seq_len=self.min_seq_len, max_seq_len=self.max_seq_len),
                              shuffle=True, num_workers=self.num_workers, pin_memory=True)


class DDPM(LightningModule):
    def __init__(self, args):
        super().__init__()
        # load model
        self.model = ParamGenerator(args)
        self.rho = args.rho
        for _, param in self.model.layers.named_parameters():
            param.requires_grad = False
        for _, param in self.model.wpe.named_parameters():
            param.requires_grad = False

        self.midi_size = args.midi_size
        self.max_seq_len = args.max_seq_len
        self.n_val = args.n_val
        self.val_seq_len = args.val_seq_len

        # forward distribution params
        self.T = args.T
        if isinstance(args.beta, list):
            assert len(args.beta) == self.T
            self.beta = torch.tensor([-1] + args.beta) # start index at 1 to be consistent with papers
        else:
            self.beta = torch.tensor([-1] + self.T*[args.beta])
        #self.beta = 1/(self.T - torch.arange(self.T + 1) + 1)
        self.alpha = -torch.ones(self.T + 1)
        for t in range(1, self.T + 1):
            self.alpha[t] = 1 - torch.prod(1 - self.beta[1:t+1])
        
        self.L00 = -torch.ones(self.T + 1)
        self.L01 = -torch.ones(self.T + 1)
        self.L10 = -torch.ones(self.T + 1)
        self.L11 = -torch.ones(self.T + 1)
        for t in range(2, self.T + 1): # gamma[1] and delta[1] can't exist
            self.L00[t] = (1-self.rho)*self.beta[t]*self.rho*self.alpha[t-1]/(1-self.rho*self.alpha[t])
            self.L01[t] = (1-(1-self.rho)*self.beta[t])*self.alpha[t-1]/self.alpha[t]
            self.L10[t] = self.beta[t]*(1-(1-self.rho)*self.alpha[t-1])/self.alpha[t]
            self.L11[t] = (1-(1-self.rho)*self.beta[t])*(1-(1-self.rho)*self.alpha[t-1])/(1-(1-self.rho)*self.alpha[t])

        # optimizer params
        self.lr_init = args.lr_init
        self.lr_min = args.lr_min
        self.n_epochs = args.n_epochs
        self.scheduler_period = args.scheduler_period

        self.save_hyperparameters()
        self.logged = False

    def store_params(self):
        # log
        for i, (alpha, beta) in enumerate(zip(self.alpha[1:], self.beta[1:])):
            self.logger.experiment.log({'Alpha':alpha})#, step=i)
            self.logger.experiment.log({'Beta':beta})#, step=i)

        for i, (gamma, delta) in enumerate(zip(self.gamma[2:], self.delta[2:])):
            self.logger.experiment.log({'Constant':gamma})#, step=i, on_step = False)
            self.logger.experiment.log({'x0 weight':-gamma + delta})#, step=i)
            self.logger.experiment.log({'xt weight':1 - gamma - delta})#, step=i)

    def get_xt(self, x0, t):
        alpha = self.alpha.to(self.device).index_select(0, t)[:, None, None]
        return torch.bernoulli(x0*(1-alpha) + self.rho*alpha)

    def forward(self, batch, batch_id):
        xt, t = batch
        ft = self.model(xt, t.to(self.device))
        return ft
    
    def loss_fn(self, ft, x0, xt, t):
        l00 = self.L00.to(self.device).index_select(0, t)[:, None, None]
        l01 = self.L01.to(self.device).index_select(0, t)[:, None, None]
        l10 = self.L10.to(self.device).index_select(0, t)[:, None, None]
        l11 = self.L11.to(self.device).index_select(0, t)[:, None, None]
        w = l00 + x0*(l10-l00) + xt*(l01-l00) + x0*xt*(l00+l11-l01-l10)
        return self.T*(-(w*torch.log(ft + 1e-12) + (1-w)*torch.log(1 - ft + 1e-12)).mean())

    def turn_on_grads(self, part = ''):
        for name, param in self.model.layers.named_parameters():
            if part in name:
                param.requires_grad = True
        for _, param in self.model.wpe.named_parameters():
            if part in name:
                param.requires_grad = True

    def training_step(self, batch, batch_id):
        #if not self.logged:
        #    self.store_params()
        #    self.logged = True
        if batch_id == 4000: self.turn_on_grads('mlp')
        if batch_id == 7000: self.turn_on_grads()
        # 1. Sample from data
        x0 = batch
        # 2. Sample random t
        batch_size = batch.shape[0]
        t = torch.randint(low = 2, high = self.T + 1, size = (batch_size,)).to(self.device)
        # 3. Sample xt
        xt = self.get_xt(x0, t)
        # 4. Generated beta
        ft = self((xt, t), None)
        # 5. Calculate loss
        loss = self.loss_fn(ft, x0, xt, t)
        # (6. Log loss)
        #self.logger.experiment.log({"train/loss": loss})
        self.log("train/loss", loss, on_epoch=False, on_step=True)
        return {'loss': loss}

    def generate(self):
        self.model = self.model.eval()
        with torch.no_grad():
            # 1. Sample XT ~ B(rho)
            xt = torch.bernoulli(self.rho*torch.ones((1, self.val_seq_len, self.midi_size), device = self.device))
            for t in range(self.T, 1, -1):
                #self.logger.experiment.log({'Mean': xt.mean(), 'epoch': self.current_epoch})
                # 2. Sample x_t-1
                ft = self.forward((xt, torch.tensor([t], device = self.device)), None)
                xtmin1 = torch.bernoulli(ft).to(self.device)
                # 3. Repeat
                xt = xtmin1
            x0 = torch.bernoulli(xt*(1-self.beta[1].to(self.device)) + self.rho*self.beta[1].to(self.device))

        self.model = self.model.train()
        return x0[0]

    def on_epoch_end(self):
        if self.current_epoch%3 == 0:
            x0 = self.generate()
            x_bt = pypianoroll.BinaryTrack(name = 'Combined', program=0, is_drum=False, pianoroll=np.array(x0.cpu().detach()))
            x_mt = pypianoroll.Multitrack(tempo = 120*np.ones(shape=(self.val_seq_len, 1), dtype=np.float64))
            x_mt.append(x_bt)
            x_pm = pypianoroll.to_pretty_midi(x_mt)
            waveform = x_pm.synthesize()
            self.logger.experiment.log({"Generated Songs": wandb.Audio(waveform, caption=f"Song, epoch {self.current_epoch}", sample_rate=44100)})
            self.logger.experiment.log({"Generated Midi": wandb.Image(x0.T.unsqueeze(0).cpu().float(), caption=f"Midi, epoch {self.current_epoch}")})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr_init)
        n_batches = len(self.train_dataloader()) // self.trainer.accumulate_grad_batches
        # second arg = how many step's (don't understand its behaviour for 'epoch') it takes to reach lr_min
        scheduler = {'scheduler': CosineAnnealingLR(optimizer, n_batches*self.n_epochs//self.scheduler_period, self.lr_min),
                     'interval': 'step', 'frequency': self.scheduler_period}
        # retardedly enough, THE HIGHER THE FREQUENCY THE LESS OFTEN IT STEPS
        # every 'frequency' steps it takes one scheduler.step and it takes 'second arg of 'scheduler'' to reach lr_min
        return [optimizer], [scheduler]

if __name__ == '__main__':
    parser = ArgumentParser()
    # model args
    parser.add_argument('--midi_size', type = int, default = 128, help='N')

    # hyper params
    parser.add_argument('--T', type = int, default = 1000, help='N')
    parser.add_argument('--beta', type = float, default = 0.005, help='N')
    parser.add_argument('--rho', type = float, default = 0.024, help='N')
    parser.add_argument('--n_val', type = int, default = 400, help='N')
    parser.add_argument('--val_seq_len', type = int, default = 512, help='N')

    # data loader args
    parser.add_argument('--tensor_path', type = str, default = '/home/aleph/repos/koenBDDPM/data', help='N')
    parser.add_argument('--min_seq_len', type = int, default = 512, help='N')
    parser.add_argument('--max_seq_len', type = int, default = 512, help='N')
    parser.add_argument('--batch_size', type=int, default = 32, help = 'Batch size training')
    parser.add_argument('--num_workers', type=int, default = 8, help = 'Number of workers')

    # trainer args
    parser.add_argument('--precision', type=int, default = 32, help = 'Bit precision')

    # optimizer args
    parser.add_argument('--lr_init', type = float, default = 1e-3, help='Initial learning rate')
    parser.add_argument('--lr_min', type = float, default = 1e-4, help='Final learning rate')
    parser.add_argument('--n_epochs', type = int, default = 100, help='Number of training epochs')
    parser.add_argument('--scheduler_period', type = int, default = 20, help='Frequency at which the learning rate gets updated')

    args = parser.parse_args()

    #wandb.init()

    wandb_logger = WandbLogger(project='BDDPM')

    Midi = MidiDataModule(args)
    model = DDPM(args)
    trainer = pl.Trainer(logger = wandb_logger, gpus = '0', max_epochs=args.n_epochs,
                        progress_bar_refresh_rate=100,
                        precision=args.precision, gradient_clip_val=0.5, accumulate_grad_batches=8,
                        log_every_n_steps=1)

    trainer.fit(model, Midi)