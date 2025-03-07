import math
import torch
from torch import nn
from torch.nn import init
import torch as T
import torch.nn.functional as F
from models.encoders.S4DWrapper import S4DWrapper
from models.encoders.OrderedMemory import OrderedMemory
from models.encoders.RecurrentGRCX import RecurrentGRCX

class HGRC(nn.Module):
    def __init__(self, config):
        super(HGRC, self).__init__()
        self.config = config
        self.word_dim = config["hidden_size"]
        self.hidden_dim = config["hidden_size"]
        self.model_chunk_size = config["model_chunk_size"]
        self.small_d = 64
        self.chunk_size = 30

        self.RNN = S4DWrapper(config)
        self.initial_transform = nn.Linear(self.hidden_dim, self.hidden_dim)
        if config and "rvnn_norm" in config:
            self.norm = config["rvnn_norm"]
        else:
            self.norm = "layer"

        if self.norm == "batch":
            self.NT = nn.BatchNorm1d(self.hidden_dim)
        elif self.norm == "skip":
            pass
        else:
            self.NT = nn.LayerNorm(self.hidden_dim)

        self.GRC = RecurrentGRCX(config)

    def normalize(self, state):
        if self.norm == "batch":
            return self.NT(state.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        elif self.norm == "skip":
            return state
        else:
            return self.NT(state)


    def forward(self, input, input_mask):

        sequence = self.RNN(input, input_mask)["sequence"]
        osequence = sequence.clone()
        oinput_mask = input_mask.clone()

        sequence = self.normalize(self.initial_transform(sequence))
        N, S, D = sequence.size()
        if not self.config["chunk_mode_inference"] and not self.training:
            self.chunk_size = S
        else:
            self.chunk_size = self.model_chunk_size

        #input 64,110,300; input_mask 64,120; sequence 64,110,300
        #print(f"start sequence = {sequence.shape}")
        while S > 1:
            N, S, D = sequence.size()
            if S >= (self.chunk_size + self.chunk_size // 2):
                if S % self.chunk_size != 0:
                    e = ((S // self.chunk_size) * self.chunk_size) + self.chunk_size - S #e 10; S 110
                    S = S + e #S 120
                    pad = T.zeros(N, e, D).float().to(sequence.device) #pad 64,10,300
                    input_mask = T.cat([input_mask, T.zeros(N, e).float().to(sequence.device)], dim=-1) #input_mask 64,120
                    sequence = T.cat([sequence, pad], dim=-2) #sequence 64,120,300
                    assert sequence.size() == (N, S, D) #N 64;S 120;D 300
                    assert input_mask.size() == (N, S) #N 64;S 120
                S1 = S // self.chunk_size #S1 4
                chunk_size = self.chunk_size #chunk_size 30
            else:
                S1 = 1
                chunk_size = S
            sequence = sequence.view(N, S1, chunk_size, D) #sequence 64,4,30,300
            sequence = sequence.view(N * S1, chunk_size, D) #sequence 256,30,300

            input_mask = input_mask.view(N, S1, chunk_size) #input_mask 64,4,30
            input_mask = input_mask.view(N * S1, chunk_size) #256,30

            N0 = N #N0 64
            N, S, D = sequence.size() # N 256; S 30; D 300
            assert N == N0 * S1
            #print(f"sequence = {sequence.shape}, input_mask = {input_mask.shape}")
            sequence = self.GRC(sequence, input_mask)["global_state"] #sequence 256,300
            assert sequence.size() == (N, D)
            sequence = sequence.view(N0, S1, D) #sequence 64,4,300
            input_mask = input_mask.view(N0, S1, chunk_size)[:, :, 0] #input_mask 64,4
            S = S1 #S 4
            N = N0 #N 64

        assert sequence.size() == (N, 1, D)
        global_state = sequence.squeeze(1)

        return {"sequence": osequence,
                "global_state": global_state,
                "input_mask": oinput_mask,
                "aux_loss": None}
