import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from utils import cc
from utils import pad_list
from utils import _seq_mask
from torch.distributions.categorical import Categorical
import random
import os

class pBLSTM(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, subsample, dropout_rate, output_dim):
        super(pBLSTM, self).__init__()
        layers, project_layers = [], []
        for i in range(n_layers):
            idim = input_dim if i == 0 else hidden_dim
            project_dim = hidden_dim * 4 if subsample[i] > 1 else hidden_dim * 2
            out_dim = output_dim if i==(n_layers-1) else hidden_dim
            layers.append(torch.nn.LSTM(idim, hidden_dim, num_layers=1,
                bidirectional=True, batch_first=True))
            project_layers.append(torch.nn.Linear(project_dim, out_dim))
        self.layers = torch.nn.ModuleList(layers)
        self.project_layers = torch.nn.ModuleList(project_layers)
        self.subsample = subsample
        self.dropout_rate = dropout_rate

    def forward(self, xpad, ilens):
        first_out = None
        first_lens = None
        for i, (layer, project_layer) in enumerate(zip(self.layers, self.project_layers)):
            total_length = xpad.size(1)
            xs_pack = pack_padded_sequence(xpad, ilens, batch_first=True)
            layer.flatten_parameters()
            xs, (_, _) = layer(xs_pack)
            ys_pad, ilens = pad_packed_sequence(xs, batch_first=True,
                                                total_length=total_length)
            ys_pad = F.dropout(ys_pad, self.dropout_rate, training=self.training)
            ilens = ilens.numpy()
            sub = self.subsample[i]
            if sub > 1:
                # pad one frame if it's not able to divide into 2 equal length
                if ys_pad.size(1) % 2 == 1:
                    ys_pad = F.pad(ys_pad.transpose(1, 2), (0, 1), mode='replicate').transpose(1, 2)
                # concat two frames
                ys_pad = ys_pad.contiguous().view(ys_pad.size(0), ys_pad.size(1) // 2, ys_pad.size(2) * 2)
                ilens = [(length + 1) // sub for length in ilens]
            projected = project_layer(ys_pad)
            xpad = torch.tanh(projected)
            xpad = F.dropout(xpad, self.dropout_rate, training=self.training)
            if i == 0:
                first_out = xpad
                first_lens = cc(torch.from_numpy(np.array(ilens, dtype=np.int64)))
        ilens = cc(torch.from_numpy(np.array(ilens, dtype=np.int64)))
        return xpad, ilens, first_out, first_lens

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, subsample, dropout_rate, output_dim):
        super(Encoder, self).__init__()
        self.enc2 = pBLSTM(input_dim=input_dim, hidden_dim=hidden_dim, 
                           n_layers=n_layers, subsample=subsample, 
                           dropout_rate=dropout_rate, output_dim=output_dim)

    def forward(self, x, ilens):
        out, ilens, first_out, first_lens = self.enc2(x, ilens)
        return out, ilens, first_out, first_lens

class AttLoc(torch.nn.Module):
    def __init__(self, encoder_dim, decoder_dim, att_dim, conv_channels, conv_kernel_size, att_odim):
        super(AttLoc, self).__init__()
        self.mlp_enc = torch.nn.Linear(encoder_dim, att_dim)
        self.mlp_dec = torch.nn.Linear(decoder_dim, att_dim)
        self.mlp_att = torch.nn.Linear(conv_channels, att_dim)
        self.loc_conv = torch.nn.Conv2d(in_channels=1, out_channels=conv_channels, 
                                        kernel_size=(1, 2 * conv_kernel_size + 1),
                                        stride=1,
                                        padding=(0, conv_kernel_size), bias=False)
        self.gvec = torch.nn.Linear(att_dim, 1, bias=False)
        self.mlp_o = torch.nn.Linear(encoder_dim, att_odim)

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.att_dim = att_dim
        self.att_odim = att_odim
        self.conv_channels = conv_channels

    def forward(self, enc_pad, enc_len, dec_h, att_prev, scaling=2.0):
        '''
        enc_pad:(batch, enc_length, enc_dim)
        enc_len:(batch) of int
        dec_h:(batch, 1, dec_dim)
        att_prev:(batch, enc_length)
        '''
        batch_size = enc_pad.size(0)
        enc_h = self.mlp_enc(enc_pad) # batch_size x enc_length x att_dim

        if dec_h is None:
            dec_h = enc_pad.new_zeros(batch_size, self.decoder_dim)
        else:
            dec_h = dec_h.view(batch_size, self.decoder_dim)

        # initialize attention weights to uniform
        if att_prev is None:
            att_prev = pad_list([enc_pad.new(l).fill_(1.0 / l) for l in enc_len], 0)

        att_conv = self.loc_conv(att_prev.view(batch_size, 1, 1, enc_pad.size(1)))
        att_conv = att_conv.squeeze(2).transpose(1, 2) 
        # att_conv: batch_size x channel x 1 x frame -> batch_size x frame x channel
        att_conv = self.mlp_att(att_conv) # att_conv: batch_size x frame x channel -> batch_size x frame x att_dim

        dec_h_tiled = self.mlp_dec(dec_h).view(batch_size, 1, self.att_dim)
        att_state = torch.tanh(enc_h + dec_h_tiled + att_conv)
        e = self.gvec(att_state).squeeze(2)
        if enc_len is not None:
            mask = []
            for b in range(batch_size):
                mask.append([0]*enc_len[b]+[1]*(enc_pad.size(1)-enc_len[b]))
            mask = cc(torch.ByteTensor(mask))
            e = e.masked_fill_(mask, -1e15)
        attn = F.softmax(scaling * e, dim=1)
        w_expanded = attn.unsqueeze(1) # w_expanded: batch_size x 1 x frame
        
        c = torch.bmm(w_expanded, enc_pad).squeeze(1) 
        # batch x 1 x frame * batch x enc_length x enc_dim => batch x 1 x enc_dim
        c = self.mlp_o(c) # batch x enc_dim
        return c, attn

class Decoder(torch.nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, attention, att_odim, 
                 dropout_rate, bos, eos, pad, ls_weight=0, labeldist=None):
        super(Decoder, self).__init__()
        self.bos, self.eos, self.pad = bos, eos, pad
        self.embedding = torch.nn.Embedding(output_dim, embedding_dim, padding_idx=pad)
        self.LSTMCell = torch.nn.LSTMCell(embedding_dim + att_odim, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, output_dim)
        self.attention = attention

        self.hidden_dim = hidden_dim
        self.att_odim = att_odim
        self.dropout_rate = dropout_rate

        # label smoothing hyperparameters
        self.ls_weight = ls_weight
        self.labeldist = labeldist
        if labeldist is not None:
            self.vlabeldist = cc(torch.from_numpy(np.array(labeldist, dtype=np.float32)))

    def zero_state(self, ori_tensor, dim=None):
        '''
        a util function that new a zero tensor at the same shape of (batch, dim)
        '''
        if not dim:
            return ori_tensor.new_zeros(int(ori_tensor.size(0)), self.hidden_dim)
        else:
            return ori_tensor.new_zeros(int(ori_tensor.size(0)), dim)

    def forward_step(self, emb, dec_h, dec_c, attn, enc_output, enc_len):
        c, attn = self.attention(enc_output, enc_len, dec_h, attn)
        cell_inp = torch.cat([emb, c], dim=-1)
        cell_inp = F.dropout(cell_inp, self.dropout_rate, training=self.training)
        dec_h, dec_c = self.LSTMCell(cell_inp, (dec_h, dec_c))
        output = F.dropout(dec_h, self.dropout_rate, training=self.training)
        logit = self.output_layer(output)
        return logit, dec_h, dec_c, attn

    def forward(self, enc_output, enc_len, dec_input=None, tf_rate=1.0, max_dec_timesteps=500, sample=False):
        batch_size = enc_output.size(0)
        enc_len = enc_len.cpu().numpy().tolist()
        if dec_input is not None:
            pad_dec_input_in = dec_input[0]
            pad_dec_input_out = dec_input[1]
            # get length info
            batch_size, olength = pad_dec_input_out.size(0), pad_dec_input_out.size(1)
            # map idx to embedding
            dec_input_embedded = self.embedding(pad_dec_input_in)

        # initialization
        dec_c = self.zero_state(enc_output)
        dec_h = self.zero_state(enc_output)
        attn = None

        logits, prediction, attns = [], [], []
        # loop for each timestep
        olength = max_dec_timesteps if not dec_input else olength
        for t in range(olength):
            if dec_input is not None:
                # teacher forcing
                tf = True if np.random.random_sample() <= tf_rate else False
                if tf or t==0:
                    emb = dec_input_embedded[:,t,:]
                else:
                    self.embedding(prediction[-1])
            else:
                if t == 0:
                    bos = cc(torch.Tensor([self.bos for _ in range(batch_size)]).type(torch.LongTensor))
                    emb = self.embedding(bos)
                else:
                    emb = self.embedding(prediction[-1])

            logit, dec_h, dec_c, attn = \
                self.forward_step(emb, dec_h, dec_c, attn, enc_output, enc_len)

            attns.append(attn)
            logits.append(logit)
            if not sample:
                prediction.append(torch.argmax(logit, dim=-1))
            else:
                sampled_indices = Categorical(logits=logit).sample() 
                prediction.append(sampled_indices)

        logits = torch.stack(logits, dim=1) # batch x length x output_dim
        log_probs = F.log_softmax(logits, dim=2)
        prediction = torch.stack(prediction, dim=1) # batch x length
        attns = torch.stack(attns, dim=1) # batch x length x enc_len

        # get the log probs of the true label # batch x length
        if dec_input:
            dec_output_log_probs = torch.gather(log_probs, dim=2, index=pad_dec_input_out.unsqueeze(2)).squeeze(2)
        else:
            dec_output_log_probs = torch.gather(log_probs, dim=2, index=prediction.unsqueeze(2)).squeeze(2)

        # label smoothing : q'(y|x) = (1-e)*q(y|x) + e*u(y)
        if self.ls_weight > 0:
            loss_reg = torch.sum(log_probs * self.vlabeldist, dim=2) # u(y)
            dec_output_log_probs = (1 - self.ls_weight) * dec_output_log_probs + self.ls_weight * loss_reg

        return logits, dec_output_log_probs, prediction, attns

class E2E(torch.nn.Module):
    def __init__(self, input_dim, enc_hidden_dim, enc_n_layers, subsample, 
                 enc_output_dim, dropout_rate, dec_hidden_dim, att_dim, 
                 conv_channels, conv_kernel_size, att_odim, embedding_dim, 
                 output_dim, ls_weight, labeldist, 
                 pad=0, bos=1, eos=2):

        super(E2E, self).__init__()

        # encoder to encode acoustic features
        self.encoder = Encoder(input_dim=input_dim, hidden_dim=enc_hidden_dim, 
                               n_layers=enc_n_layers, subsample=subsample, 
                               dropout_rate=dropout_rate, output_dim=enc_output_dim)

        # attention module
        self.attention = AttLoc(encoder_dim=enc_output_dim, 
                                decoder_dim=dec_hidden_dim, 
                                att_dim=att_dim, 
                                conv_channels=conv_channels, 
                                conv_kernel_size=conv_kernel_size, 
                                att_odim=att_odim)

        # decoder 
        self.decoder = Decoder(output_dim=output_dim, 
                               hidden_dim=dec_hidden_dim, 
                               embedding_dim=embedding_dim,
                               attention=self.attention, 
                               dropout_rate=dropout_rate, 
                               att_odim=att_odim, 
                               ls_weight=ls_weight, 
                               labeldist=labeldist, 
                               bos=bos, 
                               eos=eos, 
                               pad=pad)

    def forward(self, data, ilens, true_label=None, tf_rate=1.0, 
                max_dec_timesteps=200, sample=False):
        
        enc_outputs, enc_lens, _, _ = self.encoder(data, ilens)
        logits, log_probs, prediction, attns =\
            self.decoder(enc_outputs, enc_lens, true_label, tf_rate=tf_rate, 
                         max_dec_timesteps=max_dec_timesteps, sample=sample)
        return log_probs, prediction, attns, enc_outputs, enc_lens

    def mask_and_cal_loss(self, log_probs, ys, mask=None):
        # mask is batch x max_len
        # add 1 to EOS
        if mask is None: 
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=log_probs.size(1)))
        else:
            seq_len = [y.size(0) for y in ys]
        # divide by total length
        loss = -torch.sum(log_probs * mask) / sum(seq_len)
        return loss

class disentangle_clean(nn.Module):
    def __init__(self, clean_repre_dim, hidden_dim, nuisance_dim):
        super().__init__()
        self.LSTM = torch.nn.LSTM(clean_repre_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dnn = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, nuisance_dim)
        self.activation = nn.LeakyReLU(0.2)
        
    def forward(self, clean_repre, ilens):
        total_length = clean_repre.size(1)
        clean_pad = pack_padded_sequence(clean_repre, ilens, batch_first=True)
        self.LSTM.flatten_parameters()
        output,_ = self.LSTM(clean_pad) # batch_size x seq_len x hidden_dim
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=total_length)
        output = F.dropout(output, 0.2, training=self.training)
        output = self.dnn(output)
        output = self.activation(output)
        output = F.dropout(output, 0.2, training=self.training)
        output = self.output_layer(output)
        output = F.dropout(output, 0.2, training=self.training)
        output = torch.tanh(output)
        return output

class disentangle_nuisance(nn.Module):
    def __init__(self, nuisance_dim, hidden_dim, clean_repre_dim):
        super().__init__()
        self.LSTM = torch.nn.LSTM(nuisance_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.dnn = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.output_layer = torch.nn.Linear(hidden_dim, clean_repre_dim)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, nuisance_data, ilens):
        total_length = nuisance_data.size(1)
        nuisance_pad = pack_padded_sequence(nuisance_data, ilens, batch_first=True)
        self.LSTM.flatten_parameters()
        output, _ = self.LSTM(nuisance_pad) # batch_size x seq_len x hidden_dim
        output, _ = pad_packed_sequence(output, batch_first=True,
                                        total_length=total_length)
        output = F.dropout(output, 0.2, training=self.training)
        output = self.dnn(output)
        output = self.activation(output)
        output = F.dropout(output, 0.2, training=self.training)
        output = self.output_layer(output)
        output = F.dropout(output, 0.2, training=self.training)
        output = torch.tanh(output)
        return output

class addnoiselayer(nn.Module):
    def __init__(self, dropout_p):
        super().__init__()
        self.dropout_p = dropout_p

    def forward(self, clean_repre):
        output = F.dropout(clean_repre, self.dropout_p, training=self.training)
        return output

class inverse_pBLSTM(nn.Module):
    def __init__(self, enc_hidden_dim, hidden_dim, hidden_project_dim, output_dim, 
                 n_layers, downsample):
        super().__init__()
        layers, project_layers = [], []
        for i in range(n_layers):
            idim = (enc_hidden_dim) if i == 0 else (hidden_project_dim)
            project_dim = hidden_dim if downsample[i] > 1 else hidden_dim*2
            project_to_dim = output_dim if i == (n_layers-1) else hidden_project_dim
            layers.append(nn.LSTM(idim, hidden_dim, num_layers=1, 
                                  bidirectional=True, batch_first=True))
            project_layers.append(nn.Linear(project_dim, project_to_dim))

        self.layers = nn.ModuleList(layers)
        self.project_layers = nn.ModuleList(project_layers)
        self.downsample = downsample
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, enc_output, enc_len):
        enc_len = enc_len.cpu().numpy().tolist()
        for i, (layer, project_layer) in enumerate(zip(self.layers, self.project_layers)):
            total_length = enc_output.size(1)
            xs_pack = pack_padded_sequence(enc_output, enc_len, batch_first=True)
            layer.flatten_parameters()
            xs, (_,_) = layer(xs_pack)
            ys_pad, enc_len = pad_packed_sequence(xs, batch_first=True,
                                                  total_length=total_length)
            enc_len = enc_len.numpy()

            downsub = self.downsample[i]
            if downsub > 1:
                ys_pad = ys_pad.contiguous().view(ys_pad.size(0), ys_pad.size(1)*2, ys_pad.size(2)//2)
                enc_len = [(length*2) for length in enc_len]
            ys_pad = F.dropout(ys_pad, 0.1, training=self.training)
            projected = project_layer(ys_pad)
            enc_output = self.activation(projected)
        output_lens = cc(torch.from_numpy(np.array(enc_len, dtype=np.int64)))
        return enc_output, output_lens
