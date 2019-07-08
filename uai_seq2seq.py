import torch 
import torch.nn.functional as F
import numpy as np
from model import Encoder, AttLoc, Decoder, disentangle_clean, disentangle_nuisance, addnoiselayer, inverse_pBLSTM
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
from utils import _seq_mask
import copy
import yaml
import os
import pickle

class UAI_seq2seq(object):
    def __init__(self, config, beta, gamma, delta, load_model=False):
        self.config = config
        print(self.config)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.delta = float(delta)
        # logger
        self.logger = Logger(config['logdir']+'_b'+str(beta)+'g'+str(gamma)+'noise'+str(delta))
        # load vocab and non lang syms
        self.load_vocab()
        # get data loader
        self.get_data_loaders()
        # get label distribution
        self.get_label_dist(self.train_lab_dataset)
        # build model and optimizer
        self.build_model(load_model=load_model)

    def save_model(self, model_path):
        model_path = model_path+'_b'+str(self.beta)+'g'+str(self.gamma)+'noise'+str(self.delta)
        encoder_path = model_path+'_encoder'
        encoder2_path = model_path+'_encoder2'
        decoder_path = model_path+'_decoder'
        att_path = model_path+'_att'
        disen_clean_path = model_path+'_disen_clean'
        disen_nuisance_path = model_path+'_disen_nuisance'
        recons_path = model_path+'_recons'
        opt1_path = model_path+'_opt1'
        opt2_path = model_path+'_opt2'
        torch.save(self.encoder.state_dict(), f'{encoder_path}.ckpt')
        torch.save(self.encoder2.state_dict(), f'{encoder2_path}.ckpt')
        torch.save(self.decoder.state_dict(), f'{decoder_path}.ckpt')
        torch.save(self.attention.state_dict(), f'{att_path}.ckpt')
        torch.save(self.disen_clean.state_dict(), f'{disen_clean_path}.ckpt')
        torch.save(self.disen_nuisance.state_dict(), f'{disen_nuisance_path}.ckpt')
        torch.save(self.reconstructor.state_dict(), f'{recons_path}.ckpt')
        torch.save(self.optimizer_m1.state_dict(), f'{opt1_path}.opt')
        torch.save(self.optimizer_m2.state_dict(), f'{opt2_path}.opt')
        return

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f) # a dict;character to index
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f)
        return

    def load_model(self, model_path, load_optimizer):
        model_path = model_path+'_b'+str(self.beta)+'g'+str(self.gamma)+'noise'+str(self.delta)
        if os.path.exists(model_path+'_encoder.ckpt'):
            print(f'Load model from {model_path}')
            encoder_path = model_path+'_encoder'
            encoder2_path = model_path+'_encoder2'
            decoder_path = model_path+'_decoder'
            att_path = model_path+'_att'
            disen_clean_path = model_path+'_disen_clean'
            disen_nuisance_path = model_path+'_disen_nuisance'
            recons_path = model_path+'_recons'
            opt1_path = model_path+'_opt1'
            opt2_path = model_path+'_opt2'
            self.encoder.load_state_dict(torch.load(f'{encoder_path}.ckpt'))
            self.encoder2.load_state_dict(torch.load(f'{encoder2_path}.ckpt'))
            self.decoder.load_state_dict(torch.load(f'{decoder_path}.ckpt'))
            self.attention.load_state_dict(torch.load(f'{att_path}.ckpt'))
            self.disen_clean.load_state_dict(torch.load(f'{disen_clean_path}.ckpt'))
            self.disen_nuisance.load_state_dict(torch.load(f'{disen_nuisance_path}.ckpt'))
            self.reconstructor.load_state_dict(torch.load(f'{recons_path}.ckpt'))
            if load_optimizer:
                print(f'Load optmizer from {model_path}')
                self.optimizer_m1.load_state_dict(torch.load(f'{opt1_path}.opt'))
                self.optimizer_m2.load_state_dict(torch.load(f'{opt2_path}.opt'))
                adjust_learning_rate(self.optimizer_m1, 
                                     self.config['retrieve_learning_rate1'])
                adjust_learning_rate(self.optimizer_m2, 
                                     self.config['retrieve_learning_rate2'])
        return

    def get_data_loaders(self):
        root_dir = self.config['dataset_root_dir']
        # get labeled dataset
        labeled_set = self.config['labeled_set']
        self.train_lab_dataset = PickleDataset(os.path.join(root_dir, f'{labeled_set}.p'), 
            config=self.config, sort=True)
        self.train_lab_loader = get_data_loader(self.train_lab_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=self.config['shuffle'])

        # get dev dataset
        dev_set = self.config['dev_set']
        # do not sort dev set
        self.dev_dataset = PickleDataset(os.path.join(root_dir, f'{dev_set}.p'), sort=True)
        self.dev_loader = get_data_loader(self.dev_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False)
        return

    def get_label_dist(self, dataset):
        labelcount = np.zeros(len(self.vocab))
        for _, y, _, _ ,_ in dataset:
            for ind in y:
                labelcount[ind] += 1.
        labelcount[self.vocab['<EOS>']] += len(dataset)
        labelcount[self.vocab['<PAD>']] = 0
        labelcount[self.vocab['<BOS>']] = 0
        self.labeldist = labelcount / np.sum(labelcount)
        return

    def build_model(self, load_model=False):
        labeldist = self.labeldist
        ls_weight = self.config['ls_weight']
        self.enc_output_dim = self.config['enc_output_dim']
        self.enc_hidden_dim = self.config['enc_hidden_dim']
        self.encoder = cc_model(Encoder(input_dim=self.config['input_dim'],
                                        hidden_dim=self.enc_hidden_dim,
                                        n_layers=self.config['enc_n_layers'],
                                        subsample=self.config['subsample'],
                                        dropout_rate=self.config['dropout_rate'],
                                        output_dim=self.config['enc_output_dim']))
        self.encoder2 = cc_model(Encoder(input_dim=self.config['input_dim'],
                                         hidden_dim=self.enc_hidden_dim,
                                         n_layers=self.config['enc_n_layers'],
                                         subsample=self.config['subsample'],
                                         dropout_rate=self.config['dropout_rate'],
                                         output_dim=self.config['enc_output_dim']))
        print(self.encoder)
        print(self.encoder2)
        self.attention =cc(AttLoc(encoder_dim=self.enc_output_dim,
                                  decoder_dim=self.config['dec_hidden_dim'],
                                  att_dim=self.config['att_dim'],
                                  conv_channels=self.config['conv_channels'],
                                  conv_kernel_size=self.config['conv_kernel_size'],
                                  att_odim=self.config['att_odim']))
        print(self.attention)
        self.decoder = cc(Decoder(output_dim=len(self.vocab),
                                  hidden_dim=self.config['dec_hidden_dim'],
                                  embedding_dim=self.config['embedding_dim'],
                                  attention=self.attention,
                                  dropout_rate=self.config['dropout_rate'],
                                  att_odim=self.config['att_odim'],
                                  ls_weight=ls_weight,
                                  labeldist=labeldist,
                                  bos=self.vocab['<BOS>'],
                                  eos=self.vocab['<EOS>'],
                                  pad=self.vocab['<PAD>']))
        print(self.decoder)
        self.disen_clean = cc_model(disentangle_clean(clean_repre_dim=self.enc_output_dim,
                                                      hidden_dim=self.config['disentangle_hidden_dim'],
                                                      nuisance_dim=self.enc_output_dim))
        self.disen_nuisance = cc_model(disentangle_nuisance(nuisance_dim=self.enc_output_dim,
                                                            hidden_dim=self.config['disentangle_hidden_dim'],
                                                            clean_repre_dim=self.enc_output_dim))
        print(self.disen_clean)
        print(self.disen_nuisance)
        self.addnoiselayer = addnoiselayer(dropout_p=self.delta)
        print(self.addnoiselayer)
        self.reconstructor = cc_model(inverse_pBLSTM(enc_hidden_dim=self.enc_output_dim*2,
                                                     hidden_dim=self.config['recons_hidden_dim'],
                                                     hidden_project_dim=self.config['recons_hidden_dim'],
                                                     output_dim=self.config['input_dim'],
                                                     n_layers=self.config['rec_n_layers'],
                                                     downsample=self.config['downsample']))
        print(self.reconstructor)

        self.encoder.float()
        self.encoder2.float()
        self.attention.float()
        self.decoder.float()
        self.disen_clean.float()
        self.disen_nuisance.float()
        self.addnoiselayer.float()
        self.reconstructor.float()
        self.params_m1=list(self.encoder.parameters())+list(self.attention.parameters())+\
            list(self.decoder.parameters())+list(self.reconstructor.parameters())+\
            list(self.encoder2.parameters())
        self.params_m2=list(self.disen_clean.parameters())+list(self.disen_nuisance.parameters())
        self.optimizer_m1= torch.optim.Adam(self.params_m1,
                                            lr=self.config['learning_rate_m1'], 
                                            weight_decay=self.config['weight_decay_m1'])
        self.optimizer_m2= torch.optim.Adam(self.params_m2,
                                            lr=self.config['learning_rate_m2'], 
                                            weight_decay=self.config['weight_decay_m2'])

        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        return

    def validation(self):
        self.encoder.eval()
        self.encoder2.eval()
        self.decoder.eval()
        self.attention.eval()
        self.disen_clean.eval()
        self.disen_nuisance.eval()
        self.reconstructor.eval()
        all_prediction, all_ys = [], []
        gold_transcripts = []
        total_loss = 0.
        for step, data in enumerate(self.dev_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ilens, ys, ys_in, ys_out, _, _, trans = to_gpu(data, bos, eos, pad)

            # input the encoder
            clean_repre, enc_lens, _, _ = self.encoder(xs, ilens)
            
            # feeding
            logits, log_probs, prediction, attns = \
                self.decoder(clean_repre, enc_lens, None,
                             max_dec_timesteps=self.config['max_dec_timesteps'])
            
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=log_probs.size(1)))
            loss = (-torch.sum(log_probs*mask))/sum(seq_len)
            total_loss += loss.item()

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]
            gold_transcripts+=trans
        
        # calculate loss
        avg_loss = total_loss / len(self.dev_loader)
        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)

        self.encoder.train()
        self.encoder2.train()
        self.decoder.train()
        self.attention.train()
        self.disen_clean.train()
        self.disen_nuisance.train()
        self.reconstructor.train()
        return avg_loss, cer, prediction_sents, ground_truth_sents
    
    def ind2sent(self, all_prediction, all_ys):
        # remove eos and pad
        prediction_til_eos = remove_pad_eos(all_prediction, eos=self.vocab['<EOS>'])
        # indexes to characters
        prediction_sents = to_sents(prediction_til_eos, self.vocab, self.non_lang_syms)
        ground_truth_sents = to_sents(all_ys, self.vocab, self.non_lang_syms)
        # calculate cer
        cer = calculate_cer(prediction_til_eos, all_ys)
        return cer, prediction_sents, ground_truth_sents

    def test(self, state_dict=None):
        # load model
        if not state_dict:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.encoder.load_state_dict(state_dict[0])
            self.attention.load_state_dict(state_dict[1])
            self.decoder.load_state_dict(state_dict[2])
        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_set = self.config['test_set']
        test_file_name = self.config['test_file_name']
        test_dataset = PickleDataset(os.path.join(root_dir, f'{test_set}.p'), 
            config=None, sort=False)
        test_loader = get_data_loader(test_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False)

        self.encoder.eval()
        self.encoder2.eval()
        self.decoder.eval()
        self.attention.eval()
        self.disen_clean.eval()
        self.disen_nuisance.eval()
        self.reconstructor.eval()
        all_prediction, all_ys = [], []
        gold_transcripts = []
        for step, data in enumerate(self.dev_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ilens, ys, ys_in, ys_out, spks, envs, trans = to_gpu(data, bos, eos, pad)
            
            # input the encoder
            clean_repre, enc_lens, _, _ = self.encoder(xs, ilens)
            noisy_repre, _, _, _ = self.encoder2(xs, ilens)
            # feeding
            logits, log_probs, prediction, attns = \
                self.decoder(clean_repre, enc_lens, None,
                             max_dec_timesteps=self.config['max_dec_timesteps'])
            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]
            gold_transcripts+=trans
        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)
        print(f'dev set CER: {cer:.4f}')

        all_prediction, all_ys = [], []
        gold_transcripts = []
        for step, data in enumerate(test_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ilens, ys, ys_in, ys_out, spks, envs, trans = to_gpu(data, bos, eos, pad)
            
            # input the encoder
            clean_repre, enc_lens, _, _ = self.encoder(xs, ilens)
            noisy_repre, _, _, _ = self.encoder2(xs, ilens)
            # feeding
            logits, log_probs, prediction, attns = \
                self.decoder(clean_repre, enc_lens, None,
                             max_dec_timesteps=self.config['max_dec_timesteps'])

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]
            gold_transcripts+=trans
        self.encoder.train()
        self.encoder2.train()
        self.decoder.train()
        self.attention.train()
        self.disen_clean.train()
        self.disen_nuisance.train()
        self.reconstructor.train()
        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)

        with open(f'{test_file_name}.txt', 'w') as f:
            for p in prediction_sents:
                f.write(f'{p}\n')

        print(f'{test_file_name}: {len(prediction_sents)} utterances, CER={cer:.4f}')
        return cer

    def _random_target(self,x, embedding_activation):
        range_low = -1 if embedding_activation == 'tanh' else 0
        range_high = 1
        tfake = torch.FloatTensor(x.size()).uniform_(range_low, range_high)
        return cc(tfake) 
    
    def train_one_epoch(self, epoch, tf_rate):

        total_steps = len(self.train_lab_loader)
        total_loss_prediction = 0.
        total_loss_reconstruction = 0.
        total_loss_disclean = 0.
        total_loss_disnoise = 0.
        
        #M2
        total_loss_disclean_dis = 0.
        total_loss_disnoise_dis = 0.
        for cnt in range(self.config['m2_train_freq']):
            sub_total_loss_disclean = 0.
            sub_total_loss_disnoise = 0.
            print()
            for train_steps, data in enumerate(self.train_lab_loader):
                bos = self.vocab['<BOS>']
                eos = self.vocab['<EOS>']
                pad = self.vocab['<PAD>']
                xs, ilens, ys, ys_in, ys_out, _, _, trans = to_gpu(data, bos, eos, pad)

                # add gaussian noise after gaussian_epoch
                if self.config['add_gaussian'] and epoch >= self.config['gaussian_epoch']:
                    gau = np.random.normal(0, self.config['gaussian_std'],
                                           (xs.size(0), xs.size(1), xs.size(2)))
                    gau = cc(torch.from_numpy(np.array(gau, dtype=np.float32)))
                    xs = xs + gau

                clean_repre, enc_lens, _, _ = self.encoder(xs, ilens)
                nuisance, nuisance_lens, _, _ =self.encoder2(xs, ilens)
                predict_nuisance = self.disen_clean(clean_repre, enc_lens)
                predict_clean = self.disen_nuisance(nuisance, nuisance_lens)
                loss_disclean = torch.mean((predict_clean-clean_repre.detach())**2)*self.gamma
                sub_total_loss_disclean += loss_disclean.item()
                loss_disnoise = torch.mean((predict_nuisance-nuisance.detach())**2)*self.gamma
                sub_total_loss_disnoise += loss_disnoise.item()
                
                self.optimizer_m2.zero_grad()
                (loss_disclean+loss_disnoise).backward()
                torch.nn.utils.clip_grad_norm_(self.params_m2, max_norm=self.config['max_grad_norm'])
                self.optimizer_m2.step()
            
                total_loss_disclean_dis += (sub_total_loss_disclean/self.config['m2_train_freq'])
                total_loss_disnoise_dis += (sub_total_loss_disnoise/self.config['m2_train_freq'])
                
                # print message
                print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}],'
                      f'dis_clean_l: {loss_disclean:.4f}, dis_noise_l:{loss_disnoise:.4f}', end='\r')
                # add to logger
                tag = self.config['tag']
                self.logger.scalar_summary(tag=f'{tag}/train/m2_disen_clean_loss',
                                           value=(loss_disclean.item()/self.gamma), 
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
                self.logger.scalar_summary(tag=f'{tag}/train/m2_disen_noise_loss',
                                           value=(loss_disnoise.item()/self.gamma),
                                           step=(epoch*(self.config['m2_train_freq'])+cnt)*total_steps+train_steps+1)
        print () 
        # M1
        for train_steps, data in enumerate(self.train_lab_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ilens, ys, ys_in, ys_out, _, _, trans = to_gpu(data, bos, eos, pad)
            # add gaussian noise after gaussian_epoch
            if self.config['add_gaussian'] and epoch >= self.config['gaussian_epoch']:
                gau = np.random.normal(0, self.config['gaussian_std'], (xs.size(0), xs.size(1), xs.size(2)))
                gau = cc(torch.from_numpy(np.array(gau, dtype=np.float32)))
                xs = xs + gau

            # input the encoder
            clean_repre, enc_lens, _, _ = self.encoder(xs, ilens)
            nuisance, nuisance_lens, _, _ = self.encoder2(xs, ilens)

            # M1 feeding
            logits, log_probs, prediction, attns = \
                self.decoder(clean_repre, enc_lens, (ys_in,ys_out), tf_rate=tf_rate, 
                             sample=False, max_dec_timesteps=self.config['max_dec_timesteps'])
            noisy_clean_repre = self.addnoiselayer(clean_repre)
            recons_input = torch.cat((noisy_clean_repre, nuisance), dim=-1)
            reconstruction, output_lens = self.reconstructor(recons_input, enc_lens)
            # M1 loss and backward
            # prediction loss
            loss = -torch.mean(log_probs)* self.config['alpha']
            total_loss_prediction += loss.item()
            
            # reconstruction loss
            # if pyramid encoder, the length would different for recons and xs
            recons_loss = torch.mean((reconstruction[:,0:xs.size(1),:] - xs)**2)*self.beta
            total_loss_reconstruction += recons_loss.item()               
            
            predict_nuisance = self.disen_clean(clean_repre, enc_lens)
            predict_clean = self.disen_nuisance(nuisance, nuisance_lens)
            rand_vec1 = self._random_target(clean_repre, 'tanh')
            rand_vec2 = self._random_target(nuisance, 'tanh')
            loss_disclean = torch.mean((predict_clean-rand_vec1)**2)*self.gamma
            total_loss_disclean += loss_disclean.item()
            loss_disnoise = torch.mean((predict_nuisance-rand_vec2)**2)*self.gamma
            total_loss_disnoise += loss_disnoise.item()
            
            # calculate gradients 
            self.optimizer_m1.zero_grad()
            (loss+recons_loss+loss_disclean+loss_disnoise).backward()
            torch.nn.utils.clip_grad_norm_(self.params_m1, max_norm=self.config['max_grad_norm'])

            self.optimizer_m1.step()

            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], loss: {loss:.3f},'
                  f'rec_l: {recons_loss:.3f}, dis_clean:{loss_disclean:.4f},'
                  f'dis_noise:{loss_disnoise:.4f}', end='\r')
            tag = self.config['tag']
            self.logger.scalar_summary(tag=f'{tag}/train/loss', 
                                       value=(loss.item())/self.config['alpha'], 
                                       step=epoch * total_steps + train_steps + 1)
            self.logger.scalar_summary(tag=f'{tag}/train/recons_loss', 
                                       value=(recons_loss.item())/self.beta, 
                                       step=epoch * total_steps + train_steps + 1)
            self.logger.scalar_summary(tag=f'{tag}/train/m1_disen_clean_loss',
                                       value=(loss_disclean.item())/(self.gamma),
                                       step=epoch * total_steps + train_steps + 1)
            self.logger.scalar_summary(tag=f'{tag}/train/m1_disen_noise_loss',
                                       value=(loss_disnoise.item())/self.gamma,
                                       step=epoch * total_steps + train_steps + 1)
        return ((total_loss_prediction / total_steps),
                (total_loss_reconstruction / total_steps),
                (total_loss_disclean / total_steps),
                (total_loss_disnoise / total_steps))

    def train(self):

        best_cer = 200
        best_model = None
        early_stop_counter = 0
        cer = 100
        # tf_rate
        init_tf_rate = self.config['init_tf_rate']
        tf_start_decay_epochs = self.config['tf_start_decay_epochs']
        tf_decay_epochs = self.config['tf_decay_epochs']
        tf_rate_lowerbound = self.config['tf_rate_lowerbound']

	    # lr scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_m1,
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=20,
                                                               verbose=True,
                                                               min_lr=1e-7)
        print('------start training-------')

        for epoch in range(self.config['epochs']):
            # schedule
            #scheduler.step(cer)
            # calculate tf rate
            if epoch > tf_start_decay_epochs:
                if epoch <= tf_decay_epochs:
                    tf_rate = init_tf_rate - (init_tf_rate - tf_rate_lowerbound) * ((epoch-tf_start_decay_epochs) / (tf_decay_epochs-tf_start_decay_epochs))
                else:
                    tf_rate = tf_rate_lowerbound
            else:
                tf_rate = init_tf_rate

            # train one epoch
            train_loss = self.train_one_epoch(epoch, tf_rate)

            # validation
            avg_valid_loss, cer, prediction_sents, ground_truth_sents  = self.validation()
            print(f'Epoch: {epoch}, tf_rate={tf_rate:.3f}, train_loss={train_loss[0]:.4f}, '
                  f'train_reconstruct_loss={train_loss[1]:.4f}, '
                  f'train_disen_clean_loss={train_loss[2]:.4f}, '
                  f'train_disen_noise_loss={train_loss[3]:.4f}, '
                  f'valid_loss={avg_valid_loss:.4f}, val_CER={cer:.4f}')

            for param_group in self.optimizer_m1.param_groups:
                print('Current Learning Rate: '+ str(param_group['lr']))
            # add to tensorboard
            tag = self.config['tag']
            self.logger.scalar_summary(f'{tag}/val/cer', cer, epoch)
            self.logger.scalar_summary(f'{tag}/val/loss', avg_valid_loss, epoch)

            # only add first n samples
            lead_n = self.config['sample_num']
            print('-----------------')
            for i, (p, gt) in enumerate(zip(prediction_sents[:lead_n], ground_truth_sents[:lead_n])):
                self.logger.text_summary(f'{tag}/sample/prediction-{i}', p, epoch)
                self.logger.text_summary(f'{tag}/sample/ground_truth-{i}', gt, epoch)
                print(f'prediction-{i+1}: {p}')
                print(f'reference-{i+1}: {gt}')
            print('-----------------')
            # save model in every epoch
            if not os.path.exists(self.config['model_dir']):
                os.makedirs(self.config['model_dir'])
            model_path = os.path.join(self.config['model_dir'], self.config['model_name'])
            self.save_model(f'{model_path}-{epoch:03d}')
            self.save_model(f'{model_path}_latest')
            if cer < best_cer: 
                # save model
                model_path = os.path.join(self.config['model_dir'], self.config['model_name']+'_best')
                best_cer = cer
                self.save_model(model_path)
                best_model_enc = copy.deepcopy(self.encoder.state_dict())
                best_model_att = copy.deepcopy(self.attention.state_dict())
                best_model_dec = copy.deepcopy(self.decoder.state_dict())
                print(f'Save #{epoch} model, val_loss={avg_valid_loss:.4f}, CER={cer:.4f}')
                print('-----------------')
                early_stop_counter=0
            if epoch >= self.config['early_stop_start_epoch']:
                early_stop_counter += 1
                if early_stop_counter > self.config['early_stop']:
                    break
            best_model = (best_model_enc, best_model_att, best_model_dec)
        print("----finish training----")
        print(f'---best cer: {best_cer}----')
        return best_model, best_cer
    
    def get_z_accuracy_data(self, state_dict=None):
        # this function will collect the representation of the testing dataset, 
        # for z_accuracy_test, you need to put the "training data" in the "
        # test_set" config so that able to go further test
        if not state_dict:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.encoder.load_state_dict(state_dict[0])
            self.attention.load_state_dict(state_dict[1])
            self.decoder.load_state_dict(state_dict[2])
        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_set = self.config['test_set']
        test_file_name = self.config['test_file_name']
        test_dataset = PickleDataset(os.path.join(root_dir, f'{test_set}.p'), 
            config=None, sort=False)
        test_loader = get_data_loader(test_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False)
        self.encoder.eval()
        self.encoder2.eval()
        self.decoder.eval()
        self.attention.eval()
        self.disen_clean.eval()
        self.disen_nuisance.eval()
        self.reconstructor.eval()
        stored_data_clean = dict()
        stored_data_noisy = dict()
        for step, data in enumerate(test_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ilens, ys, ys_in, ys_out, spks, envs, trans = to_gpu(data, bos, eos, pad)
            clean_repre, enc_lens, _, _ = self.encoder(xs, ilens)
            noisy_repre, _, _, _ = self.encoder2(xs, ilens)
            clean_representation = clean_repre.cpu().detach()
            noisy_representation = noisy_repre.cpu().detach()
            speakers = spks.cpu().tolist()
            environments = envs.cpu().tolist()
            for instance in range(clean_repre.size(0)):
                stored_data_clean[((step*self.config['batch_size'])+instance)]={
                    'embedding':trim_representation(clean_representation[instance], enc_lens[instance]),
                    'speaker':speakers[instance],
                    'env':environments[instance],
                    'transcripts':trans[instance]
                }
                stored_data_noisy[((step*self.config['batch_size'])+instance)]={
                    'embedding':trim_representation(noisy_representation[instance], enc_lens[instance]),
                    'speaker':speakers[instance],
                    'env':environments[instance],
                    'transcripts':trans[instance]
                }

        pickle.dump(stored_data_clean, open(self.config['z_data_path_clean']+'_b'+str(self.beta)+'g'+str(self.gamma)+'.p', 'wb'))
        pickle.dump(stored_data_noisy, open(self.config['z_data_path_noisy']+'_b'+str(self.beta)+'g'+str(self.gamma)+'.p', 'wb'))
        self.encoder.train()
        self.encoder2.train()
        self.decoder.train()
        self.attention.train()
        self.disen_clean.train()
        self.disen_nuisance.train()
        self.reconstructor.train()
        return 
