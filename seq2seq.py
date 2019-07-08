import torch 
import torch.nn.functional as F
import numpy as np
from model import E2E
from dataloader import get_data_loader
from dataset import PickleDataset
from utils import *
from utils import _seq_mask
import copy
import yaml
import os
import pickle

class Seq2seq(object):
    def __init__(self, config, load_model=False):

        self.config = config
        print(self.config)
        # logger
        self.logger = Logger(config['logdir'])
        # load vocab and non lang syms
        self.load_vocab()
        # get data loader
        self.get_data_loaders()
        # get label distribution
        self.get_label_dist(self.train_lab_dataset)
        # calculate proportion between features and characters
        self.proportion = self.calculate_length_proportion()
        # build model and optimizer
        self.build_model(load_model=load_model)

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), f'{model_path}.ckpt')
        torch.save(self.gen_opt.state_dict(), f'{model_path}.opt')
        return

    def load_vocab(self):
        with open(self.config['vocab_path'], 'rb') as f:
            self.vocab = pickle.load(f) # a dict;character to index
        with open(self.config['non_lang_syms_path'], 'rb') as f:
            self.non_lang_syms = pickle.load(f)
        return

    def load_model(self, model_path, load_optimizer):
        print(f'Load model from {model_path}.ckpt')
        self.model.load_state_dict(torch.load(f'{model_path}.ckpt'))
        if load_optimizer:
            print(f'Load optmizer from {model_path}.opt')
            self.gen_opt.load_state_dict(torch.load(f'{model_path}.opt'))
            self.gen_opt=adjust_learning_rate(self.gen_opt, self.config['retrieve_learning_rate'])
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
                batch_size=self.config['batch_size'] // 2, 
                shuffle=False)
        return

    def get_label_dist(self, dataset):
        labelcount = np.zeros(len(self.vocab))
        for _, y,_,_,_ in dataset:
            for ind in y:
                labelcount[ind] += 1.
        labelcount[self.vocab['<EOS>']] += len(dataset)
        labelcount[self.vocab['<PAD>']] = 0
        labelcount[self.vocab['<BOS>']] = 0
        self.labeldist = labelcount / np.sum(labelcount)
        return

    def calculate_length_proportion(self):
        x_len, y_len = 0, 0
        for x, y,_,_,_ in self.train_lab_dataset:
            x_len += x.shape[0]
            y_len += len(y)
        return y_len / x_len
             
    def build_model(self, load_model=False):
        labeldist = self.labeldist
        ls_weight = self.config['ls_weight']

        self.model = cc_model(E2E(input_dim=self.config['input_dim'],
            enc_hidden_dim=self.config['enc_hidden_dim'],
            enc_n_layers=self.config['enc_n_layers'],
            subsample=self.config['subsample'],
            enc_output_dim=self.config['enc_output_dim'],
            dropout_rate=self.config['dropout_rate'],
            dec_hidden_dim=self.config['dec_hidden_dim'],
            att_dim=self.config['att_dim'],
            conv_channels=self.config['conv_channels'],
            conv_kernel_size=self.config['conv_kernel_size'],
            att_odim=self.config['att_odim'],
            output_dim=len(self.vocab),
            embedding_dim=self.config['embedding_dim'],
            ls_weight=ls_weight,
            labeldist=labeldist,
            pad=self.vocab['<PAD>'],
            bos=self.vocab['<BOS>'],
            eos=self.vocab['<EOS>']
            ))
        print(self.model)
        self.model.float()
        self.gen_opt = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.config['learning_rate'], 
                                        weight_decay=self.config['weight_decay'])
        if load_model:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        return

    def validation(self):
        self.model.eval()
        all_prediction, all_ys = [], []
        gold_transcripts = []
        total_loss = 0.
        for step, data in enumerate(self.dev_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ilens, ys, ys_in, ys_out, _, _, trans = to_gpu(data, bos, eos, pad)
            log_probs , prediction, attns, _, _ =\
                self.model(xs, ilens, None, 
                           max_dec_timesteps=self.config['max_dec_timesteps'])
            seq_len = [y.size(0) + 1 for y in ys]
            mask = cc(_seq_mask(seq_len=seq_len, max_len=log_probs.size(1)))
            loss = (-torch.sum(log_probs*mask))/sum(seq_len)
            total_loss += loss.item()

            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]
            gold_transcripts+=trans
        
        avg_loss = total_loss / len(self.dev_loader)
        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)
        self.model.train()
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
            self.model.load_state_dict(state_dict)
        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_set = self.config['test_set']
        test_file_name = self.config['test_file_name']
        test_dataset = PickleDataset(os.path.join(root_dir, f'{test_set}.p'), 
            config=None, sort=False)
        test_loader = get_data_loader(test_dataset, 
                batch_size=2, 
                shuffle=False)
        
        self.model.eval()
        all_prediction, all_ys = [], []
        gold_transcripts = []
        for step, data in enumerate(self.dev_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ilens, ys, ys_in, ys_out, spks, envs, trans = to_gpu(data, bos, eos, pad)
            # feed previous
            _ , prediction, _, enc_outputs, enc_lens = self.model(xs, ilens, None, 
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
            # feed previous
            _ , prediction, _ , enc_outputs, enc_lens= self.model(xs, ilens, None, 
                    max_dec_timesteps=self.config['max_dec_timesteps'])
            all_prediction = all_prediction + prediction.cpu().numpy().tolist()
            all_ys = all_ys + [y.cpu().numpy().tolist() for y in ys]
            gold_transcripts+=trans
        self.model.train()
        cer, prediction_sents, ground_truth_sents = self.ind2sent(all_prediction, all_ys)
        with open(f'{test_file_name}.txt', 'w') as f:
            for p in prediction_sents:
                f.write(f'{p}\n')
        print(f'{test_file_name}: {len(prediction_sents)} utterances, CER={cer:.4f}')
        return cer
    
    def train_one_epoch(self, epoch, tf_rate):
        total_steps = len(self.train_lab_loader)
        total_loss = 0.
        for train_steps, data in enumerate(self.train_lab_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ilens, ys, ys_in, ys_out, _, _,trans = to_gpu(data, bos, eos, pad)
            # add gaussian noise after gaussian_epoch
            if self.config['add_gaussian'] and epoch >= self.config['gaussian_epoch']:
                gau = np.random.normal(0, self.config['gaussian_std'], (xs.size(0), xs.size(1), xs.size(2)))
                gau = cc(torch.from_numpy(np.array(gau, dtype=np.float32)))
                xs = xs + gau
            # input the model
            log_probs, prediction, attns, _, _ = self.model(xs, ilens, (ys_in,ys_out), 
                                                      tf_rate=tf_rate, sample=False)
            # mask and calculate loss
            loss = -torch.mean(log_probs)
            total_loss += loss.item()
            # calculate gradients 
            self.gen_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['max_grad_norm'])
            self.gen_opt.step()
            
            print(f'epoch: {epoch}, [{train_steps + 1}/{total_steps}], loss: {loss:.3f}', end='\r')
            # add to logger
            tag = self.config['tag']
            self.logger.scalar_summary(tag=f'{tag}/train/loss', value=loss.item(), 
                    step=epoch * total_steps + train_steps + 1)

        return total_loss / total_steps

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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.gen_opt, 
                                                               mode='min',
                                                               factor=0.5,
                                                               patience=25,
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
            avg_train_loss = self.train_one_epoch(epoch, tf_rate)

            # validation
            avg_valid_loss, cer, prediction_sents, ground_truth_sents= self.validation()
            print(f'Epoch: {epoch}, tf_rate={tf_rate:.3f}, train_loss={avg_train_loss:.4f}, '
                  f'valid_loss={avg_valid_loss:.4f}, val_CER={cer:.4f}')
            for param_group in self.gen_opt.param_groups:
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
                best_model = copy.deepcopy(self.model.state_dict())
                print(f'Save #{epoch} model, val_loss={avg_valid_loss:.3f}, val_CER={cer:.3f}')
                print('-----------------')
                early_stop_counter=0
            if epoch >= self.config['early_stop_start_epoch']:
                early_stop_counter += 1
                if early_stop_counter > self.config['early_stop']:
                    break
        return best_model, best_cer

    def get_z_accuracy_data(self, state_dict=None):
        # this function will collect the representation of the testing dataset, 
        # for z_accuracy_test, you need to put the "training data" in the "
        # test_set" config so that able to go further test
        if not state_dict:
            self.load_model(self.config['load_model_path'], self.config['load_optimizer'])
        else:
            self.model.load_state_dict(state_dict)
        # get test dataset
        root_dir = self.config['dataset_root_dir']
        test_set = self.config['test_set']
        test_file_name = self.config['test_file_name']
        test_dataset = PickleDataset(os.path.join(root_dir, f'{test_set}.p'), 
            config=None, sort=False)
        test_loader = get_data_loader(test_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=False)
        self.model.eval()
        stored_data = dict()
        for step, data in enumerate(test_loader):
            bos = self.vocab['<BOS>']
            eos = self.vocab['<EOS>']
            pad = self.vocab['<PAD>']
            xs, ilens, ys, ys_in, ys_out, spks, envs, trans = to_gpu(data, bos, eos, pad)
            # feed previous
            _ , prediction, _, enc_outputs, enc_lens = self.model(xs, ilens, None, 
                    max_dec_timesteps=self.config['max_dec_timesteps'])
            representations = enc_outputs.cpu().detach()
            speakers = spks.cpu().tolist()
            environments = envs.cpu().tolist()
            for instance in range(enc_outputs.size(0)):
                stored_data[((step*self.config['batch_size'])+instance)]={
                    'embedding':trim_representation(representations[instance], enc_lens[instance]),
                    'speaker':speakers[instance],
                    'env':environments[instance],
                    'transcripts':trans[instance]
                }
        pickle.dump(stored_data, open(self.config['z_data_path'], 'wb'))
        self.model.train()
        return 
