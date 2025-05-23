import argparse

import wandb.sdk

from exp.exp_basic import Exp_Basic

import numpy as np

import os
import pickle

import src.layers.graphs
from src.models import Persistence, MLP, LSTM, FFTransformer, GraphPersistence, GraphMLP, GraphLSTM, GraphFFTransformer
from src.data.data_factory import data_provider
from src.data.data_loader import Dataset_wind_data, Dataset_wind_data_graph
from src.utils.graph_utils import data_dicts_to_graphs_tuple
from src.utils.metrics import metric
from src.utils.tools import visual, EarlyStopping, adjust_learning_rate, PlotLossesSame


#import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import wandb

import time



import warnings

warnings.filterwarnings("error")

class Exp_Main(Exp_Basic):
    def __init__(self, args: argparse.Namespace) -> None:
        super(Exp_Main, self).__init__(args)
        if self.args.data == 'WindGraph':
            self.args.seq_len = self.args.label_len
        
        
    def _build_model(self) -> nn.Module:
        model_dict = {
            'FFTransformer': FFTransformer,
            'LSTM': LSTM,
            'MLP': MLP,
            'persistence': Persistence,
            'GraphLSTM': GraphLSTM,
            'GraphFFTransformer': GraphFFTransformer,
            'GraphMLP': GraphMLP,
            'GraphPersistence': GraphPersistence,
        }

        model = model_dict[self.args.model].Model(self.args).float()

        return model
    


    def _get_data(self, flag: str) -> list[Dataset_wind_data, DataLoader]:
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
        
        
    def _select_criterion(self) -> nn.modules.loss.MSELoss:
        criterion = nn.MSELoss()
        return criterion
    
    
    
    def _select_optimizer(self) -> optim.Adam:
        model_optim = optim.Adam(self.model.parameters(),
                                 lr=self.args.learning_rate)
        return model_optim
    
    
    
    def vali(self, setting: str,
             vali_data: Dataset_wind_data | Dataset_wind_data_graph, 
             vali_loader: DataLoader,
             criterion: nn.modules.loss.MSELoss,
             epoch: int = 0,
             plot_res: int = 1,
             save_path: str | None = None
             ) -> np.float32:
        total_loss = []
        total_mse = []
        total_mape = []
        self.model.eval()
        
        
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                
                #graph data
                if self.args.data == "WindGraph":
                    #convert graphs to GraphTuple
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y       #decoder input starts as a copy of the target graph
                    
                    #move time data to device
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    #construct decoder input
                    #1 select the last observed node feature (at label_len-1)
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]        
                    #2 repeat this value for each prediction step (acts as placeholder)
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                        
                    #3 concatenate observed part of decoder input (first label_len steps)
                    #    with repeated last observed value to create full decoder input
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros) #update the dec_inp to be the data with placeholder

                    #select last `dec_in` features for decoder and `enc_in` features for encoder
                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])
                    
                
                #non-graph data
                else:
                    #move data to device
                    dec_inp = batch_y

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]         # Select last value ([32, 54, 23] --> [32, 1, 23]), both corresponding to (batch size, # rows, # features)
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                 # Repeat for pred_len
                    #select first <label_len> rows from dec_inp (aka batch_y) and append the dec_zeros tensor (concat((32,48,23), (32,6,23), dim=1) --> (32,54, 23))
                    dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)   # Add Placeholders
                    dec_inp = dec_inp.float().to(self.device)

                    #select only the last <dec_in/enc_in> columns (Think this should simply be # of features in dataset)
                    dec_inp = dec_inp[:, :, -self.args.dec_in:]
                    batch_x = batch_x[:, :, -self.args.enc_in:]
                    

                #Extract only outputs (models with attention mechanism also return attn from the forward functoin)
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                #Uni- or Multivariate (extra guard-railing -> should not be neccessary, assuming model only ouput c_out features)
                if 'M' in self.args.features:
                    f_dim = -self.args.c_out #select last c_out features of the model output 
                else:
                    f_dim = 0 #there is only one feature in case "S"
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                
                #Extract true predictions (last pred_len values of batch_y)
                if self.args.data == 'WindGraph':
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                #Get predictions and true values back to cpu
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                #compute loss (MSE loss) -> same as ((pred - true)**2).mean()
                # that is, average squared dist between pred and true summed over
                # all nodes             (# of turbines)
                # all prediction steps  (pred_len)
                # all output features   (c_out)
                loss = criterion(pred, true)
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss
                  
                    
    
    def train(self, setting: str) -> ...:
        train_data, train_loader = self._get_data(flag = "train")
        vali_data, vali_loader = self._get_data(flag = "val")
        test_data, test_loader = self._get_data(flag = "test")
        
        
        if 'sistence' in self.args.model:
            criterion = self._select_criterion()
            vali_loss = self.vali(setting, vali_data, vali_loader, criterion)
            test_loss = self.vali(setting, test_data, test_loader, criterion)

            self.test('persistence_' + str(self.args.pred_len), test=0, save_flag=True)
        
            print("Persistence model - skipping training")
            print('vali_loss: ', vali_loss)
            print('test_loss: ', test_loss)
            return False #assert False
        
        
        self.vali_losses = [] #store validation lossses
        
        
        #Ensure there exists a dir for storing checkpoints (if specified)
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and self.args.checkpoint_flag:
            os.makedirs(path)
        
    
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience,
                                       verbose=True,
                                       checkpoint=self.args.checkpoint_flag, 
                                       model_setup=self.args)    
        
        
        #Tries to load best model from checkpoint,
        if self.args.checkpoint_flag:
            load_path = os.path.join(path, "checkpoint.pth")
            if os.path.exists(load_path) and self.load_check(path=os.path.join(path, 'model_setup.pickle')):
                self.model.load_state_dict(torch.load(load_path, weights_only=True))
                
                epoch_pickle_path = os.path.join('./checkpoints/' + setting, 'epoch_loss.pickle')
                with open(epoch_pickle_path, "rb") as f:
                    epoch_info = pickle.load(f)
                
                start_epoch = epoch_info['epoch']
                early_stopping.val_losses = epoch_info['val_losses']
                early_stopping.val_loss_min = epoch_info['val_loss_min']
                self.vali_losses = epoch_info['val_losses']
                del epoch_info
            else:
                start_epoch = 0
                print('Could not load best model')
        else:
            start_epoch = 0
        
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        
        teacher_forcing_ratio = 0.8     # For LSTM Enc-Dec training (not used for others).
        total_num_iter = 0
        
        if self.args.use_wandb:
            wandb.init(project="WattTheForecast", name=setting, config=self.args)
            wandb.watch(self.model, log="all", log_freq=10)
        
        #Track train time and memory usage
        training_start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=self.device)
        
        for epoch in range(start_epoch, self.args.train_epochs):
            print(epoch)
        
            if self.args.model == "LSTM":
                teacher_forcing_ratio -= 0.08
                teacher_forcing_ratio = max(0., teacher_forcing_ratio)
                print("teacher_forcing_ratio: ", teacher_forcing_ratio)
        
            #type4 lr scheduling is updated more frequently
            if self.args.lradj != 'type4':
                adjust_learning_rate(model_optim, epoch+1, self.args)
                
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            num_iters = len(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                
                if self.args.lradj == 'type4':
                    adjust_learning_rate(model_optim, total_num_iter + 1, self.args)
                    total_num_iter += 1
            
                if isinstance(batch_y, dict): #this is only true in case of WindGraph data
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y

                    model_optim.zero_grad()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]        # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                        # Repeat for pred_len
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)  # Add Placeholders
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)

                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])

                else:
                    dec_inp = batch_y
                    
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)
                    
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)
                    
                    #decoder input
                    dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]         # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                 # Repeat for pred_len
                    dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)   # Add Placeholders
                    dec_inp = dec_inp.float().to(self.device)

                    dec_inp = dec_inp[:, :, -self.args.dec_in:]
                    batch_x = batch_x[:, :, -self.args.enc_in:]
                
                                
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         teacher_forcing_ratio=teacher_forcing_ratio, batch_y=batch_y)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         teacher_forcing_ratio=teacher_forcing_ratio, batch_y=batch_y)
                
                #uni- vs multivariate
                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                #wrangle batch_y
                if isinstance(batch_y, src.layers.graphs.GraphsTuple): #this is only true in case of WindGraph data
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                loss.backward()
                model_optim.step()
                
                if (i + 1) % 10 == 0 and self.args.verbose == 1:
                    print("\titers: {0}/{1}, epoch: {2} | loss: {3:.7f}".format(i + 1, num_iters, epoch + 1, np.average(train_loss)))
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(setting, vali_data, vali_loader, criterion, epoch=epoch, save_path=path)
            test_flag = False
            if test_flag:
                test_loss = self.vali(setting, test_data, test_loader, criterion, epoch=epoch, save_path=path)
            
            
            print(model_optim.param_groups[0]["lr"])
            if self.args.use_wandb:
                wandb.log({
                    "epoch" : epoch + 1,
                    "train_loss" : train_loss,
                    "val_loss" : vali_loss,
                    "learning_rate" : model_optim.param_groups[0]["lr"]
                })
            
                        
            #plot the losses (either create new viz or update existing)
            if self.args.plot_flag and self.args.checkpoint_flag:
                loss_save_dir = path + '/pic/train_loss.png'
                loss_save_dir_pkl = path + '/train_loss.pickle'
                
                if os.path.exists(loss_save_dir_pkl):
                    with open(loss_save_dir_pkl, "rb") as f:
                        fig_progress = pickle.load(f)
                
                #if the above if-statement "fails" and viz can't be loaded, then create new viz
                if "fig_progress" not in locals():
                    fig_progress = PlotLossesSame(epoch + 1, 
                                                  Training=train_loss,
                                                  Validation=vali_loss)
                #else: update the loaded figure
                else:
                    fig_progress.on_epoch_end(Training=train_loss,
                                              Validation=vali_loss)
                
                if not os.path.exists(os.path.dirname(loss_save_dir)):
                    os.makedirs(os.path.dirname(loss_save_dir))
                
                fig_progress.fig.savefig(loss_save_dir, dpi=100)
                with open(loss_save_dir_pkl, "wb") as f:
                    pickle.dump(fig_progress, f)
                
            
            if test_flag:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))
            
            
            
            early_stopping(vali_loss, self.model, path, epoch)
            self.vali_losses += [vali_loss]       # Append validation loss
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        
        #Get the training time and max memory usage and store them
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        print("TOTAL TRAIN TIME:", total_training_time)
        
        if self.device.type == "cuda":
            max_memory = torch.cuda.max_memory_allocated(device=self.device) / (1024 ** 2)
            print("MAX MEMORY USAGE:", max_memory)   
        
        stats_path = os.path.join(path, "training_stats.txt")
        with open(stats_path, "w") as f:
            f.write(f"Total training time (s): {total_training_time}\n")
            if self.device.type == "cuda":
                f.write(f"Peak GPU memory usage (MB): {max_memory}")
            else:
                f.write(f"Peak GPU memory usage (MB): None - need to train model using CUDA to get this stat")
        
        
        
        #after training, load the best model
        if self.args.checkpoint_flag:
           best_model_path = path + "/" + "checkpoint.pth"
           self.model.load_state_dict(torch.load(best_model_path, weights_only=True))
        
        if self.args.use_wandb:
            wandb.unwatch(self.model)
            wandb.finish()
            
        return self.model
    
    
    
    # Function to check that the checkpointed and current settings are compatible.
    def load_check(self, path: str,
                   ignore_vars: list[str] = None,
                   ignore_paths: bool = False
                   ) -> bool:
        if ignore_vars is None:
            ignore_vars = [
                'is_training',
                'train_epochs',
                'plot_flag',
                'root_path',
                'data_path',
                'data_path',
                'checkpoints',
                'checkpoint_flag',
                'output_attention',
                'do_predict',
                'des',
                'n_closest',
                'verbose',
                'data_step',
                'itr',
                'patience',
                'des',
                'gpu',
                'use_gpu',
                'use_multi-gpu',
                'devices',
            ]
        if ignore_paths:
            ignore_vars += [
                'model_id',
                'test_dir',
            ]

        with open(path, "rb") as f:
            setting2 = pickle.load(f)
        for key, val in self.args.__dict__.items():
            if key in ignore_vars:
                continue
            if val != setting2[key]:
                print(val, ' is not equal to ', setting2[key], ' for ', key)
                return False

        return True
    
        
        
    def test(self, setting: str,
             test: int = 1,
             base_dir: str = '',
             save_dir: str | None = None,
             ignore_paths: bool = False,
             save_flag: bool = True,
             ) -> dict[str, ]:
        
        test_data, test_loader = self._get_data(flag='test')
        
        #if save_dir is not specified, use base_dir as default
        if save_dir is None:
            save_dir = base_dir
        
        
        #Load and test model if <test> flag is enabled
        if test:
            print('loading model')
            
            #construct path to checkpoint file
            if len(base_dir) == 0:
                load_path = os.path.normpath(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            else:
                load_path = os.path.normpath(os.path.join(base_dir + 'checkpoints/' + setting, 'checkpoint.pth'))
            
            #validate the current arguments against the saved model config
            load_check_flag = self.load_check(path=os.path.normpath(os.path.join(os.path.dirname(load_path),
                                                                                 'model_setup.pickle')),
                                              ignore_paths=ignore_paths)
            
            #load model weights if checkpoints exists and the config is valid (as per previous check)
            if os.path.exists(load_path) and load_check_flag:
                self.model.load_state_dict(torch.load(load_path, weights_only=True))
            else:
                print('Could not load best model')
        
                
        #Ensure that a dir exists for storing results
        if save_flag:
            if len(save_dir) == 0:
                folder_path = './test_results/' + setting + '/'
            else:
                folder_path = save_dir + 'test_results/' + setting + '/'        
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
        
        
        preds = []
        trues = []
        node_ids = []
        
        self.model.eval()
        
        inference_start_time = time.time()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device=self.device)
        
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if self.args.data == "WindGraph":
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]        # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                        # Repeat for pred_len
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)  # Add Placeholders
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)

                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])
                else:
                    #move data to device
                    dec_inp = batch_y

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]         # Select last value ([32, 54, 23] --> [32, 1, 23]), both corresponding to (batch size, # rows, # features)
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                 # Repeat for pred_len
                    #select first <label_len> rows from dec_inp (aka batch_y) and append the dec_zeros tensor (concat((32,48,23), (32,6,23), dim=1) --> (32,54, 23))
                    dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)   # Add Placeholders
                    dec_inp = dec_inp.float().to(self.device)

                    #select only the last <dec_in/enc_in> columns (Think this should simply be # of features in dataset)
                    dec_inp = dec_inp[:, :, -self.args.dec_in:]
                    batch_x = batch_x[:, :, -self.args.enc_in:]
                    

                #Only store outputs (transformer models can also ouput attention, we don't store that)
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)


                #Select ouput features
                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                
                
                #Extract true values
                if self.args.data == 'WindGraph':
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)


                #Get predictions and true values back to cpu
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
        
        
                if self.args.data == "WindGraph":
                    node_ids.append(batch_x.station_names)
                
            
                #Create and save visualization
                if i % 20 == 0:
                    if self.args.data == "WindGraph":
                        input = batch_x.nodes.detach().cpu().numpy()
                    else:
                        input = batch_x.detach().cpu().numpy()
                                        
                    ground_truth = np.concatenate((input[0, :, -1], true[0, :, -1]), axis = 0)
                    prediction = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis = 0)
                    if save_flag:
                        visual(ground_truth, prediction, os.path.join(folder_path, str(i) + '.png'))
        
        preds = np.vstack(preds)
        trues = np.vstack(trues)
        if self.args.data == "WindGraph":
            node_ids = np.concatenate(node_ids)
        
        print('test shape:', preds.shape, trues.shape)
        
        # For saving NOTE: seems redundant --> similar already implemented earlier...
        if save_flag:
           if len(save_dir) == 0:
               folder_path = './results/' + setting + '/'
           else:
               folder_path = save_dir + 'results/' + setting + '/'
           if not os.path.exists(folder_path):
               os.makedirs(folder_path)
        
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
            
                
        losses = {
            'mae_sc': mae,
            'mse_sc': mse,
            'rmse_sc': rmse,
            'mape_sc': mape,
            'mspe_sc': mspe,
        }
        
        #Get the inference time and max memory usage and store them
        inference_end_time = time.time()
        total_inference_time = inference_end_time - inference_start_time
        print("TOTAL INFERENCE TIME:", total_inference_time)
        
        if self.device.type == "cuda":
            max_memory = torch.cuda.max_memory_allocated(device=self.device) / (1024 ** 2)
            print("MAX MEMORY USAGE (inference):", max_memory)   
        
        stats_path = os.path.join(folder_path, "inference_stats.txt")
        with open(stats_path, "w") as f:
            f.write(f"Total inference time (s): {total_inference_time}\n")
            if self.device.type == "cuda":
                f.write(f"Peak GPU memory usage (MB): {max_memory}")
            else:
                f.write(f"Peak GPU memory usage (MB): None - need to train model using CUDA to get this stat")
        
        if self.args.data == "WindGraph":
            all_unscaled_preds = []    
            all_unscaled_trues = []           
            
            for i, node in enumerate(np.unique(node_ids)):
                
                #Get the prediction and true values associated with the specific turbine
                indxs_i = np.where(node_ids == node)[0]
                preds_i = preds[indxs_i]
                trues_i = trues[indxs_i]

                #compute metrics of scaled values
                mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(preds_i, trues_i)

                #Unscale the values and save them 
                preds_unscaled_i = test_data.inverse_transform(preds_i, i)
                trues_unscaled_i = test_data.inverse_transform(trues_i, i)
                all_unscaled_preds.append(preds_unscaled_i)
                all_unscaled_trues.append(trues_unscaled_i)
                
                #compute metrics of un-scaled values
                mae_un_i, mse_un_i, rmse_un_i, mape_un_i, mspe_un_i = metric(preds_unscaled_i, trues_unscaled_i)

                losses_i = {
                    '': '\n\n',
                    'mae_sc_' + node: mae_i,
                    'mse_sc_' + node: mse_i,
                    'rmse_sc_' + node: rmse_i,
                    'mape_sc_' + node: mape_i,
                    'mspe_sc_' + node: mspe_i,
                    'mae_un_' + node: mae_un_i,
                    'mse_un_' + node: mse_un_i,
                    'rmse_un_' + node: rmse_un_i,
                    'mape_un_' + node: mape_un_i,
                    'mspe_un_' + node: mspe_un_i,
                }
                losses.update(losses_i)
        
            #Combine all unscaled values
            preds_unscaled = np.vstack(all_unscaled_preds)
            trues_unscaled = np.vstack(all_unscaled_trues)
            
            #compute loss of unscaled values
            mae_un, mse_un, rmse_un, mape_un, mspe_un = metric(preds_unscaled, trues_unscaled)

            losses_un = {
                "mae_un" : mae_un,
                "mse_un" : mse_un,
                "rmse_un" : rmse_un,
                "mape_un" : mape_un,
                "mspe_un" : mspe_un,
            }

            losses.update(losses_un)
        
        
        if not save_flag:
            return losses
        
        #Save losses
        with open(folder_path + "results_loss.txt", 'w') as f:
            for key, value in losses.items():
                f.write('%s:%s\n' % (key, value))
                
        print('mse:{}, mae:{}'.format(mse, mae))

        
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'pred_un.npy', preds_unscaled)
        np.save(folder_path + 'true_un.npy', trues_unscaled)
        
        if self.args.data == "WindGraph":
            np.save(folder_path + 'station_ids.npy', node_ids)
        
        with open(folder_path + 'metrics.txt', 'w') as f:
            f.write('mse: ' + str(mse) + '\n')
            f.write('mae: ' + str(mae) + '\n')
            f.write('rmse: ' + str(rmse) + '\n')
            f.write('mape: ' + str(mape) + '\n')
            f.write('mspe: ' + str(mspe) + '\n')

        return losses