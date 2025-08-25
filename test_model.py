# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 19:25:08 2025

@author: adamc




"""
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        # disable any autoreload extension that may have been loaded
        ip.extension_manager.disable('autoreload')
        # also turn off the magic in case it’s already active
        ip.run_line_magic('autoreload', '0')
except Exception:
    pass


import sys,os,warnings
import pandas as pd
import numpy as np
import copy
import torch
import torch.nn  as nn
from torch_geometric.loader import DataLoader
import time

from gnn_embedding_classes import (AttentiveFP, AttentiveFP_zindo_A, dense_one_descriptor, dense_multi_descriptor_Z)
from file_io import load_pkl_data, parse_sh_args, load_variables_from_str, load_scalers_robust
from train_and_plot_functions import calc_rhoc, true_vs_pred_plot_v2, scale_feats
from process_mol_dataset_class import MoleculeDataset
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error



if __name__ == "__main__":
    from pathlib import Path

    cwd = Path.cwd().as_posix()
    save_dir=cwd+'/saves'
    
    test_data_path=cwd+"/data/pre-processed_testset_407_mols.pkl"
    train_data_path=cwd+"/data/pre-processed_trainset_9050_mols.pkl"
    kfold_indices_path=cwd+"/data/10_kfold_indices_9050_mol.pkl"
    n_kfolds=10
    runs=["AttentiveFP_with_Zindo","AttentiveFP_only_electronic_atom_descriptors","AttentiveFP_only_default_atom_descriptors"]
    csv_name="test_data.csv"

    
    show_plots, save_plots, save_csv = False, True, True
    
    print("\nTest data:",test_data_path.split('/')[-1])
    print("Train data:",train_data_path.split('/')[-1])
    print("Fold indices:",kfold_indices_path.split('/')[-1])
    
    test_dataset_init=load_pkl_data(test_data_path)[:]
    train_dataset_init=load_pkl_data(train_data_path) # Loading this in to use validation folds
    kfold_indices=load_pkl_data(kfold_indices_path)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device='cpu'
    criterion = nn.MSELoss()
    
    # Lists to hold metrics for each model after averaging the fold data
    val_r_mean_list,test_r_mean_list,val_loss_mean_list,test_loss_mean_list=[],[],[],[]
    val_r_std_list,test_r_std_list,val_loss_std_list,test_loss_std_list=[],[],[],[]
    test_r_from_mean=[]
    pearson_test_r_err_model_list,pearson_test_r_model_list,pearson_test_r_val_model_list,pearson_test_r_err_val_model_list=[],[],[],[]

    print("10 folds, "+str(len(test_dataset_init))+" test set, "+str(len(train_dataset_init))+" validation set" )
    
    for runname in runs:
        # Loading model settings from .sh file. This can be removed if you want to pass settings as sys.argv
        # via cmdline or bash script. model_settings.sh can be submitted via slurm using sbatch
        sys.argv = [sys.argv[0]]  # Reset sys.argv to just the script name
        model_dir=cwd+"/pretrained_models/"+runname
        if len(sys.argv) <= 1:
            sh_file = model_dir+"/model_settings.sh"
            if os.path.exists(sh_file):
                sys.argv = [sys.argv[0]] + parse_sh_args(sh_file)
            else:
                raise FileNotFoundError(f"Could not find {sh_file}")
                
        #Parse expected command-line arguments
        args = sys.argv[1:]
        if len(args) < 21:
            raise ValueError(f"Expected 21 command-line arguments but got {len(args)}")
        
        (kfold_iter_str,
         main_dir,
         main_save_dir,
         base_run_name,
         dataset_name,
         fold_indices_file,
         extra_cols_str,
         mode_str,
         model_variant,
         n_epochs_str,
         batch_size_str,
         activ_str,
         lr_str,
         lr_schedule_start_epoch_str,
         add_residual_str,
         gnn_cfg_str,
         gnn_dense_cfg_str,
         dense_zindo_cfg_str,
         dense_fp_cfg_str,
         dense_joint_cfg_str,
         dense_final_cfg_str) = args[:21]
        
    
        (gnn_cfg, dense_cfg, dense_fp_cfg, dense_joint_cfg, dense_final_cfg, mode, add_residual, extra_feat_cols) = load_variables_from_str(args)
        if mode == 'gnn_default':
            x_cutoff = 41
            extra_feat_cols=[]
        elif mode == 'gnn_xyz':
            x_cutoff = 44
        elif mode == 'gnn_4extra_xyz':
            x_cutoff = 48
        elif mode == 'gnn_4extra_noxyz':
            x_cutoff= 45
            
        # Lists to hold metrics for each fold of current model
        loss_list,r_list,pred_list=[],[],[]
        loss_list_val,r_list_val,pred_list_val,true_list_val=[],[],[],[]
        pearson_r_list_val, pearson_r_list = [],[]
        rmse_list_val, mae_list_val = [],[]
        rmse_list, mae_list = [], []
        time_list=[]
        
        for i in range(1,n_kfolds+1):
            print("Fold:",i)
            model_path = model_dir+"/fold_"+str(i)+"/model.pt"
            scaler_path = model_dir+"/fold_"+str(i)+"/scalers.pkl"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                scalers = load_scalers_robust(scaler_path)
            val_dataset=copy.deepcopy(train_dataset_init[kfold_indices[i-1][1]]) # Extract unseen validation data for this fold
            

                
            for data_val in val_dataset:
                x, e_zindo, y = scale_feats(data_val,scalers)
                if len(extra_feat_cols)==0:
                    data_val.x=x[:, 0:41] # Basic default AttentiveFP node parameters
                else:
                    data_val.x=torch.cat([x[:, 0:41], x[:, 44:48]], dim=1) # Remove XYZ columns as they make performance worse
                data_val.e_zindo=e_zindo
                data_val.y=y.squeeze()

            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset),
                                    shuffle=False)
            del val_dataset
            
            try:
                model=None
                if model_variant=="AttentiveFP":

                    gnn_cfg['gnn_in_channels']=x_cutoff
                    model = AttentiveFP(in_channels=gnn_cfg['gnn_in_channels'], hidden_channels=gnn_cfg['gnn_hidden_channels'], out_channels=gnn_cfg['gnn_out_channels'],
                                        edge_dim=gnn_cfg['gnn_edge_dim'], num_layers=gnn_cfg['gnn_num_layers'], num_timesteps=gnn_cfg['gnn_num_timesteps'],
                                        dropout=gnn_cfg['gnn_dropout']).to(device)
                elif model_variant=="AttentiveFP_zindo_A":
                    gnn_dense_cfg=gnn_cfg['gnn_final_dense_cfg']
                    model = AttentiveFP_zindo_A(1, gnn_cfg, gnn_dense_cfg, add_residual).to(device)
                elif model_variant=="dense_one_descriptor":
                    model = dense_one_descriptor(1, dense_joint_cfg).to(device)
                elif model_variant=="dense_multi_descriptor_Z":
                    if mode == 'zindo_morgan': dense_joint_cfg['input_dim']=1+2048
                    if mode == 'zindo_rdf_homo': dense_joint_cfg['input_dim']=1+726
                    if mode == 'zindo_morgan_rdf_homo': dense_joint_cfg['input_dim']=1+2048+726
                    if mode == 'morgan_rdf_homo': dense_joint_cfg['input_dim']=2048+726
                    model = dense_multi_descriptor_Z(1, dense_joint_cfg, add_residual).to(device)
                else:
                    print("No model defined")
                    exit(1)

                state= model.state_dict()
                state_dict = torch.load(model_path)
                model.load_state_dict(state_dict)

                # Applying model to Validation set ##################################################
                data_val = next(iter(val_loader))
                del val_loader
                data_val = data_val.to(device)
                e_zindo_val = data_val.e_zindo.view(-1, 1)  # ensures shape [batch_size, 1]
                    
                model.eval() 
                if model_variant=="AttentiveFP":
                    out_val = model(data_val.x, data_val.edge_index, data_val.edge_attr, data_val.batch)
                    
                if model_variant=="AttentiveFP_zindo_A":
                    out_val = model(data_val.x, data_val.edge_index, data_val.edge_attr, data_val.batch, e_zindo_val)

                if model_variant=="dense_one_descriptor":
                    if mode == 'zindo_only':
                        descriptor_val = e_zindo_val
                    if mode == 'morgan_only':
                        descriptor_val=data_val.morganfp.view(-1, 2048)
                    if mode == 'rdf_homo':
                        descriptor_val=data_val.rdf_homo.view(-1,726)
                    if mode == 'rdf_lumo':
                        descriptor_val=data_val.rdf_lumo.view(-1,726)
                    out_val = model(data_val.batch, descriptor_val)
                    
                if model_variant == "dense_multi_descriptor_Z":
                    if mode == 'zindo_morgan': 
                        descriptor_val=torch.cat((e_zindo_val, data_val.morganfp.view(-1, 2048)), dim=1)
                    if mode == 'zindo_rdf_homo':
                        descriptor_val=torch.cat((e_zindo_val,data_val.rdf_homo.view(-1,726)), dim=1)
                    if mode == 'zindo_morgan_rdf_homo':
                        descriptor_val=torch.cat((e_zindo_val,data_val.morganfp.view(-1, 2048),data_val.rdf_homo.view(-1,726)), dim=1)
                    if mode == 'morgan_rdf_homo':
                        descriptor_val=torch.cat((data_val.morganfp.view(-1, 2048),data_val.rdf_homo.view(-1,726)), dim=1)
                    out_val = model(data_val.batch, e_zindo_val, descriptor_val)
                    
                true_val = scalers["E_tddft"].inverse_transform(data_val.y.cpu().detach().numpy().reshape(-1,1))
                pred_val = scalers["E_tddft"].inverse_transform(out_val.cpu().detach().numpy().reshape(-1,1))
                del data_val, out_val

                r_val = calc_rhoc(pred_val.squeeze(), true_val.squeeze())
                r_pearson_val, _ = pearsonr(true_val.squeeze(), pred_val.squeeze())
                rmse_val = (mean_squared_error(true_val.squeeze(), pred_val.squeeze()))**0.5
                mae_val = mean_absolute_error(true_val.squeeze(), pred_val.squeeze())
                
                # Applying model to Test set ##################################################
                
                test_dataset = copy.deepcopy(test_dataset_init) # Process and scale test set according to fold model
                for data in test_dataset:
                    x, e_zindo, y = scale_feats(data,scalers)
                    if len(extra_feat_cols)==0:
                        data.x=x[:, 0:41] # Basic default AttentiveFP node parameters
                    else:
                        data.x=torch.cat([x[:, 0:41], x[:, 44:48]], dim=1) # Remove XYZ columns as they make performance worse
                    data.e_zindo=e_zindo
                    data.y=y.squeeze()
                test_loader = DataLoader(test_dataset, batch_size=len(test_dataset),
                                        shuffle=False)
                    
                data = next(iter(test_loader))
                del test_loader, test_dataset
                data = data.to(device)
                e_zindo = data.e_zindo.view(-1, 1)  # ensures shape [batch_size, 1]
                
                model.eval() 
                t1=time.time()
                if model_variant=="AttentiveFP":
                    
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    
                if model_variant=="AttentiveFP_zindo_A":
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch, e_zindo)

                if model_variant=="dense_one_descriptor":
                    if mode == 'zindo_only':
                        descriptor=e_zindo
                    if mode == 'morgan_only':
                        descriptor=data.morganfp.view(-1, 2048)
                    if mode == 'rdf_homo':
                        descriptor=data.rdf_homo.view(-1,726)
                    if mode == 'rdf_lumo':
                        descriptor=data.rdf_lumo.view(-1,726)
                    out = model(data.batch, descriptor)

                    
                if model_variant == "dense_multi_descriptor_Z":
                    if mode == 'zindo_morgan': 
                        descriptor=torch.cat((e_zindo, data.morganfp.view(-1, 2048)), dim=1)
                    if mode == 'zindo_rdf_homo':
                        descriptor=torch.cat((e_zindo,data.rdf_homo.view(-1,726)), dim=1)
                    if mode == 'zindo_morgan_rdf_homo':
                        descriptor=torch.cat((e_zindo,data.morganfp.view(-1, 2048),data.rdf_homo.view(-1,726)), dim=1)
                    if mode == 'morgan_rdf_homo':
                        descriptor=torch.cat((data.morganfp.view(-1, 2048),data.rdf_homo.view(-1,726)), dim=1)
                    out = model(data.batch, e_zindo, descriptor)

                t2=time.time()
                del model
                model=None
                true = scalers["E_tddft"].inverse_transform(data.y.cpu().detach().numpy().reshape(-1,1))
                pred = scalers["E_tddft"].inverse_transform(out.cpu().detach().numpy().reshape(-1,1))
                del data, out
                
                r = calc_rhoc(pred.squeeze(), true.squeeze())
                r_pearson, _ = pearsonr(true.squeeze(), pred.squeeze())
                rmse = (mean_squared_error(true.squeeze(), pred.squeeze()))**0.5
                mae = mean_absolute_error(true.squeeze(), pred.squeeze())
                t=t2-t1
                
                
            except Exception as error:
                print("Fold:",i)
                print("\nAn exception occurred:", error)
                print("Exception type:", type(error).__name__)
                exit(1)
    
            r_list.append(r), r_list_val.append(r_val)
            pearson_r_list.append(r_pearson), pearson_r_list_val.append(r_pearson_val)
            pred_list.append(pred), pred_list_val.append(pred_val)
            true_list_val.append(true_val)
            rmse_list.append(rmse), rmse_list_val.append(rmse_val)
            mae_list.append(mae), mae_list_val.append(mae_val)
            time_list.append(t)
            

        del (model, state_dict, state)
        del gnn_cfg, dense_cfg, dense_fp_cfg, dense_joint_cfg, dense_final_cfg, mode, add_residual
        r_mean, r_std = np.mean(r_list), np.std(r_list)
        r_mean_val, r_std_val = np.mean(r_list_val), np.std(r_list_val)
        
        mae_mean, mae_std, val_mae_mean, val_mae_std   = np.mean(mae_list),  np.std(mae_list), np.mean(mae_list_val), np.std(mae_list_val)
        rmse_mean, rmse_std, rmse_mean_val, rmse_std_val = np.mean(rmse_list), np.std(rmse_list), np.mean(rmse_list_val), np.std(rmse_list_val)
        r_mean_pearson, r_std_pearson, r_mean_pearson_val, r_std_pearson_val = np.mean(pearson_r_list), np.std(pearson_r_list), np.mean(pearson_r_list_val), np.std(pearson_r_list_val)
        
        mean_pred = np.mean(pred_list,axis=0).squeeze()
        r_mean_pred,_ = pearsonr(mean_pred, true.squeeze())
        rmse_mean_pred = (mean_squared_error(mean_pred, true.squeeze()))**0.5
        mae_mean_pred = mean_absolute_error(mean_pred, true.squeeze())


        
        #3 sig figs
        print("\n---------------------------------------------\nModel:",runname)
        print(f"Time to predict {len(test_dataset_init)} molecules = {np.mean(time_list):.3f} ± {np.std(time_list):.3f} seconds per fold")
        print(f"Time per molecule =  {(1000*(np.mean(time_list))/len(test_dataset_init)):.3f} milli-seconds")
        
        print("\nCalculate metrics for each fold, then average metrics")
        print(f"Val pearson r     = {r_mean_pearson_val:.3f} ± {r_std_pearson_val:.3f}")
        print(f"Val rmse          = {rmse_mean_val:.3f} ± {rmse_std_val:.3f}")
        print(f"Val mae           = {val_mae_mean:.3f} ± {val_mae_std:.3f}")
        print(f"Test pearson r    = {r_mean_pearson:.3f} ± {r_std_pearson:.3f}")
        print(f"Test rmse         = {rmse_mean:.3f} ± {rmse_std:.3f}")
        print(f"Test mae          = {mae_mean:.3f} ± {mae_std:.3f}")
        
        print("\nAverage fold predictions, then calculate metrics")
        print(f"Test mean pearson r    = {r_mean_pred:.3f}")
        print(f"Test mean rmse         = {rmse_mean_pred:.3f}")
        print(f"Test mean mae          = {mae_mean_pred:.3f}")
        print("---------------------------------------------\n")
        


        xylims=None
        xylims=[[2.5,5.5],[2.5,5.5]]
        title=''
        true_vs_pred_plot_v2(true,np.mean(pred_list,axis=0).squeeze(),np.std(pred_list,axis=0).squeeze(),title=title,
                          show=False,savepath=save_dir,savename=str(len(true))+'_test_pred_plot_'+runname,save=save_plots,extra_str=r"$E_{\mathrm{TDDFT}}$ (eV)",xylims=xylims,
                          table_fontsize=15, table_bbox=(0.02, 0.82, 0.432, 0.168))
        

        
        val_r_mean_list.append(r_mean_val), test_r_mean_list.append(r_mean), val_loss_mean_list.append(rmse_mean_val), test_loss_mean_list.append(rmse_mean)
        val_r_std_list.append(r_std_val), test_r_std_list.append(r_std), val_loss_std_list.append(rmse_std_val), test_loss_std_list.append(rmse_std)
        test_r_from_mean.append(r_mean_pred)
        pearson_test_r_model_list.append(r_mean_pearson),pearson_test_r_val_model_list.append(r_mean_pearson_val)
        pearson_test_r_err_model_list.append(r_mean_pearson),pearson_test_r_err_val_model_list.append(r_std_pearson_val)
        
    summary_df = pd.DataFrame({
        'val rhoc':            val_r_mean_list,
        'val rhoc err':        val_r_std_list,
        'test rhoc':           test_r_mean_list,
        'test rhoc err':       test_r_std_list,
        'val pearson r':       pearson_test_r_val_model_list,
        'val pearson r err':   pearson_test_r_err_val_model_list,
        'test pearson r':      pearson_test_r_model_list,
        'test pearson r err':  pearson_test_r_err_model_list,
        'val rmse':            val_loss_mean_list,
        'val rmse err':        val_loss_std_list,
        'test rmse':           test_loss_mean_list,
        'test rmse err':       test_loss_std_list,
        'test r from mean':    test_r_from_mean,
        'runname':             runs
    })
    out_path = os.path.join(save_dir, csv_name)
    if save_csv is True: summary_df.to_csv(out_path, index=False)

        