# -*- coding: utf-8 -*-
'''
Created on Thu Dec 21 14:26:27 2023

@author: adamc
'''


import os, ast, re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.nn as nn
import pickle
import pandas as pd

def load_pkl_data(filepath, slice_tuple=None, pd_dataframe=False):
    if not filepath.endswith('.pkl'):
        filepath += '.pkl'
    #data = None  # Initialize data to ensure it's defined
    if pd_dataframe is False:
        try:
            with open(filepath, 'rb') as load_file:
                if slice_tuple is None:
                    try:
                        data = pickle.load(load_file)
                    except Exception as error:
                        print("An exception occurred during loading:", error)
                else:
                    try:
                        data = pickle.load(load_file)[slice_tuple]
                    except Exception as error:
                        print("An exception occurred during slicing:", error)
        
        except Exception as error:
            print("An exception occurred while loading the pickle file:", error)
            print("Filepath:", filepath)
            print("Filepath type:", type(filepath))
            exit(1)
    else:
        try:
            with open(filepath, 'rb') as load_file:
                if slice_tuple is None:
                    try:
                        data = pd.read_pickle(load_file)
                    except Exception as error:
                        print("An exception occurred during loading:", error)
                        if 'pandas' in str(error):
                            print("Current environment pandas version:", pd.__version__)
                else:
                    try:
                        data = pd.read_pickle(load_file)[slice_tuple]
                    except Exception as error:
                        print("An exception occurred during slicing:", error)
                        if 'pandas' in str(error):
                            print("Current environment pandas version:", pd.__version__)
        
        except Exception as error:
            print("An exception occurred while loading the pickle file:", error)
            if 'pandas' in str(error):
                print("Current environment pandas version:", pd.__version__)
            print("Filepath:", filepath)
            print("Filepath type:", type(filepath))
            exit(1)
    return data


def save_pkl_data(filepath, filename, savedata):
    try:
        if not os.path.exists(filepath): os.makedirs(filepath)
        if filename[-4:]!='.pkl': filename=filename+'.pkl'
        with open(filepath+'/'+filename,'wb') as save_file:
            pickle.dump(savedata, save_file)
    except Exception as error:
        print("An exception occurred:", error)
        print("Exception type:", type(error).__name__)
        print("File cannot be saved")
        exit(1)
    return None

class NumpyCompatUnpickler(pickle.Unpickler):
    """
    A pickle.Unpickler that remaps numpy._core.* → numpy.core.* under the hood.
    """
    def find_class(self, module, name):
        # Redirect any numpy._core references to numpy.core
        if module.startswith("numpy._core"):
            new_mod = module.replace("numpy._core", "numpy.core")
            mod = __import__(new_mod, fromlist=[name])
            return getattr(mod, name)
        # Otherwise default behavior
        return super().find_class(module, name)

def load_scalers_robust(path: str):
    """
    Load a pickled object (e.g. sklearn scalers) even if
    it was written against a newer numpy that stored things under
    numpy._core.* internally.
    """
    with open(path, "rb") as f:
        unpickler = NumpyCompatUnpickler(f)
        return unpickler.load()

def safe_eval(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return eval(s, {'__builtins__': {}}, {'nn': nn})
        except Exception as e:
            raise RuntimeError(f"safe_eval failed to parse: {s!r}\nError: {e}")
            
def safe_eval2(s: str, activ): # Redefine to accept activ
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return eval(s, {'__builtins__': {}}, {'nn': nn, 'activ': activ})
        except Exception as e:
            raise RuntimeError(f"safe_eval failed to parse: {s!r}\nError: {e}")

def parse_sh_args(sh_path):
    """
    Read a bash script that defines VAR=… lines and then a
    python_arguments=( … ) block, and return the list of actual
    argument‐values in order.
    """
    var_map = {}
    args = []
    in_args = False

    with open(sh_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue

            if not in_args:
                if line.startswith('python_arguments'):
                    in_args = True
                    continue

                # capture KEY=VALUE
                m = re.match(r'^(\w+)=(.*)$', line)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    # strip surrounding quotes if present
                    if (val.startswith('"') and val.endswith('"')) or \
                       (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    var_map[key] = val
            else:
                if line.startswith(')'):
                    break

                entry = line.rstrip('\\').strip()
                # if it's quoted:
                if (entry.startswith('"') and entry.endswith('"')) or \
                   (entry.startswith("'") and entry.endswith("'")):
                    inner = entry[1:-1]
                    # **expand** if it was a $VAR
                    if inner.startswith('$'):
                        varname = inner[1:]
                        arg = var_map.get(varname, '')
                    else:
                        arg = inner
                # unquoted $VAR
                elif entry.startswith('$'):
                    varname = entry[1:]
                    arg = var_map.get(varname, '')
                else:
                    arg = entry

                args.append(arg)

    return args

def load_variables_from_str(args):
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
    
    
    # Convert types and evaluate configuration strings
    extra_feat_cols = ast.literal_eval(extra_cols_str)
    activ = safe_eval(activ_str)
    mode = safe_eval(mode_str)
    add_residual = ast.literal_eval(add_residual_str)
    gnn_cfg = safe_eval(gnn_cfg_str)
    dense_cfg  = safe_eval2(dense_zindo_cfg_str, activ)
    dense_fp_cfg  = safe_eval2(dense_fp_cfg_str, activ)
    dense_joint_cfg  = safe_eval2(dense_joint_cfg_str, activ)
    dense_final_cfg  = safe_eval2(dense_final_cfg_str, activ)
    
    # Evaluate gnn_dense_cfg if provided
    if gnn_dense_cfg_str not in ("None", ""):
        ctx = {'__builtins__': {}}
        loc = {'gnn_cfg': gnn_cfg, 'activ': activ}
        try:
            gnn_dense_cfg = eval(gnn_dense_cfg_str, ctx, loc)
        except Exception as e:
            raise RuntimeError(f"couldn't parse gnn_dense_cfg: {e}")
    else:
        gnn_dense_cfg = None
    
    # Integrate gnn_dense_cfg into gnn_cfg if applicable
    if gnn_cfg is not None:
        gnn_cfg['gnn_final_dense_cfg'] = gnn_dense_cfg
    
    # Set output_dim for dense configurations if not present
    if dense_cfg is not None:
        dense_cfg['output_dim'] = dense_cfg.get('output_dim', dense_cfg.get('neurons', [None])[-1])
    if dense_fp_cfg is not None:
        dense_fp_cfg['output_dim'] = dense_fp_cfg.get('output_dim', dense_fp_cfg.get('neurons', [None])[-1])
    if dense_joint_cfg is not None:
        dense_joint_cfg['output_dim'] = dense_joint_cfg.get('output_dim', dense_joint_cfg.get('neurons', [None])[-1])

        
    return (gnn_cfg, dense_cfg, dense_fp_cfg, dense_joint_cfg, dense_final_cfg, mode, add_residual, extra_feat_cols)

    




    

    
    
    

    
