from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
from torch.nn import GRUCell, Linear, Parameter

from torch_geometric.nn import GATConv, MessagePassing, global_add_pool
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import softmax



class GATEConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__(aggr='add', node_dim=0)

        self.dropout = dropout

        self.att_l = Parameter(torch.empty(1, out_channels))
        self.att_r = Parameter(torch.empty(1, in_channels))

        self.lin1 = Linear(in_channels + edge_dim, out_channels, False)
        self.lin2 = Linear(out_channels, out_channels, False)

        self.bias = Parameter(torch.empty(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.lin1.weight)
        glorot(self.lin2.weight)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        # edge_updater_type: (x: Tensor, edge_attr: Tensor)
        alpha = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        # propagate_type: (x: Tensor, alpha: Tensor)
        out = self.propagate(edge_index, x=x, alpha=alpha)
        out = out + self.bias
        return out

    def edge_update(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor,
                    index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:
        x_j = F.leaky_relu_(self.lin1(torch.cat([x_j, edge_attr], dim=-1)))
        alpha_j = (x_j @ self.att_l.t()).squeeze(-1)
        alpha_i = (x_i @ self.att_r.t()).squeeze(-1)
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu_(alpha)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return alpha

    def message(self, x_j: Tensor, alpha: Tensor) -> Tensor:
        return self.lin2(x_j) * alpha.unsqueeze(-1)



class AttentiveFP(torch.nn.Module):
    r"""
    DEFAULT AttentiveFP MODEL from PyTorch Geometric
    This is the same as 'from torch_geometric.nn.models import AttentiveFP'
    The Attentive FP model for molecular representation learning from the
    `"Pushing the Boundaries of Molecular Representation for Drug Discovery
    with the Graph Attention Mechanism"
    <https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959>`_ paper, based on
    graph attention mechanisms.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Hidden node feature dimensionality.
        out_channels (int): Size of each output sample.
        edge_dim (int): Edge feature dimensionality.
        num_layers (int): Number of GNN layers.
        num_timesteps (int): Number of iterative refinement steps for global
            readout.
        dropout (float, optional): Dropout probability. (default: :obj:`0.0`)

    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        edge_dim: int,
        num_layers: int,
        num_timesteps: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.edge_dim = edge_dim
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.dropout = dropout

        self.lin1 = Linear(in_channels, hidden_channels)

        self.gate_conv = GATEConv(hidden_channels, hidden_channels, edge_dim,
                                  dropout)
        self.gru = GRUCell(hidden_channels, hidden_channels)

        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            conv = GATConv(hidden_channels, hidden_channels, dropout=dropout,
                           add_self_loops=False, negative_slope=0.01)
            self.atom_convs.append(conv)
            self.atom_grus.append(GRUCell(hidden_channels, hidden_channels))

        self.mol_conv = GATConv(hidden_channels, hidden_channels,
                                dropout=dropout, add_self_loops=False,
                                negative_slope=0.01)
        self.mol_conv.explain = False  # Cannot explain global pooling.
        self.mol_gru = GRUCell(hidden_channels, hidden_channels)

        self.lin2 = Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor,
                batch: Tensor) -> Tensor:
        """"""  # noqa: D419
        # Atom Embedding:
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()

        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()

        # Molecule Embedding:
        row = torch.arange(batch.size(0), device=batch.device)
        edge_index = torch.stack([row, batch], dim=0)

        out = global_add_pool(x, batch).relu_()
        for t in range(self.num_timesteps):
            h = F.elu_(self.mol_conv((x, out), edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()

        # Predictor:
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.lin2(out)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')
    

    
def create_dense_block(config_dict):
    """
    Construct a sequential dense (fully-connected) block from a configuration dict object.
    Note, no input validation is performed. The block may include Linear layers,
    optional Batch Normalization, Dropout, and non-linear activation functions,
    depending on the settings in `config_dict`.
    
    E.G.
    gnn_dense_cfg={
        'input_dim':100, # 100 length feature vector
        'neurons': [256, 128, 32, 1], # 3 layer MLP. Output dim of 1
        'activations':[nn.ReLU,nn.ReLU,nn.ReLU,None], # Final layer is linear
        'dropout':[0.3, 0.3, 0.3, 0.0],
        'use_BatchNorm':True}" # Only works for batch_size > 1



    Parameters
    ----------
    config_dict : dict
        Dictionary specifying the block architecture. Must contain:
            - 'input_dim' (int): Input feature dimension.
            - 'neurons' (list[int]): Number of units in each dense layer.
            - 'activations' (list[torch.nn.Module or None]): Activation modules, one per layer. 
              Use None for linear-only layers.
            - 'dropout' (list[float]): Dropout probabilities for each layer (0 disables dropout).
            - 'use_BatchNorm' (bool): Whether to insert BatchNorm1d after each Linear layer.

    Returns
    -------
    layers : torch.nn.ModuleList
        A list of PyTorch modules representing the dense block.
    """
    if type(config_dict) is not dict:
        print("config_dict in create_dense_block is not a valid type")
        exit(1)
    layers = nn.ModuleList()
    in_dim = config_dict['input_dim']  
    for i, out_dim in enumerate(config_dict['neurons']):
        activation=config_dict['activations'][i]
        dropout=config_dict['dropout'][i]
        
        layers.append(nn.Linear(in_dim, out_dim))
        if config_dict['use_BatchNorm'] is True:
            layers.append(nn.BatchNorm1d(out_dim))  # Consider position. BatchNorm improves stability, caution with ReLU/SeLU
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        if activation is not None:
            assert isinstance(activation, nn.Module), "Activation must be a torch.nn.Module."
            layers.append(activation)
        in_dim = out_dim  # Update input dimension for next layer
    return layers

class AttentiveFP_zindo_A(torch.nn.Module):
    """
    Pred = Dense(MPNN(DA, EA), E_S1_ZINDO)
    AttentiveFP-based neural fingerprint with E_ZINDO embedding.

    This model is a hybrid between a graph neural network (AttentiveFP-style) 
    and a dense block that processes both molecular embeddings and E_ZINDO descriptors. 
    It supports an optional residual connection from E_ZINDO inputs.

    Parameters
    ----------
    target_out_channels : int
        Dimension of the final output layer.
    gnn_cfg : dict
        Dictionary specifying GNN hyperparameters:
            - 'gnn_in_channels'     (int): Node feature input dimension.
            - 'gnn_hidden_channels' (int): Hidden feature dimension.
            - 'gnn_out_channels'    (int): Output dimension after GNN layers.
            - 'gnn_edge_dim'        (int): Edge feature dimension.
            - 'gnn_num_layers'      (int): Number of atom-level GNN layers.
            - 'gnn_num_timesteps'   (int): Iterations of molecule-level refinement.
            - 'gnn_dropout'         (float): Dropout probability.
    gnn_dense_cfg : dict
        Configuration dictionary for the post-GNN dense block 
        (see `create_dense_block`).
    add_residual : bool, optional (default=False)
        If True, adds a residual linear connection on EZINDO input.

    Forward Inputs from PyTorch Geoemtric dataset object
    --------------
    x : torch.Tensor
        Node feature matrix of shape [num_nodes, in_channels].
    edge_index : torch.Tensor
        Graph connectivity (2 x num_edges).
    edge_attr : torch.Tensor
        Edge feature matrix of shape [num_edges, edge_dim].
    batch : torch.Tensor
        Batch assignment vector of shape [num_nodes].
    e_zindo : torch.Tensor
        EZINDO feature tensor of shape [batch_size, 1].

    Returns
    -------
    pred : torch.Tensor
        Model predictions of shape [batch_size, target_out_channels].
    """
    def __init__(self,
                 target_out_channels: int,
                 gnn_cfg:dict,
                 gnn_dense_cfg:dict,
                 add_residual:False
                 ):
        super().__init__()
        self.add_residual=add_residual
        # GNN config params
        self.in_channels       = gnn_cfg['gnn_in_channels']
        self.hidden_channels   = gnn_cfg['gnn_hidden_channels']
        self.gnn_out_channels  = gnn_cfg['gnn_out_channels']
        self.edge_dim          = gnn_cfg['gnn_edge_dim']
        self.num_layers        = gnn_cfg['gnn_num_layers']
        self.num_timesteps     = gnn_cfg['gnn_num_timesteps']
        self.dropout           = gnn_cfg['gnn_dropout']
        
        
        self.lin1 = Linear(self.in_channels, self.hidden_channels)
        self.gate_conv = GATEConv(self.hidden_channels, self.hidden_channels, self.edge_dim, self.dropout)
        self.gru = GRUCell(self.hidden_channels, self.hidden_channels)
        self.atom_convs = torch.nn.ModuleList()
        self.atom_grus = torch.nn.ModuleList()
        
        # Atom level embeding
        for _ in range(self.num_layers - 1):
            self.atom_convs.append(GATConv(self.hidden_channels, self.hidden_channels, dropout=self.dropout,
                                           add_self_loops=False, negative_slope=0.01))
            self.atom_grus.append(GRUCell(self.hidden_channels, self.hidden_channels))
            
        
        self.mol_conv = GATConv(self.hidden_channels, self.hidden_channels, dropout=self.dropout,
                                 add_self_loops=False, negative_slope=0.01)
        self.mol_conv.explain = False  # (AttentiveFP specific: no explain for global pooling)
        self.mol_gru = GRUCell(self.hidden_channels, self.hidden_channels)
        
        # gnn_filler puts either a non-linear layer or linear layer between the molecular level embedding
        # and the subsequent Dense block. Or choose none to directly concatentate EZindo with the mol_gru output.
        # I recommend linear layer, it worked best. Note this is hardcoded as a list and not an input option.
        self.gnn_filler=[
            nn.Sequential(Linear(self.hidden_channels, self.gnn_out_channels),nn.ReLU(), nn.Dropout(p=self.dropout)),
            Linear(self.hidden_channels, self.gnn_out_channels),
            None][1]

        gnn_final_layers= create_dense_block(gnn_dense_cfg)
        self.final_gnn_dense= nn.Sequential(*gnn_final_layers)
        self.zindo_res = Linear(1, 1) # Residual connection
        self.reset_parameters()  

    def reset_parameters(self):

        self.lin1.reset_parameters()
        self.gate_conv.reset_parameters()
        self.gru.reset_parameters()
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            conv.reset_parameters()
            gru.reset_parameters()
        self.mol_conv.reset_parameters()
        self.mol_gru.reset_parameters()
        self.zindo_res.reset_parameters()
        
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                batch: torch.Tensor, e_zindo: torch.Tensor) -> torch.Tensor:

        # ---- Original AttentiveFP message passing (atom and bond updates) ----
        # Atom embedding and first graph attention layer
        x = F.leaky_relu_(self.lin1(x))
        h = F.elu_(self.gate_conv(x, edge_index, edge_attr))
        h = F.dropout(h, p=self.dropout, training=self.training)
        x = self.gru(h, x).relu_()
        # Additional GNN layers (atom_convs and atom_grus)
        for conv, gru in zip(self.atom_convs, self.atom_grus):
            h = conv(x, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()
        # ---- Global pooling and iterative refinement (molecule-level embedding) ----
        out = global_add_pool(x, batch)          # sum pool node features to get graph embedding
        out = F.relu(out)                        # activation on pooled embedding
        # Iterative refinement of molecule embedding using attention mechanism
        row = torch.arange(batch.size(0), device=batch.device)  # (for creating mol->atom index)
        mol_edge_index = torch.stack([row, batch], dim=0)       # shape [2, batch_size]
        for t in range(self.num_timesteps):
            # Attention-based update of global embedding
            h = F.elu_(self.mol_conv((x, out), mol_edge_index))
            h = F.dropout(h, p=self.dropout, training=self.training)
            out = self.mol_gru(h, out).relu_()
        # Apply dropout to the final molecule embedding as in original model
        gnn_out = F.dropout(out, p=self.dropout, training=self.training)
        #print(len(gnn_out))
        if self.gnn_filler is not None:
            gnn_out=self.gnn_filler(gnn_out)
        dense_in = torch.cat((e_zindo, gnn_out), dim=1)
        pred=self.final_gnn_dense(dense_in)

        if self.add_residual is True:
            pred= pred + self.zindo_res(e_zindo)

        return pred
    
    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'out_channels={self.out_channels}, '
                f'edge_dim={self.edge_dim}, '
                f'num_layers={self.num_layers}, '
                f'num_timesteps={self.num_timesteps}'
                f')')
    

class dense_multi_descriptor_Z(torch.nn.Module): #################################################
    """
    Dense neural network for multiple descriptors with optional EZINDO residual.

    Processes a concatenated descriptor vector through a configurable dense block.
    Optionally includes a residual linear path for EZINDO features.

    Parameters
    ----------
    target_out_channels : int
        Dimension of the final output layer.
    dense_joint_cfg : dict
        Configuration dictionary for the dense block (see `create_dense_block`).
    add_residual : bool, optional (default=False)
        If True, adds a residual linear connection on EZINDO input.

    Forward Inputs
    --------------
    batch : torch.Tensor
        Batch assignment tensor (unused in this model, but kept for interface consistency).
    e_zindo : torch.Tensor
        EZINDO feature tensor of shape [batch_size, 1].
    descriptor : torch.Tensor
        Input descriptor vector of shape [batch_size, n_features].

    Returns
    -------
    pred : torch.Tensor
        Model predictions of shape [batch_size, target_out_channels].
    """
    def __init__(self,
                 target_out_channels: int,
                 dense_joint_cfg:dict,
                 add_residual:False
                 ):
        super().__init__()
        self.add_residual=add_residual
        
        # ---- Dense block creation ----
        self.zindo_res = Linear(1, 1) # Residual connection
        joint_layers= create_dense_block(dense_joint_cfg)
        self.joint_dense=nn.Sequential(*joint_layers) # convert module list to sequential object for forward
          
        
    def forward(self, batch: torch.Tensor, e_zindo: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        pred=self.joint_dense(descriptor)
        if self.add_residual is True:
            pred=pred+ self.zindo_res(e_zindo)
        return pred 
    
class dense_one_descriptor(torch.nn.Module): #################################################
    """
    Very basic Multi-Layer Perceptron
    Dense neural network for a single descriptor vector.

    Processes a descriptor vector through a configurable dense block.
    Unlike `dense_multi_descriptor_Z`, this class does not support 
    EZINDO residual connections.

    Parameters
    ----------
    target_out_channels : int
        Dimension of the final output layer.
    dense_joint_cfg : dict
        Configuration dictionary for the dense block (see `create_dense_block`).

    Forward Inputs
    --------------
    batch : torch.Tensor
        Batch assignment tensor (unused in this model, but kept for interface consistency).
    descriptor : torch.Tensor
        Input descriptor vector of shape [batch_size, n_features].

    Returns
    -------
    pred : torch.Tensor
        Model predictions of shape [batch_size, target_out_channels].
    """
    def __init__(self,
                 target_out_channels: int,
                 dense_joint_cfg:dict,
                 ):
        super().__init__()
        # ---- Dense block creation ----
        joint_layers= create_dense_block(dense_joint_cfg)
        self.joint_dense=nn.Sequential(*joint_layers) # convert module list to sequential object for forward
    def forward(self, batch: torch.Tensor, descriptor: torch.Tensor) -> torch.Tensor:
        pred=self.joint_dense(descriptor)
        return pred

    
