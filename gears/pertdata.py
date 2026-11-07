from torch_geometric.data import Data
import torch
import numpy as np
import pickle
import json
from torch_geometric.data import DataLoader
import os
import pandas as pd
from scipy.sparse import issparse
import scanpy as sc
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
sc.settings.verbosity = 0

from .data_utils import get_DE_genes, get_dropout_non_zero_genes, DataSplitter
from .utils import print_sys, zip_data_download_wrapper, dataverse_download,\
                  filter_pert_in_go, get_genes_from_perts, tar_data_download_wrapper,\
                  file_sha256

class PertData:
    """
    Class for loading and processing perturbation data

    Attributes
    ----------
    data_path: str
        Path to save/load data
    gene_set_path: str
        Path to gene set to use for perturbation graph
    default_pert_graph: bool
        Whether to use default perturbation graph or not
    dataset_name: str
        Name of dataset
    dataset_path: str
        Path to dataset
    adata: AnnData
        AnnData object containing dataset
    dataset_processed: bool
        Whether dataset has been processed or not
    ctrl_adata: AnnData
        AnnData object containing control samples
    gene_names: list
        List of gene names
    node_map: dict
        Dictionary mapping gene names to indices
    split: str
        Split type
    seed: int
        Seed for splitting
    subgroup: str
        Subgroup for splitting
    train_gene_set_size: int
        Number of genes to use for training

    """
    
    def __init__(self, data_path, 
                 gene_set_path=None, 
                 default_pert_graph=True):
        """
        Parameters
        ----------

        data_path: str
            Path to save/load data
        gene_set_path: str
            Path to gene set to use for perturbation graph
        default_pert_graph: bool
            Whether to use default perturbation graph or not

        """

        
        # Dataset/Dataloader attributes
        self.data_path = data_path
        self.default_pert_graph = default_pert_graph
        self.gene_set_path = gene_set_path
        self.dataset_name = None
        self.dataset_path = None
        self.adata = None
        self.dataset_processed = None
        self.ctrl_adata = None
        self.gene_names = []
        self.node_map = {}

        # Frozen split protocol attributes
        self.frozen_split_dir = None
        self.control_rows = None

        # Split attributes
        self.split = None
        self.seed = None
        self.subgroup = None
        self.train_gene_set_size = None

        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        server_path = 'https://dataverse.harvard.edu/api/access/datafile/6153417'
        dataverse_download(server_path,
                           os.path.join(self.data_path, 'gene2go_all.pkl'))
        with open(os.path.join(self.data_path, 'gene2go_all.pkl'), 'rb') as f:
            self.gene2go = pickle.load(f)
    
    def set_pert_genes(self):
        """
        Set the list of genes that can be perturbed and are to be included in 
        perturbation graph
        """
        
        if self.gene_set_path is not None:
            # If gene set specified for perturbation graph, use that
            path_ = self.gene_set_path
            self.default_pert_graph = False
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
            
        elif self.default_pert_graph is False:
            # Use a smaller perturbation graph 
            all_pert_genes = get_genes_from_perts(self.adata.obs['condition'])
            essential_genes = list(self.adata.var['gene_name'].values)
            essential_genes += all_pert_genes
            
        else:
            # Otherwise, use a large set of genes to create perturbation graph
            server_path = 'https://dataverse.harvard.edu/api/access/datafile/6934320'
            path_ = os.path.join(self.data_path,
                                     'essential_all_data_pert_genes.pkl')
            dataverse_download(server_path, path_)
            with open(path_, 'rb') as f:
                essential_genes = pickle.load(f)
    
        gene2go = {i: self.gene2go[i] for i in essential_genes if i in self.gene2go}

        self.pert_names = np.unique(list(gene2go.keys()))
        self.node_map_pert = {x: it for it, x in enumerate(self.pert_names)}
            
    def load(self, data_name = None, data_path = None, frozen_split_dir = None):
        """
        Load existing dataloader
        Use data_name for loading 'norman', 'adamson', 'dixit' datasets
        For other datasets use data_path

        Parameters
        ----------
        data_name: str
            Name of dataset
        data_path: str
            Path to dataset
        frozen_split_dir: str
            Path to a frozen split snapshot (e.g. splits/norman). If given, the
            gene and perturbation order are checked against it, and each cell is
            paired with its saved control instead of a random one

        Returns
        -------
        None

        """
        
        if data_name in ['norman', 'adamson', 'dixit', 
                         'replogle_k562_essential', 
                         'replogle_rpe1_essential']:
            ## load from harvard dataverse
            if data_name == 'norman':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'
            elif data_name == 'adamson':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154417'
            elif data_name == 'dixit':
                url = 'https://dataverse.harvard.edu/api/access/datafile/6154416'
            elif data_name == 'replogle_k562_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458695'
            elif data_name == 'replogle_rpe1_essential':
                ## Note: This is not the complete dataset and has been filtered
                url = 'https://dataverse.harvard.edu/api/access/datafile/7458694'
            data_path = os.path.join(self.data_path, data_name)
            zip_data_download_wrapper(url, data_path, self.data_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)

        elif os.path.exists(data_path):
            adata_path = os.path.join(data_path, 'perturb_processed.h5ad')
            self.adata = sc.read_h5ad(adata_path)
            self.dataset_name = data_path.split('/')[-1]
            self.dataset_path = data_path
        else:
            raise ValueError("data attribute is either norman, adamson, dixit "
                             "replogle_k562 or replogle_rpe1 "
                             "or a path to an h5ad file")
        
        self.set_pert_genes()
        print_sys('These perturbations are not in the GO graph and their '
                  'perturbation can thus not be predicted')
        not_in_go_pert = np.array(self.adata.obs[
                                  self.adata.obs.condition.apply(
                                  lambda x:not filter_pert_in_go(x,
                                        self.pert_names))].condition.unique())
        print_sys(not_in_go_pert)
        
        filter_go = self.adata.obs[self.adata.obs.condition.apply(
                              lambda x: filter_pert_in_go(x, self.pert_names))]
        self.adata = self.adata[filter_go.index.values, :]
        pyg_path = os.path.join(data_path, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        if frozen_split_dir is not None:
            # key the cache by the control assignments, so random-control
            # cell graphs are never reused for the frozen protocol
            control_sha256 = self.load_frozen_protocol(frozen_split_dir)
            dataset_fname = os.path.join(pyg_path, 'cell_graphs_frozen_' +
                                         control_sha256[:16] + '.pkl')
                
        if os.path.isfile(dataset_fname):
            print_sys("Local copy of pyg dataset is detected. Loading...")
            self.dataset_processed = pickle.load(open(dataset_fname, "rb"))        
            print_sys("Done!")
        else:
            self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
            self.gene_names = self.adata.var.gene_name
            
            
            print_sys("Creating pyg object for each cell in the data...")
            self.create_dataset_file()
            print_sys("Saving new dataset pyg object at " + dataset_fname) 
            pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
            print_sys("Done!")
            
    def load_frozen_protocol(self, split_dir):
        """
        Check the loaded data against a frozen split snapshot and read its
        saved control assignments

        Parameters
        ----------
        split_dir: str
            Path to the frozen split snapshot (e.g. splits/norman)

        Returns
        -------
        str
            SHA-256 of the control assignment file

        """
        with open(os.path.join(split_dir, 'freeze.json')) as f:
            frozen_sha256 = json.load(f)['artifact_sha256']

        def read_frozen_csv(name):
            path = os.path.join(split_dir, name)
            if file_sha256(path) != frozen_sha256[name]:
                raise ValueError(name + ' does not match freeze.json')
            # keep gene names such as 'NA' as strings
            return pd.read_csv(path, dtype=str, keep_default_na=False)

        genes = read_frozen_csv('expression_genes.csv')
        if list(self.adata.var['gene_name']) != genes['gene_name'].tolist():
            raise ValueError('Expression gene order differs from expression_genes.csv')

        perts = read_frozen_csv('perturbation_nodes.csv')
        if list(self.pert_names) != perts['gene_name'].tolist():
            raise ValueError('Perturbation gene order differs from perturbation_nodes.csv')

        cells = read_frozen_csv('cells.csv.gz')
        if list(self.adata.obs_names) != cells['cell_id'].tolist() or \
           list(self.adata.obs['condition']) != cells['source_condition'].tolist():
            raise ValueError('Retained cells or their order differ from cells.csv.gz')

        controls = read_frozen_csv('control_assignments.csv.gz')
        if controls['target_cell_id'].tolist() != cells['cell_id'].tolist():
            raise ValueError('Control assignments do not follow the cell order')
        row_of_cell = {c: i for i, c in enumerate(self.adata.obs_names)}
        self.control_rows = np.array([row_of_cell[c] for c in controls['control_cell_id']])

        self.frozen_split_dir = split_dir
        return frozen_sha256['control_assignments.csv.gz']

    def new_data_process(self, dataset_name,
                         adata = None,
                         skip_calc_de = False):
        """
        Process new dataset

        Parameters
        ----------
        dataset_name: str
            Name of dataset
        adata: AnnData object
            AnnData object containing gene expression data
        skip_calc_de: bool
            If True, skip differential expression calculation

        Returns
        -------
        None

        """
        
        if 'condition' not in adata.obs.columns.values:
            raise ValueError("Please specify condition")
        if 'gene_name' not in adata.var.columns.values:
            raise ValueError("Please specify gene name")
        if 'cell_type' not in adata.obs.columns.values:
            raise ValueError("Please specify cell type")
        
        dataset_name = dataset_name.lower()
        self.dataset_name = dataset_name
        save_data_folder = os.path.join(self.data_path, dataset_name)
        
        if not os.path.exists(save_data_folder):
            os.mkdir(save_data_folder)
        self.dataset_path = save_data_folder
        self.adata = get_DE_genes(adata, skip_calc_de)
        if not skip_calc_de:
            self.adata = get_dropout_non_zero_genes(self.adata)
        self.adata.write_h5ad(os.path.join(save_data_folder, 'perturb_processed.h5ad'))
        
        self.set_pert_genes()
        self.ctrl_adata = self.adata[self.adata.obs['condition'] == 'ctrl']
        self.gene_names = self.adata.var.gene_name
        pyg_path = os.path.join(save_data_folder, 'data_pyg')
        if not os.path.exists(pyg_path):
            os.mkdir(pyg_path)
        dataset_fname = os.path.join(pyg_path, 'cell_graphs.pkl')
        print_sys("Creating pyg object for each cell in the data...")
        self.create_dataset_file()
        print_sys("Saving new dataset pyg object at " + dataset_fname) 
        pickle.dump(self.dataset_processed, open(dataset_fname, "wb"))    
        print_sys("Done!")
        
    def prepare_split(self, split = 'simulation', 
                      seed = 1, 
                      train_gene_set_size = 0.75,
                      combo_seen2_train_frac = 0.75,
                      combo_single_split_test_set_fraction = 0.1,
                      test_perts = None,
                      only_test_set_perts = False,
                      test_pert_genes = None,
                      split_dict_path=None):

        """
        Prepare splits for training and testing

        Parameters
        ----------
        split: str
            Type of split to use. Currently, we support 'simulation',
            'simulation_single', 'combo_seen0', 'combo_seen1', 'combo_seen2',
            'single', 'no_test', 'no_split', 'custom'
        seed: int
            Random seed
        train_gene_set_size: float
            Fraction of genes to use for training
        combo_seen2_train_frac: float
            Fraction of combo seen2 perturbations to use for training
        combo_single_split_test_set_fraction: float
            Fraction of combo single perturbations to use for testing
        test_perts: list
            List of perturbations to use for testing
        only_test_set_perts: bool
            If True, only use test set perturbations for testing
        test_pert_genes: list
            List of genes to use for testing
        split_dict_path: str
            Path to dictionary used for custom split. Sample format:
                {'train': [X, Y], 'val': [P, Q], 'test': [Z]}
            Defaults to seed_<seed>/gears_custom_split.pkl of the frozen
            split snapshot when load() was given frozen_split_dir

        Returns
        -------
        None

        """
        available_splits = ['simulation', 'simulation_single', 'combo_seen0',
                            'combo_seen1', 'combo_seen2', 'single', 'no_test',
                            'no_split', 'custom']
        if split not in available_splits:
            raise ValueError('currently, we only support ' + ','.join(available_splits))
        self.split = split
        self.seed = seed
        self.subgroup = None
        
        if split == 'custom':
            if split_dict_path is None and self.frozen_split_dir is not None:
                split_dict_path = os.path.join(self.frozen_split_dir,
                                               'seed_' + str(seed),
                                               'gears_custom_split.pkl')
            try:
                with open(split_dict_path, 'rb') as f:
                    self.set2conditions = pickle.load(f)
            except:
                    raise ValueError('Please set split_dict_path for custom split')
            return
            
        self.train_gene_set_size = train_gene_set_size
        split_folder = os.path.join(self.dataset_path, 'splits')
        if not os.path.exists(split_folder):
            os.mkdir(split_folder)
        split_file = self.dataset_name + '_' + split + '_' + str(seed) + '_' \
                                       +  str(train_gene_set_size) + '.pkl'
        split_path = os.path.join(split_folder, split_file)
        
        if test_perts:
            split_path = split_path[:-4] + '_' + test_perts + '.pkl'
        
        if os.path.exists(split_path):
            print('here1')
            print_sys("Local copy of split is detected. Loading...")
            set2conditions = pickle.load(open(split_path, "rb"))
            if split == 'simulation':
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                subgroup = pickle.load(open(subgroup_path, "rb"))
                self.subgroup = subgroup
        else:
            print_sys("Creating new splits....")
            if test_perts:
                test_perts = test_perts.split('_')
                    
            if split in ['simulation', 'simulation_single']:
                # simulation split
                DS = DataSplitter(self.adata, split_type=split)
                
                adata, subgroup = DS.split_data(train_gene_set_size = train_gene_set_size, 
                                                combo_seen2_train_frac = combo_seen2_train_frac,
                                                seed=seed,
                                                test_perts = test_perts,
                                                only_test_set_perts = only_test_set_perts
                                               )
                subgroup_path = split_path[:-4] + '_subgroup.pkl'
                pickle.dump(subgroup, open(subgroup_path, "wb"))
                self.subgroup = subgroup
                
            elif split[:5] == 'combo':
                # combo perturbation
                split_type = 'combo'
                seen = int(split[-1])

                if test_pert_genes:
                    test_pert_genes = test_pert_genes.split('_')
                
                DS = DataSplitter(self.adata, split_type=split_type, seen=int(seen))
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      test_perts=test_perts,
                                      test_pert_genes=test_pert_genes,
                                      seed=seed)

            elif split == 'single':
                # single perturbation
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(test_size=combo_single_split_test_set_fraction,
                                      seed=seed)

            elif split == 'no_test':
                # no test set
                DS = DataSplitter(self.adata, split_type=split)
                adata = DS.split_data(seed=seed)
            
            elif split == 'no_split':
                # no split
                adata = self.adata
                adata.obs['split'] = 'test'
                 
            set2conditions = dict(adata.obs.groupby('split').agg({'condition':
                                                        lambda x: x}).condition)
            set2conditions = {i: j.unique().tolist() for i,j in set2conditions.items()} 
            pickle.dump(set2conditions, open(split_path, "wb"))
            print_sys("Saving new splits at " + split_path)
            
        self.set2conditions = set2conditions

        if split == 'simulation':
            print_sys('Simulation split test composition:')
            for i,j in subgroup['test_subgroup'].items():
                print_sys(i + ':' + str(len(j)))
        print_sys("Done!")
        
    def get_dataloader(self, batch_size, test_batch_size = None):
        """
        Get dataloaders for training and testing

        Parameters
        ----------
        batch_size: int
            Batch size for training
        test_batch_size: int
            Batch size for testing

        Returns
        -------
        dict
            Dictionary of dataloaders

        """
        if test_batch_size is None:
            test_batch_size = batch_size
            
        self.node_map = {x: it for it, x in enumerate(self.adata.var.gene_name)}
        self.gene_names = self.adata.var.gene_name
       
        # Create cell graphs
        cell_graphs = {}
        if self.split == 'no_split':
            i = 'test'
            cell_graphs[i] = []
            for p in self.set2conditions[i]:
                if p != 'ctrl':
                    cell_graphs[i].extend(self.dataset_processed[p])
                
            print_sys("Creating dataloaders....")
            # Set up dataloaders
            test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)

            print_sys("Dataloaders created...")
            return {'test_loader': test_loader}
        else:
            if self.split =='no_test':
                splits = ['train','val']
            else:
                splits = ['train','val','test']
            for i in splits:
                cell_graphs[i] = []
                for p in self.set2conditions[i]:
                    cell_graphs[i].extend(self.dataset_processed[p])

            print_sys("Creating dataloaders....")
            
            # Set up dataloaders
            train_loader = DataLoader(cell_graphs['train'],
                                batch_size=batch_size, shuffle=True, drop_last = True)
            val_loader = DataLoader(cell_graphs['val'],
                                batch_size=batch_size, shuffle=True)
            
            if self.split !='no_test':
                test_loader = DataLoader(cell_graphs['test'],
                                batch_size=batch_size, shuffle=False)
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader,
                                    'test_loader': test_loader}

            else: 
                self.dataloader =  {'train_loader': train_loader,
                                    'val_loader': val_loader}
            print_sys("Done!")

    def get_pert_idx(self, pert_category):
        """
        Get perturbation index for a given perturbation category

        Parameters
        ----------
        pert_category: str
            Perturbation category

        Returns
        -------
        list
            List of perturbation indices

        """
        try:
            pert_idx = [np.where(p == self.pert_names)[0][0]
                    for p in pert_category.split('+')
                    if p != 'ctrl']
        except:
            print(pert_category)
            pert_idx = None
            
        return pert_idx

    def create_cell_graph(self, X, y, de_idx, pert, pert_idx=None):
        """
        Create a cell graph from a given cell

        Parameters
        ----------
        X: np.ndarray
            Gene expression matrix
        y: np.ndarray
            Label vector
        de_idx: np.ndarray
            DE gene indices
        pert: str
            Perturbation category
        pert_idx: list
            List of perturbation indices

        Returns
        -------
        torch_geometric.data.Data
            Cell graph to be used in dataloader

        """

        feature_mat = torch.Tensor(X).T
        if pert_idx is None:
            pert_idx = [-1]
        return Data(x=feature_mat, pert_idx=pert_idx,
                    y=torch.Tensor(y), de_idx=de_idx, pert=pert)

    def create_cell_graph_dataset(self, split_adata, pert_category,
                                  num_samples=1):
        """
        Combine cell graphs to create a dataset of cell graphs

        Parameters
        ----------
        split_adata: anndata.AnnData
            Annotated data matrix
        pert_category: str
            Perturbation category
        num_samples: int
            Number of samples to create per perturbed cell (i.e. number of
            control cells to map to each perturbed cell)

        Returns
        -------
        list
            List of cell graphs

        """

        if self.control_rows is not None:
            return self._frozen_cell_graph_dataset(split_adata, pert_category)

        num_de_genes = 20
        adata_ = split_adata[split_adata.obs['condition'] == pert_category]
        if 'rank_genes_groups_cov_all' in adata_.uns:
            de_genes = adata_.uns['rank_genes_groups_cov_all']
            de = True
        else:
            de = False
            num_de_genes = 1
        Xs = []
        ys = []

        # When considering a non-control perturbation
        if pert_category != 'ctrl':
            # Get the indices of applied perturbation
            pert_idx = self.get_pert_idx(pert_category)

            # Store list of genes that are most differentially expressed for testing
            pert_de_category = adata_.obs['condition_name'][0]
            if de:
                de_idx = np.where(adata_.var_names.isin(
                np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes
            if self.control_rows is None:
                for cell_z in adata_.X:
                    # Use samples from control as basal expression
                    ctrl_samples = self.ctrl_adata[np.random.randint(0,
                                            len(self.ctrl_adata), num_samples), :]
                    for c in ctrl_samples.X:
                        Xs.append(c)
                        ys.append(cell_z)
            else:
                # Use the saved control of each cell from the frozen protocol
                rows = np.where(split_adata.obs['condition'] == pert_category)[0]
                for row, cell_z in zip(rows, adata_.X):
                    Xs.append(split_adata.X[self.control_rows[row]])
                    ys.append(cell_z)

        # When considering a control perturbation
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            for cell_z in adata_.X:
                Xs.append(cell_z)
                ys.append(cell_z)

        # Create cell graphs
        cell_graphs = []
        for X, y in zip(Xs, ys):
            cell_graphs.append(self.create_cell_graph(X.toarray(),
                                y.toarray(), de_idx, pert_category, pert_idx))

        return cell_graphs

    def _frozen_cell_graph_dataset(self, split_adata, pert_category):
        """
        Cell graphs for one condition under the frozen protocol.

        Slices the condition's cells and their saved controls from the
        expression matrix as whole blocks, instead of reading one AnnData
        view row at a time. The cell order, controls, DE genes and values
        are the same as in the row-by-row loop of create_cell_graph_dataset.
        """
        cache = getattr(self, '_frozen_matrix_cache', None)
        if cache is None or cache[0] is not split_adata:
            X = split_adata.X
            X = X.tocsr() if issparse(X) else np.asarray(X)
            conditions = np.asarray(split_adata.obs['condition'])
            self._frozen_matrix_cache = (split_adata, X, conditions)
        _, X, conditions = self._frozen_matrix_cache

        def dense(block):
            return block.toarray() if issparse(block) else np.asarray(block)

        rows = np.where(conditions == pert_category)[0]
        num_de_genes = 20
        de = 'rank_genes_groups_cov_all' in split_adata.uns
        if not de:
            num_de_genes = 1

        ys = dense(X[rows])
        if pert_category != 'ctrl':
            pert_idx = self.get_pert_idx(pert_category)
            pert_de_category = split_adata.obs['condition_name'].iloc[rows[0]]
            if de:
                de_genes = split_adata.uns['rank_genes_groups_cov_all']
                de_idx = np.where(split_adata.var_names.isin(
                    np.array(de_genes[pert_de_category][:num_de_genes])))[0]
            else:
                de_idx = [-1] * num_de_genes
            xs = dense(X[self.control_rows[rows]])
        else:
            pert_idx = None
            de_idx = [-1] * num_de_genes
            xs = ys

        return [self.create_cell_graph(xs[i:i + 1], ys[i:i + 1], de_idx,
                                       pert_category, pert_idx)
                for i in range(len(rows))]

    def create_dataset_file(self):
        """
        Create dataset file for each perturbation condition
        """
        print_sys("Creating dataset file...")
        self.dataset_processed = {}
        for p in tqdm(self.adata.obs['condition'].unique()):
            self.dataset_processed[p] = self.create_cell_graph_dataset(self.adata, p)
        print_sys("Done!")
