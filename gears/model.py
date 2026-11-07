import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU

from torch_geometric.nn import SGConv

class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)


class MultiheadAttentionBlock(torch.nn.Module):

    def __init__(self, query_size, key_size, output_size, num_heads, layer_norm=True):
        """
        Multihead attention block (Lee et al., 2019)
        :param query_size: dimension of the queries
        :param key_size: dimension of the keys and values
        :param output_size: output dimension
        :param num_heads: number of attention heads
        :param layer_norm: whether to use layer normalization
        """
        super(MultiheadAttentionBlock, self).__init__()
        assert output_size % num_heads == 0, 'output_size must be divisible by num_heads'
        self.output_size = output_size
        self.num_heads = num_heads
        self.query_linear = nn.Linear(query_size, output_size)
        self.key_linear = nn.Linear(key_size, output_size)
        self.value_linear = nn.Linear(key_size, output_size)
        self.output_linear = nn.Linear(output_size, output_size)
        self.norm_attention = nn.LayerNorm(output_size) if layer_norm else nn.Identity()
        self.norm_output = nn.LayerNorm(output_size) if layer_norm else nn.Identity()

    def forward(self, queries, keys, mask=None):
        """
        :param queries: [batch, num_queries, query_size]
        :param keys: [batch, num_keys, key_size]
        :param mask: [batch, num_keys] bool, True for real elements, False for padding
        """
        batch_size = queries.shape[0]
        head_size = self.output_size // self.num_heads

        queries = self.query_linear(queries)
        values = self.value_linear(keys)
        keys = self.key_linear(keys)

        # split into heads: [batch, num_heads, set_size, head_size]
        def split_heads(tensor):
            return tensor.view(batch_size, -1, self.num_heads, head_size).transpose(1, 2)
        queries_heads = split_heads(queries)
        keys_heads = split_heads(keys)
        values_heads = split_heads(values)

        scores = queries_heads @ keys_heads.transpose(-2, -1) / head_size ** 0.5
        if mask is not None:
            # padded keys get zero attention weight
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        attention = torch.softmax(scores, dim=-1)

        output = queries_heads + attention @ values_heads
        output = output.transpose(1, 2).reshape(batch_size, -1, self.output_size)
        output = self.norm_attention(output)
        output = output + F.relu(self.output_linear(output))
        output = self.norm_output(output)
        return output


class SetAttentionBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, num_heads, layer_norm=True):
        """
        Set attention block: self-attention among the elements of a set
        """
        super(SetAttentionBlock, self).__init__()
        self.attention_block = MultiheadAttentionBlock(input_size, input_size, output_size,
                                                       num_heads, layer_norm=layer_norm)

    def forward(self, elements, mask=None):
        return self.attention_block(elements, elements, mask)


class PoolingByMultiheadAttention(torch.nn.Module):

    def __init__(self, hidden_size, num_heads, num_seeds=1, layer_norm=True):
        """
        Pooling by multihead attention: learnable seed vectors attend over the set
        """
        super(PoolingByMultiheadAttention, self).__init__()
        self.seeds = nn.Parameter(torch.Tensor(1, num_seeds, hidden_size))
        nn.init.xavier_uniform_(self.seeds)
        self.attention_block = MultiheadAttentionBlock(hidden_size, hidden_size, hidden_size,
                                                       num_heads, layer_norm=layer_norm)

    def forward(self, elements, mask=None):
        seeds = self.seeds.repeat(elements.shape[0], 1, 1)
        return self.attention_block(seeds, elements, mask)


class GEARS_Model(torch.nn.Module):
    """
    GEARS model

    """

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(GEARS_Model, self).__init__()
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.pert_emb_lambda = 0.2
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')

        # perturbation set aggregation: 'sum' (original GEARS) or 'set_transformer'
        # built on a forked RNG so that, with the same seed, both variants get the
        # same init for all shared layers and the same training batch order
        self.pert_agg = args.get('pert_agg', 'sum')
        with torch.random.fork_rng(devices=[]):
            if self.pert_agg == 'set_transformer':
                self.set_attention = SetAttentionBlock(hidden_size, hidden_size, num_heads=4)
                self.set_pooling = PoolingByMultiheadAttention(hidden_size, num_heads=4)
            elif self.pert_agg == 'sum':
                self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
            else:
                raise ValueError("pert_agg must be 'sum' or 'set_transformer'")
        
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
    def forward(self, data):
        """
        Forward pass of the model
        """
        x, pert_idx = data.x, data.pert_idx
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(data.batch.unique())

            ## get base gene embeddings
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        

            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        

            ## augment global perturbation embedding with GNN
            for idx, layer in enumerate(self.sim_layers):
                pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                if idx < self.num_layers - 1:
                    pert_global_emb = pert_global_emb.relu()

            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

            if pert_index.shape[0] != 0 and self.pert_agg == 'set_transformer':
                ### group perturbed genes by sample, then pad to [num_perturbed_samples, max_set_size, hidden]
                pert_sets = {}
                for sample, pert in zip(pert_index[0].tolist(), pert_index[1].tolist()):
                    pert_sets.setdefault(sample, []).append(pert)

                samples = list(pert_sets.keys())
                max_set_size = max(len(perts) for perts in pert_sets.values())
                padded_perts = torch.zeros(len(samples), max_set_size, dtype=torch.long)
                mask = torch.zeros(len(samples), max_set_size, dtype=torch.bool)
                for row, sample in enumerate(samples):
                    perts = pert_sets[sample]
                    padded_perts[row, :len(perts)] = torch.tensor(perts)
                    mask[row, :len(perts)] = True
                padded_perts = padded_perts.to(self.args['device'])
                mask = mask.to(self.args['device'])

                pert_set_emb = pert_global_emb[padded_perts]
                pert_set_emb = self.set_attention(pert_set_emb, mask)
                emb_total = self.set_pooling(pert_set_emb, mask).squeeze(1)

                for row, sample in enumerate(samples):
                    base_emb[sample] = base_emb[sample] + emb_total[row]

            elif pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            
            return torch.stack(out)
        
