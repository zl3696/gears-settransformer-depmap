"""Pooled single-target GEARS variants; historical model/checkpoint paths are unchanged."""

import torch
from torch import nn

from .depmap import AlignedDepMap
from .model import GEARS_Model


class ReplogleModel(GEARS_Model):
    """Original sum fusion with shared coexpression batching and optional context/prior."""

    def __init__(self, args, variant='A', aligned=None):
        if variant not in ('A', 'B', 'C'):
            raise ValueError('variant must be A, B or C')
        if args.get('pert_agg', 'sum') != 'sum' or args.get('uncertainty', False):
            raise ValueError('Replogle A/B/C use original sum fusion without uncertainty')
        super().__init__(args)
        self.variant = variant
        # Base GEARS keeps graphs as plain attributes. Register them for portable
        # state dictionaries and device moves in this independent model path.
        for name in ('G_coexpress', 'G_coexpress_weight', 'G_sim', 'G_sim_weight'):
            value = getattr(self, name)
            delattr(self, name)
            self.register_buffer(name, value)
        self.args = {k: v for k, v in args.items() if not isinstance(v, torch.Tensor)}
        self.depmap_metadata = None
        if variant in ('B', 'C'):
            with torch.random.fork_rng(devices=[]):
                self.context_embedding = nn.Embedding(2, args['hidden_size'])
                nn.init.zeros_(self.context_embedding.weight)
        if variant == 'C':
            if not isinstance(aligned, AlignedDepMap):
                raise ValueError('Variant C requires validated aligned DepMap features')
            aligned.validate(args['perturbation_genes'])
            self.depmap_metadata = aligned.metadata
            self.register_buffer('depmap_features', torch.as_tensor(aligned.values).clone())
            self.register_buffer('depmap_mask', torch.as_tensor(aligned.mask).clone())
            with torch.random.fork_rng(devices=[]):
                self.depmap_projection = nn.Linear(aligned.values.shape[1], args['hidden_size'])
                nn.init.zeros_(self.depmap_projection.weight)
                nn.init.zeros_(self.depmap_projection.bias)

    def forward(self, control_expression, perturbation_nodes, cell_line_ids):
        """One observed control and one single-target node (-1 for control) per cell."""
        x = control_expression
        if x.ndim != 2 or x.shape[1] != self.num_genes:
            raise ValueError('Control expression must have shape [cells, expression genes]')
        size, device = len(x), x.device
        if perturbation_nodes.shape != (size,) or cell_line_ids.shape != (size,):
            raise ValueError('One perturbation and context identifier is required per cell')
        if torch.any((perturbation_nodes < -1) | (perturbation_nodes >= self.num_perts)):
            raise ValueError('Invalid GO perturbation node index')
        if torch.any((cell_line_ids < 0) | (cell_line_ids > 1)):
            raise ValueError('Invalid K562/RPE1 context index')
        gene_ids = torch.arange(self.num_genes, device=device)
        base = self.emb_trans(self.bn_emb(self.gene_emb(gene_ids.repeat(size))))
        position = self.emb_pos(gene_ids)
        for i, layer in enumerate(self.layers_emb_pos):
            position = layer(position, self.G_coexpress, self.G_coexpress_weight)
            if i + 1 < len(self.layers_emb_pos):
                position = position.relu()
        base = self.emb_trans_v2(base + .2 * position.repeat(size, 1))
        perturbation = self.pert_emb(torch.arange(self.num_perts, device=device))
        for i, layer in enumerate(self.sim_layers):
            perturbation = layer(perturbation, self.G_sim, self.G_sim_weight)
            if i + 1 < len(self.sim_layers):
                perturbation = perturbation.relu()
        samples = torch.nonzero(perturbation_nodes >= 0).flatten()
        base = base.reshape(size, self.num_genes, -1)
        if len(samples):
            if self.variant == 'C':
                selected = perturbation_nodes[samples].unique()
                correction = self.depmap_projection(self.depmap_features[selected])
                correction = correction * self.depmap_mask[selected, None]
                perturbation = perturbation.index_add(0, selected, correction)
            selected = perturbation[perturbation_nodes[samples]]
            # Preserve original GEARS' singleton workaround for BatchNorm.
            fused = self.pert_fuse(selected if len(samples) > 1 else selected.repeat(2, 1))[:len(samples)]
            if self.variant in ('B', 'C'):
                fused = fused + self.context_embedding(cell_line_ids[samples])
            base = base.index_add(0, samples, fused[:, None, :].expand(-1, self.num_genes, -1))
        base = self.transform(self.bn_pert_base(base.reshape(size * self.num_genes, -1)))
        decoded = self.recovery_w(base).reshape(size, self.num_genes, -1)
        individual = (decoded.unsqueeze(-1) * self.indv_w1).sum(2) + self.indv_b1
        cross = self.cross_gene_state(individual.squeeze(-1))[:, None, :].expand(-1, self.num_genes, -1)
        combined = torch.cat((individual, cross), dim=2)
        return (combined * self.indv_w2).sum(2) + self.indv_b2 + x


def restore_model(payload, device='cpu', expression_genes=None, perturbation_genes=None):
    """Restore a self-contained model, checking ordered axes and fixed priors."""
    config, state = payload['model_config'], payload['state_dict']
    if expression_genes is not None and expression_genes != config['expression_genes']:
        raise ValueError('Checkpoint expression gene order differs')
    if perturbation_genes is not None and perturbation_genes != config['perturbation_genes']:
        raise ValueError('Checkpoint perturbation gene order differs')
    args = dict(config, device='cpu', G_coexpress=state['G_coexpress'],
                G_coexpress_weight=state['G_coexpress_weight'], G_go=state['G_sim'],
                G_go_weight=state['G_sim_weight'])
    aligned = None
    if payload['variant'] == 'C':
        aligned = AlignedDepMap(state['depmap_features'].cpu().numpy(),
                               state['depmap_mask'].cpu().numpy(), payload['depmap_metadata'])
        aligned.validate(config['perturbation_genes'])
    model = ReplogleModel(args, payload['variant'], aligned)
    model.load_state_dict(state, strict=True)
    return model.to(device)
