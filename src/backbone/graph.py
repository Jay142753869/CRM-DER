import dgl
import torch


def _triples(triples):
    triples = torch.as_tensor(triples)
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError('expected integer [rows,3] triples; timestamps belong outside the graph')
    if triples.dtype not in (torch.int64, torch.int32):
        raise ValueError('triple IDs must be integer')
    return triples.long()


def invert_triples(triples, num_relations):
    triples = _triples(triples)
    if num_relations < 1 or (triples[:, 1] < 0).any() or (triples[:, 1] >= num_relations).any():
        raise ValueError('invert_triples accepts forward relations only; inverse IDs must not be doubled twice')
    inverse = triples[:, [2, 1, 0]].clone()
    inverse[:, 1] += num_relations
    return inverse


def build_graph(num_nodes, num_relations, triples, device='cpu'):
    """Original duplicate-preserving forward + inverse edge order and indegree norm."""
    triples = _triples(triples).to(device)
    if num_nodes < 1:
        raise ValueError('num_nodes must be positive')
    if (triples[:, [0, 2]] < 0).any() or (triples[:, [0, 2]] >= num_nodes).any():
        raise ValueError('entity IDs outside graph axis')
    inverse = invert_triples(triples, num_relations)
    directed = torch.cat([triples, inverse], dim=0)
    graph = dgl.graph((directed[:, 0], directed[:, 2]), num_nodes=num_nodes, device=device)
    graph.ndata['id'] = torch.arange(num_nodes, device=device).view(-1, 1)
    graph.ndata['norm'] = graph.in_degrees().float().clamp(min=1).reciprocal().view(-1, 1)
    graph.edata['type'] = directed[:, 1]
    graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    return graph
