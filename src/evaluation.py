"""Direction-separated requests and timestamp-local filtered ranking."""
from bisect import bisect_left
import numpy as np
import torch


def preceding_frames(frames, timestamp, history_len):
    end = bisect_left([f.timestamp for f in frames], timestamp)
    return list(frames[max(0, end-history_len):end])


def known_direction(facts, num_relations, task, inverse=False):
    triples = np.array(facts[:, :3], copy=True)
    if inverse:
        triples = triples[:, [2, 1, 0]]
        triples[:, 1] += num_relations
    if task == 'entity':
        return triples[:, :2].copy(), triples[:, 2].copy()
    return triples[:, [0, 2]].copy(), triples[:, 1].copy()


def filtered_ranks(scores, requests, targets, chunk_size=256, tie_policy='stable_id', answer_requests=None, answer_targets=None):
    # The true score is retained; other answers of the SAME timestamp are removed.
    answers = {}
    key_rows = requests.detach().cpu().tolist()
    all_keys=key_rows if answer_requests is None else answer_requests.detach().cpu().tolist()
    all_targets=targets if answer_targets is None else answer_targets
    for key, target in zip(all_keys, all_targets.detach().cpu().tolist()):
        answers.setdefault(tuple(key), set()).add(target)
    ranks = []
    ids = torch.arange(scores.size(1), device=scores.device)
    for start in range(0, len(scores), chunk_size):
        stop = min(len(scores), start+chunk_size)
        local = scores[start:stop].clone()
        labels = targets[start:stop]
        true = local.gather(1, labels[:, None])
        for row, key in enumerate(key_rows[start:stop]):
            forbidden = list(answers[tuple(key)])
            local[row, forbidden] = -float('inf')
        local.scatter_(1, labels[:, None], true)
        if tie_policy == 'stable_id':
            rank = 1 + (local > true).sum(1) + ((local == true) & (ids[None] < labels[:, None])).sum(1)
        elif tie_policy == 'torch_original':
            order = torch.argsort(local, dim=1, descending=True)
            rank = 1 + (order == labels[:, None]).long().argmax(1)
        else:
            raise ValueError('Unknown evaluation tie policy')
        ranks.append(rank)
    return torch.cat(ranks) if ranks else targets.new_empty(0)


def ranking_metrics(ranks):
    if not len(ranks):
        return dict(mrr=None, hits1=None, hits3=None, hits10=None, facts=0)
    ranks = ranks.float()
    return dict(mrr=float(ranks.reciprocal().mean()), hits1=float((ranks<=1).float().mean()),
                hits3=float((ranks<=3).float().mean()), hits10=float((ranks<=10).float().mean()), facts=len(ranks))
