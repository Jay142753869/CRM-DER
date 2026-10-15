import hashlib
import random
from collections import defaultdict

import numpy as np
import torch


def isolated_module(factory, seed):
    """CPU factory RNG isolation for new layers; instantiate on CPU then model.to(device)."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            # Seed CPU only; do not mutate or initialize CUDA in module construction.
            torch.random.default_generator.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            return factory()
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def build_with_seed(config, seed, **kwargs):
    from .model import BaselineModel
    return isolated_module(lambda: BaselineModel(config, **kwargs), seed)


def alias_groups(model):
    groups = defaultdict(list)
    def visit(module, prefix, ancestors):
        if id(module) in ancestors:
            raise ValueError('cyclic module registration')
        for name, value in module._parameters.items():
            if value is not None:
                groups[id(value)].append(prefix + name)
        for name, value in module._buffers.items():
            if value is not None and name not in module._non_persistent_buffers_set:
                groups[id(value)].append(prefix + name)
        for name, child in module._modules.items():
            if child is not None:
                visit(child, prefix + name + '.', ancestors | {id(module)})
    visit(model, '', set())
    return [names for names in groups.values() if len(names) > 1]


def validate_alias_state(model, state):
    for names in alias_groups(model):
        present = [name for name in names if name in state]
        for name in present[1:]:
            if not torch.equal(state[present[0]], state[name]):
                raise ValueError('conflicting checkpoint values for tied aliases: ' + repr(present))


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(value.shape).encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def align_shared_state(untrained_source, target):
    """Source must be an untrained common-state model, enforced by experiment caller.

    Copies all source keys (parameters AND buffers), retains extra target keys,
    and rejects incompatible axes. It never aliases source/target storage.
    """
    source, destination = untrained_source.state_dict(), target.state_dict()
    for key, value in source.items():
        if key not in destination or destination[key].shape != value.shape or destination[key].dtype != value.dtype:
            raise ValueError('shared state key/shape/dtype mismatch: ' + key)
    validate_alias_state(untrained_source, source)
    merged = {key: value.clone() for key, value in destination.items()}
    for key, value in source.items():
        merged[key] = value.to(destination[key].device).clone()
    target.load_state_dict(merged, strict=True)
    return {'shared_tensors': len(source), 'shared_sha256': state_hash(source),
            'target_extra_keys': sorted(set(destination)-set(source))}


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        torch.cuda.set_rng_state_all(state['torch_cuda'])
