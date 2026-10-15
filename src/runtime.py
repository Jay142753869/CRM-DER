"""Reproducible initialization and complete, explicit training state."""
import os
import random
from pathlib import Path

import numpy as np
import torch

ARCH_VERSION = 'crm-der-v2'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def isolated_init(factory, seed):
    """New CRM projections must not consume the shared backbone RNG stream."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return factory()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_random(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def save_checkpoint(path, model, optimizer, progress, config, data_identity, scheduler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=1, arch_version=ARCH_VERSION, model=model.state_dict(),
                   optimizer=optimizer.state_dict(), progress=progress, config=config,
                   data_identity=data_identity, rng=rng_state(),
                   scheduler=None if scheduler is None else scheduler.state_dict())
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(payload, stream)
    os.replace(str(temporary), str(path))


def load_checkpoint(path, model, optimizer=None, restore_rng=False, expected_data=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Explicit checkpoint does not exist: {}'.format(path))
    payload = torch.load(path, map_location='cpu')
    if payload.get('arch_version') != ARCH_VERSION:
        raise ValueError('Checkpoint architecture version differs from this implementation')
    if expected_data is not None and payload['data_identity'] != expected_data:
        raise ValueError('Checkpoint dataset identity does not match current assets')
    model.load_state_dict(payload['model'], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload['optimizer'])
    if restore_rng:
        restore_random(payload['rng'])
    return payload


def code_hash():
    """Hash of the CRM mechanism sources actually imported, for resume identity."""
    import hashlib
    here = Path(__file__).absolute().parent
    digest = hashlib.sha256()
    for name in ('crm/core.py', 'crm/model.py', 'crm/factory.py', 'crm/frame_groups.py',
                 'backbone/model.py', 'backbone/layers.py', 'trainer.py', 'evaluation.py', 'runtime.py',
                 'experiment.py', 'data.py'):
        path = here / name
        digest.update(name.encode('utf-8'))
        digest.update(path.read_bytes())
    return digest.hexdigest()
