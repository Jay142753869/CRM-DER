from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineConfig:
    num_entities: int
    num_relations: int
    hidden_dim: int = 256
    history_len: int = 7
    num_layers: int = 2
    num_static_relations: int = 0
    num_words: int = 0
    use_static: bool = False
    layer_norm: bool = True
    self_loop: bool = True
    skip_connect: bool = False
    dropout: float = 0.2
    add_query: bool = True
    use_onehot: bool = False
    task: str = 'entity'
    temporal: str = 'TF'
    q_type: str = 'conv'
    profile: str = 'baseline_fixed'

    def __post_init__(self):
        for name in ('num_entities', 'num_relations', 'hidden_dim', 'history_len', 'num_layers'):
            if getattr(self, name) < 1:
                raise ValueError(name + ' must be positive')
        if self.hidden_dim % 4:
            raise ValueError('hidden_dim must be divisible by four attention heads')
        if self.num_words < 0 or self.num_static_relations < 0:
            raise ValueError('static axes must be nonnegative')
        if self.use_static and (self.num_words < 1 or self.num_static_relations < 1):
            raise ValueError('static mode needs explicit word and relation axes')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0,1)')
        for name, allowed in [('task', ('entity', 'relation')),
                              ('temporal', ('TF', 'GRU')),
                              ('q_type', ('conv', 'linear')),
                              ('profile', ('baseline_original', 'baseline_fixed'))]:
            if getattr(self, name) not in allowed:
                raise ValueError(name + ' must be one of ' + repr(allowed))
