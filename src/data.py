"""Standalone NumPy temporal data contract. No global flags, Torch, DGL or test I/O by default."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import json
import numpy as np


def _readonly(array):
    array.setflags(write=False)
    return array


def read_dictionary(path, *, contiguous=True):
    mapping = {}
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                fields = line.rstrip('\r\n').split('\t')
                # YAGO has birth/death columns after ID; original loader uses
                # the second tab column, not the last whitespace token.
                if len(fields) >= 2:
                    label, raw_id = fields[:2]
                else:
                    label, raw_id = line.strip().rsplit(None, 1)
                identifier = int(raw_id)
            except ValueError as error:
                raise ValueError('{}:{} invalid dictionary row'.format(path, line_no)) from error
            if identifier < 0 or identifier in mapping:
                raise ValueError('{}:{} duplicate/negative ID'.format(path, line_no))
            mapping[identifier] = label
    if contiguous and set(mapping) != set(range(len(mapping))):
        raise ValueError('{}: dictionary IDs must cover [0,count)'.format(path))
    return mapping


def load_rows(path, N, R):
    """Read first four integer columns; fifth column remains uninterpreted.

    Physical duplicate rows/order remain intact. This function does not infer splits.
    """
    if N <= 0 or R <= 0:
        raise ValueError('N and R must be positive')
    # np.loadtxt avoids a Python tuple/list for each of millions of GDELT rows.
    with Path(path).open('r', encoding='utf-8-sig') as handle:
        rows = np.loadtxt(handle, dtype=np.int64, usecols=(0, 1, 2, 3), ndmin=2,
                          converters={column: int for column in range(4)}, comments=None)
    if rows.size == 0:
        rows = np.empty((0, 4), dtype=np.int64)
    if len(rows) and (np.any(rows[:, [0, 2]] < 0) or np.any(rows[:, [0, 2]] >= N)
                      or np.any(rows[:, 1] < 0) or np.any(rows[:, 1] >= R)):
        raise ValueError('{}: entity/forward relation ID out of bounds'.format(path))
    return _readonly(rows)


@dataclass(frozen=True)
class StaticMeta:
    available: bool
    num_nodes: int = 0
    num_words: int = 0
    relation_axis: int = 0
    relation_ids: Tuple[int, ...] = ()
    relation_ids_contiguous: bool = True
    edge_count: int = 0
    reused_entity_edges: int = 0
    word_dictionary_checked: bool = False
    axis_evidence: str = 'unavailable'
    word_dictionary_present: bool = False
    unmapped_word_node_count: int = 0
    word_source_edges: int = 0


def inspect_static(directory, N, *, require_word_mapping=False):
    directory = Path(directory)
    path = directory / 'e-w-graph-llm.txt'
    if not path.is_file():
        return StaticMeta(False)
    with path.open('r', encoding='utf-8-sig') as handle:
        triples = np.loadtxt(handle, dtype=np.int64, ndmin=2,
                            converters={column: int for column in range(3)}, comments=None)
    if triples.size == 0 or triples.shape[1] != 3:
        raise ValueError('static graph must contain three-column edges')
    if np.any(triples < 0):
        raise ValueError('static graph has negative ID')
    word_path = directory / 'word2id_llm.txt'
    present = word_path.is_file()
    words = read_dictionary(word_path, contiguous=False) if present else {}
    valid_targets = set(range(N)) | set(words)
    missing = set(map(int, triples[:, [0, 2]].ravel())) - valid_targets
    checked = present and not missing
    if require_word_mapping and not checked:
        raise ValueError('static graph word label mapping unavailable/incomplete')
    # Words may reuse entity IDs; do not discard those edges. Sparse relation
    # IDs use max+1, not len(unique), so inverse relation offsets cannot collide.
    max_node = max(N - 1, int(triples[:, [0, 2]].max()), max(words, default=-1))
    ids = tuple(map(int, np.unique(triples[:, 1])))
    return StaticMeta(available=True, num_nodes=max_node + 1, num_words=max_node + 1 - N,
                      relation_axis=max(ids) + 1, relation_ids=ids,
                      relation_ids_contiguous=ids == tuple(range(max(ids) + 1)),
                      edge_count=len(triples), reused_entity_edges=int(np.sum(triples[:, 2] < N)),
                      word_dictionary_checked=checked,
                      axis_evidence=('entity_plus_word_dictionary' if checked else
                                     'graph_max_with_partial_word_dictionary' if present else 'graph_max_id_only'),
                      word_dictionary_present=present, unmapped_word_node_count=len(missing),
                      word_source_edges=int(np.sum(triples[:, 0] >= N)))


@dataclass(frozen=True)
class StaticGraph:
    triples: np.ndarray
    meta: StaticMeta


def load_static_graph(directory, N, *, require_word_mapping=False):
    meta = inspect_static(directory, N, require_word_mapping=require_word_mapping)
    if not meta.available:
        raise ValueError('requested static graph is unavailable; choose explicit no-static mode')
    with (Path(directory) / 'e-w-graph-llm.txt').open('r', encoding='utf-8-sig') as handle:
        triples = np.loadtxt(handle, dtype=np.int64, ndmin=2,
                            converters={column: int for column in range(3)}, comments=None)
    return StaticGraph(_readonly(triples), meta)


@dataclass(frozen=True)
class DatasetMeta:
    name: str
    N: int
    R: int
    static: StaticMeta
    time_unit: str = 'native_integer_timestamp_tick'
    physical_time_unit: str = 'unverified; do not interpret as snapshot count/days'
    source: str = 'local self-contained directory'

    @property
    def directed_relations(self):
        return 2 * self.R


@dataclass(frozen=True)
class Frame:
    timestamp: int
    ordinal: int
    facts: np.ndarray
    split: str = ''


def make_frames(rows, split=''):
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError('frames require [rows,4] s/r/o/real_timestamp')
    if not len(rows):
        return ()
    order = np.argsort(rows[:, 3], kind='stable')
    sorted_rows = rows[order]
    timestamps, starts = np.unique(sorted_rows[:, 3], return_index=True)
    ends = np.r_[starts[1:], len(rows)]
    return tuple(Frame(int(timestamp), ordinal, _readonly(sorted_rows[start:end]), split)
                 for ordinal, (timestamp, start, end) in enumerate(zip(timestamps, starts, ends)))


@dataclass(frozen=True)
class Dataset:
    meta: DatasetMeta
    train: Tuple[Frame, ...]
    valid: Tuple[Frame, ...]
    test: Optional[Tuple[Frame, ...]] = None


def load_dataset(data_root, name, *, include_test=False):
    """Only explicit final evaluation may pass include_test=True."""
    if not name or Path(name).name != name or name in ('.', '..') or ':' in name or '\\' in name:
        raise ValueError('dataset name must be a single directory component')
    directory = Path(data_root) / name
    entities = read_dictionary(directory / 'entity2id.txt')
    relations = read_dictionary(directory / 'relation2id.txt')
    with (directory / 'stat.txt').open('r', encoding='utf-8-sig') as handle:
        stat = tuple(map(int, handle.read().split()))
    N, R = len(entities), len(relations)
    if len(stat) < 2 or stat[:2] != (N, R):
        raise ValueError('stat N/R disagree with dictionary axes')
    source = 'local self-contained directory'
    provenance = directory / 'PROVENANCE.json'
    if provenance.is_file():
        with provenance.open('r', encoding='utf-8') as handle:
            source = json.load(handle).get('raw_source', source)
    meta = DatasetMeta(name, N, R, inspect_static(directory, N), source=source)
    train = make_frames(load_rows(directory / 'train.txt', N, R), 'train')
    valid = make_frames(load_rows(directory / 'valid.txt', N, R), 'valid')
    test = make_frames(load_rows(directory / 'test.txt', N, R), 'test') if include_test else None
    return Dataset(meta, train, valid, test)


def with_inverse(rows, R):
    """Accept only forward rows, preventing accidental double augmentation."""
    rows = np.asarray(rows, dtype=np.int64)
    if R <= 0 or rows.ndim != 2 or rows.shape[1] != 4:
        raise ValueError('inverse augmentation requires positive R and [rows,4]')
    if len(rows) and (np.any(rows[:, 1] < 0) or np.any(rows[:, 1] >= R)):
        raise ValueError('input already directed or invalid forward relation')
    inverse = rows[:, [2, 1, 0, 3]].copy()
    inverse[:, 1] += R
    return _readonly(np.concatenate((rows, inverse), axis=0))


@dataclass(frozen=True)
class KnownRequests:
    """pairs are (s,r) for entity task or (s,o) for relation task; never gold."""
    task: str
    timestamp: int
    pairs: np.ndarray

    def __post_init__(self):
        if self.task not in ('entity', 'relation'):
            raise ValueError('unknown task')
        pairs = np.array(self.pairs, dtype=np.int64, copy=True)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or np.any(pairs < 0):
            raise ValueError('known requests must have two nonnegative columns')
        object.__setattr__(self, 'pairs', _readonly(pairs))

    def context_by_subject(self):
        if self.task != 'entity':
            raise ValueError('unknown relations cannot enter entity query context')
        context = {}
        for subject, relation in self.pairs:
            context.setdefault(int(subject), []).append(int(relation))
        return {subject: tuple(relations) for subject, relations in context.items()}


def make_requests(frame, task='entity'):
    if frame.facts.ndim != 2 or frame.facts.shape[1] != 4 or np.any(frame.facts[:, 3] != frame.timestamp):
        raise ValueError('frame facts must agree with declared real timestamp')
    if task == 'entity':
        pairs, targets = frame.facts[:, :2], frame.facts[:, 2]
    elif task == 'relation':
        pairs, targets = frame.facts[:, [0, 2]], frame.facts[:, 1]
    else:
        raise ValueError('task must be entity or relation')
    return KnownRequests(task, frame.timestamp, pairs), _readonly(targets.copy())


class PastIndex:
    """Fixed fact-only index; each read is bounded by real T, not update order.

    Stored frames must be observed facts from authorized splits. Combining train
    and valid supports rolling valid: all reads still filter strictly < T.
    No embeddings/activations are cached here.
    """
    def __init__(self, frames, N, R, *, add_inverse=True, allow_test=False):
        if N <= 0 or R <= 0:
            raise ValueError('positive axes required')
        self.N, self.R = N, R
        supplied = tuple(frames)
        if not allow_test and any(frame.split == 'test' for frame in supplied):
            raise ValueError('test index requires explicit final evaluation allow_test=True')
        for frame in supplied:
            if frame.facts.ndim != 2 or frame.facts.shape[1] != 4 or np.any(frame.facts[:, 3] != frame.timestamp):
                raise ValueError('frame facts disagree with real timestamp')
        raw = np.concatenate([f.facts for f in supplied]) if supplied else np.empty((0, 4), np.int64)
        if len(raw) and (np.any(raw[:, [0, 2]] < 0) or np.any(raw[:, [0, 2]] >= N)):
            raise ValueError('entity index out of bounds')
        self._frames = make_frames(raw)
        self._timestamps = np.array([f.timestamp for f in self._frames], dtype=np.int64)
        self._rows = with_inverse(raw, R) if add_inverse else _readonly(raw.copy())
        if len(raw) and (np.any(raw[:, 1] < 0) or np.any(raw[:, 1] >= R)):
            raise ValueError('PastIndex accepts original forward facts only')
        # Compact ordered occurrence indices avoid millions of Python lists.
        self._orders = {}
        self._keys = {}
        for kind, key in [('out', self._rows[:, 0]), ('in', self._rows[:, 2]),
                          ('sr', self._rows[:, 0] * (2 * R) + self._rows[:, 1])]:
            order = np.lexsort((self._rows[:, 3], key))
            self._orders[kind] = order
            self._keys[kind] = key[order]

    def before(self, timestamp):
        return PastView(self, int(timestamp))

    def _occurrences(self, kind, key, timestamp):
        keys = self._keys[kind]
        lo, hi = np.searchsorted(keys, key, side='left'), np.searchsorted(keys, key, side='right')
        order = self._orders[kind][lo:hi]
        end = np.searchsorted(self._rows[order, 3], timestamp, side='left')
        return self._rows[order[:end]]


@dataclass(frozen=True)
class PastView:
    _index: PastIndex
    timestamp: int

    def _subject(self, subject):
        if not 0 <= subject < self._index.N:
            raise ValueError('subject out of bounds')

    def _sr(self, subject, relation):
        self._subject(subject)
        if not 0 <= relation < 2 * self._index.R:
            raise ValueError('directed relation out of bounds')
        return self._index._occurrences('sr', subject * (2 * self._index.R) + relation, self.timestamp)

    def frames(self, history_len=None):
        end = int(np.searchsorted(self._index._timestamps, self.timestamp, side='left'))
        if history_len is not None and history_len < 0:
            raise ValueError('history_len must be nonnegative')
        start = 0 if history_len is None else max(0, end - history_len)
        return self._index._frames[start:end]

    def outgoing(self, subject):
        """Rows [directed_relation,object,real_timestamp]; duplicates retained."""
        self._subject(subject)
        return _readonly(self._index._occurrences('out', subject, self.timestamp)[:, [1, 2, 3]])

    def incoming(self, receiver):
        """Rows [source,directed_incoming_relation,real_timestamp]."""
        self._subject(receiver)
        return _readonly(self._index._occurrences('in', receiver, self.timestamp)[:, [0, 1, 3]])

    def objects(self, subject, relation):
        return tuple(map(int, np.unique(self._sr(subject, relation)[:, 2])))

    def neighbors(self, subject):
        # Also works without inverse augmentation: interaction means either direction.
        outgoing = self.outgoing(subject)[:, 1]
        incoming = self.incoming(subject)[:, 0]
        return tuple(map(int, np.unique(np.r_[outgoing, incoming])))

    def query_activity(self, subject, relation):
        return tuple(map(int, np.unique(self._sr(subject, relation)[:, 3])))

    def object_activity(self, subject, relation):
        rows = self._sr(subject, relation)
        return {int(obj): tuple(map(int, np.unique(rows[rows[:, 2] == obj, 3])))
                for obj in np.unique(rows[:, 2])}

    def age(self, past_timestamp):
        if past_timestamp >= self.timestamp:
            raise ValueError('age requires a strictly past real timestamp')
        return self.timestamp - int(past_timestamp)
