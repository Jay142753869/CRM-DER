# -*- coding: utf-8 -*-
"""重建/校验 docs/PACKAGED_SOURCE_SHA256.json（交付源码哈希清单）。

用法（在 CRM-DER 根目录）：
    python docs/build_packaged_hashes.py            # 重写清单
    python docs/build_packaged_hashes.py --check    # 只校验，不一致时退出码 1

覆盖范围与历史清单一致：入口脚本 + src/ 全部模块 + tests/test_standalone.py。
不含 docs/、data/、models/、result/，因此更新文档不会让清单过期。

背景：原清单在 opt.py 与 tests/test_standalone.py 按数据集窗口改动之后没有
重新生成，导致清单与实际源码不符。手工维护的清单会持续漂移，故改为脚本生成。
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]
MANIFEST = ROOT / 'docs' / 'PACKAGED_SOURCE_SHA256.json'

# 交付源码范围（保持与历史清单相同的顺序与条目）
SOURCES = [
    'main.py',
    'opt.py',
    'verify_project.py',
    'src/__init__.py',
    'src/backbone/__init__.py',
    'src/backbone/config.py',
    'src/backbone/decoder.py',
    'src/backbone/encoder.py',
    'src/backbone/graph.py',
    'src/backbone/initialization.py',
    'src/backbone/layers.py',
    'src/backbone/model.py',
    'src/backbone/query.py',
    'src/c2/__init__.py',
    'src/c2/history_selector.py',
    'src/crm/__init__.py',
    'src/crm/core.py',
    'src/crm/factory.py',
    'src/crm/frame_groups.py',
    'src/crm/model.py',
    'src/data.py',
    'src/evaluation.py',
    'src/experiment.py',
    'src/runtime.py',
    'src/trainer.py',
    'tests/test_standalone.py',
]


def digest(relative):
    path = ROOT / relative
    if not path.exists():
        raise FileNotFoundError(relative)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    return {name: digest(name) for name in SOURCES}


def main():
    check = '--check' in sys.argv[1:]
    current = build()

    if check:
        if not MANIFEST.exists():
            print('FAIL: manifest missing')
            return 1
        recorded = json.loads(MANIFEST.read_text(encoding='utf-8'))
        problems = []
        for name in SOURCES:
            if name not in recorded:
                problems.append('missing entry: ' + name)
            elif recorded[name] != current[name]:
                problems.append('stale: %s\n  recorded %s\n  actual   %s'
                                % (name, recorded[name], current[name]))
        for name in recorded:
            if name not in SOURCES:
                problems.append('unexpected entry: ' + name)
        if problems:
            print('\n'.join(problems))
            print('FAIL: manifest does not match the working tree')
            return 1
        print('OK: %d entries match' % len(current))
        return 0

    MANIFEST.write_text(json.dumps(current, indent=2) + '\n', encoding='utf-8')
    print('wrote %s (%d entries)' % (MANIFEST.relative_to(ROOT), len(current)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
