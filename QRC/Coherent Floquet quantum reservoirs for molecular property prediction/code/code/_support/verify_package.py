#!/usr/bin/env python3
"""Read-only checks of release hashes, paths, manuscript assets and Python syntax.

This program does not import research modules, fit models, rebuild figures,
compile the manuscript, submit cloud tasks, or create an execution directory.
"""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def relative_path(path):
    p = Path(path)
    assert not p.is_absolute() and '..' not in p.parts, path
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true', help='print the report as JSON')
    args = parser.parse_args()
    errors = []
    hashes = {}
    records = rows(ROOT / 'index/MANIFEST_SHA256.csv')
    for row in records:
        try:
            p = ROOT / relative_path(row['path'])
            assert p.is_file() and not p.is_symlink(), row['path']
            assert p.stat().st_size == int(row['bytes']), row['path']
            hashes[row['path']] = sha(p)
            assert hashes[row['path']] == row['sha256'], row['path']
        except (OSError, AssertionError) as e:
            errors.append('Manifest: ' + str(e))
    seen = set()
    for row in rows(ROOT / 'index/RUNTIME_PATHS.csv'):
        try:
            relative_path(row['runtime_path'])
            assert row['runtime_path'] not in seen, row['runtime_path']
            seen.add(row['runtime_path'])
            assert hashes.get(row['release_path']) == row['sha256'], row['release_path']
        except AssertionError as e:
            errors.append('Runtime map: ' + str(e))
    for row in rows(ROOT / 'index/FIGURE_CODE_DATA_INDEX.csv'):
        for field in ('figure_file', 'code', 'data'):
            for rel in row[field].split(';'):
                if rel and not (ROOT / relative_path(rel)).exists():
                    errors.append('Figure index: ' + rel)
    python_count = 0
    for p in (ROOT / 'code').rglob('*.py'):
        python_count += 1
        try:
            ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        except (SyntaxError, UnicodeError) as e:
            errors.append('Python syntax: ' + str(p.relative_to(ROOT)) + ': ' + str(e))
    for p in (ROOT / 'tools').glob('*.py'):
        try:
            ast.parse(p.read_text(), filename=str(p))
        except SyntaxError as e:
            errors.append('Tool syntax: ' + str(e))
    figure_rows = rows(ROOT / 'index/FIGURE_CODE_DATA_INDEX.csv')
    assets = {r['figure_file'] for r in figure_rows}
    if sorted(int(r['figure']) for r in figure_rows) != list(range(1, 11)) or len(assets) != 10:
        errors.append('Expected the ten numbered publication figures.')
    if any(not (ROOT / relative_path(name)).is_file() for name in assets):
        errors.append('A publication figure is missing.')
    for p in list((ROOT / 'index').glob('*.md')):
        if re.search(r'[\u3400-\u9fff]', p.read_text()):
            errors.append('Non-English release documentation: ' + str(p.relative_to(ROOT)))
    declared = set(hashes) | {'index/MANIFEST_SHA256.csv'}
    extra = [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
             and 'runs' not in p.relative_to(ROOT).parts
             and '__pycache__' not in p.parts and not p.name.startswith('.')
             and p.relative_to(ROOT).as_posix() not in declared]
    if extra:
        errors.append('Unlisted release files: ' + repr(extra))
    report = dict(status='passed' if not errors else 'failed', files_checked=len(records),
                  runtime_paths=len(seen), python_sources_parsed=python_count,
                  manuscript_figures=len(assets), experiments_run=0, errors=errors)
    print(json.dumps(report, indent=2))
    raise SystemExit(bool(errors))


if __name__ == '__main__':
    main()
