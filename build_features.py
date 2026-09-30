"""Build one numeric feature row per Home Credit application with DuckDB SQL."""
import argparse
import glob
import json
from pathlib import Path
import re
import time
import duckdb
from common import META, digest, load_json, save_json


def identifier(name):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        raise ValueError(f'Unsafe SQL identifier: {name!r}')
    return '"' + name + '"'


def build(root, config_path, output, split='train', sample_fraction=1.0, fixture=False, memory='2GB'):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new feature-build directory')
    if not 0 < sample_fraction <= 1:
        raise ValueError('sample_fraction must be in (0,1]')
    config = load_json(config_path)
    output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'status': 'running', 'source': config['dataset'], 'split': split,
              'data_kind': 'test-fixture' if fixture else 'home-credit', 'files': [],
              'config_sha256': digest(config_path), 'sample_fraction': sample_fraction,
              'engine': f'DuckDB {duckdb.__version__}', 'tables': {}}
    con = duckdb.connect()
    con.execute('SET memory_limit = ?', [memory])
    con.execute('SET temp_directory = ?', [str(output / 'spill')])
    sql_log = []

    def read_view(pattern, name, required):
        files = sorted(glob.glob(str(root / pattern.format(split=split))))
        if not files:
            raise FileNotFoundError(f'No files found for {pattern}. See DATA_ACCESS.md.')
        for filename in files:
            available = set(con.read_parquet(filename).columns)
            if not set(required).issubset(available):
                raise ValueError(f'{Path(filename).name}: missing {sorted(set(required) - available)}')
            report['files'].append({'name': str(Path(filename).relative_to(root)),
                                    'bytes': Path(filename).stat().st_size,
                                    'sha256': digest(filename)})
        con.read_parquet(files, union_by_name=True).create_view(name)

    def execute(sql):
        sql_log.append(sql)
        return con.execute(sql)

    def unique(view, keys):
        cols = ','.join(identifier(k) for k in keys)
        nulls = ' OR '.join(identifier(k) + ' IS NULL' for k in keys)
        if execute(f'SELECT count(*) FROM {view} WHERE {nulls}').fetchone()[0]:
            raise ValueError(f'Null join key in {view}')
        if execute(f'SELECT count(*) FROM (SELECT {cols} FROM {view} GROUP BY {cols} HAVING count(*)>1)').fetchone()[0]:
            raise ValueError(f'Duplicate key in {view}; check overlapping shards or releases')

    try:
        base_columns = ['case_id', 'WEEK_NUM', 'date_decision'] + (['target'] if split == 'train' else [])
        read_view(config['base'], 'base_raw', base_columns)
        unique('base_raw', ['case_id'])
        if execute('SELECT count(*) FROM base_raw WHERE WEEK_NUM IS NULL OR NOT isfinite(try_cast(WEEK_NUM AS DOUBLE)) OR try_cast(WEEK_NUM AS DOUBLE) IS NULL OR try_cast(WEEK_NUM AS DOUBLE) != floor(try_cast(WEEK_NUM AS DOUBLE))').fetchone()[0]:
            raise ValueError('Invalid base WEEK_NUM')
        if split == 'train' and execute('SELECT count(*) FROM base_raw WHERE target IS NULL OR target NOT IN (0,1)').fetchone()[0]:
            raise ValueError('Invalid target')
        if execute('SELECT count(*) FROM base_raw WHERE try_cast(date_decision AS DATE) IS NULL').fetchone()[0]:
            raise ValueError('Invalid date_decision')
        threshold = int(sample_fraction * 10000)
        execute(f'CREATE TEMP VIEW base AS SELECT * FROM base_raw WHERE hash(case_id) % 10000 < {threshold}')
        n = execute('SELECT count(*) FROM base').fetchone()[0]
        if not n:
            raise ValueError('No applications selected')
        select = ['b.' + identifier(c) for c in base_columns]
        joins, features = [], []
        names = set()
        for table in config['tables']:
            name = table['name']
            identifier(name)
            if name in names or name in {'base', 'base_raw'}:
                raise ValueError('Table names must be unique and not reserved')
            names.add(name)
            numeric = table.get('numeric', [])
            if set(numeric) & META:
                raise ValueError('Identifiers, targets and calendar metadata cannot become predictors')
            for col in numeric:
                identifier(col)
            keys = ['case_id'] if table['kind'] == 'one_to_one' else table['keys']
            if 'case_id' not in keys:
                raise ValueError('History keys must include case_id')
            raw = identifier(name + '_raw')
            read_view(table['pattern'], name + '_raw', keys + numeric)
            unique(raw, keys)
            for col in numeric:
                c = identifier(col)
                if execute(f'SELECT count(*) FROM {raw} WHERE {c} IS NOT NULL AND (try_cast({c} AS DOUBLE) IS NULL OR NOT isfinite(try_cast({c} AS DOUBLE)))').fetchone()[0]:
                    raise ValueError(f'Invalid numeric values in {name}.{col}')
            fields = []
            if table['kind'] == 'one_to_one':
                fields = [f'cast({identifier(c)} AS DOUBLE) AS {identifier(name + "__" + c)}' for c in numeric]
                query = f'SELECT case_id, 1.0 AS {identifier(name + "__present")}' + (',' + ','.join(fields) if fields else '') + f' FROM {raw}'
            elif table['kind'] == 'history':
                fields = [f'{op}(cast({identifier(c)} AS DOUBLE)) AS {identifier(name + "__" + c + "__" + op)}' for c in numeric for op in ('avg', 'max')]
                query = f'SELECT case_id, count(*)::DOUBLE AS {identifier(name + "__records")}, 1.0 AS {identifier(name + "__present")}' + (',' + ','.join(fields) if fields else '') + f' FROM {raw} GROUP BY case_id'
            else:
                raise ValueError('Supported table kinds: one_to_one, history')
            execute(f'CREATE TEMP VIEW {identifier(name)} AS {query}')
            columns = [c for c in con.table(name).columns if c != 'case_id']
            features.extend(columns)
            select.extend([f'coalesce({identifier(name)}.{identifier(c)}, 0.0) AS {identifier(c)}' if c.endswith('__present') else f'{identifier(name)}.{identifier(c)}' for c in columns])
            joins.append(f'LEFT JOIN {identifier(name)} ON b.case_id={identifier(name)}.case_id')
            missing = execute(f'SELECT count(*) FROM base b LEFT JOIN {identifier(name)} t ON b.case_id=t.case_id WHERE t.case_id IS NULL').fetchone()[0]
            report['tables'][name] = {'unmatched_applications': missing, 'missing_join_fraction': missing / n}
        query = 'SELECT ' + ','.join(select) + ' FROM base b ' + ' '.join(joins)
        destination = str(output / 'features.parquet').replace("'", "''")
        execute(f"COPY ({query}) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        report.update(status='complete', rows=n, features=features, elapsed_seconds=round(time.perf_counter()-started, 3),
                      feature_sha256=digest(output / 'features.parquet'))
        (output / 'features_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        return output / 'features.parquet'
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        save_json(output / 'manifest.json', report)
        (output / 'queries.sql').write_text(';\n\n'.join(sql_log) + ';\n', encoding='utf-8')
        con.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--config', default='features.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', choices=['train', 'test'], default='train')
    parser.add_argument('--sample-fraction', type=float, default=1.0)
    parser.add_argument('--memory', default='2GB')
    args = parser.parse_args()
    print(build(args.data, args.config, args.output, args.split, args.sample_fraction, memory=args.memory))
