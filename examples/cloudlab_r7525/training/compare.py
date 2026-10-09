#!/usr/bin/env python3
"""Compares the runs of 09.training_with_nic_failure.sh with the upstream-NCCL run of the same seed.

usage: compare.py <out dir> <seed> [<seed> ...]

Reads <out>/<COND>_<SEED>/{train_log.csv,summary.json} for COND in VNF, NF, BALF, ARF and prints one table per seed,
the test perplexity table (mean over the seeds) and the checks. The exit code is 1 if a run is missing or a check
fails.
"""
import csv, json, math, sys
from pathlib import Path

CONDITIONS = ['VNF', 'NF', 'BALF', 'ARF']
LABEL = {'VNF': 'NCCL 2.23.4, no failure', 'NF': 'R2CC, no failure',
         'BALF': 'R2CC-Balance, failure', 'ARF': 'R2CC-AllReduce, failure'}
FP32_E_REL = 1e-6   # FP32 rounding gives about 6e-8


def load(out, name):
    d = out / name
    if not (d / 'summary.json').exists() or not (d / 'train_log.csv').exists():
        return None
    with open(d / 'train_log.csv') as f:
        rows = list(csv.DictReader(f))
    sm = json.loads((d / 'summary.json').read_text())
    return dict(loss=[float(r['train_loss']) for r in rows], rx=[float(r['mlx5_2_rx_mb']) for r in rows], summary=sm,
                grad={c['update']: c['out_digest'] for c in sm['captures']},
                params={p['update']: p['digest'] for p in sm['param_digests']})


def failure_update(run):
    """First update at or after the disconnect command in which mlx5_2 carried less than 20% of its usual traffic."""
    cmd = run['summary']['fail_at_step']
    if not cmd:
        return None
    before = sorted(run['rx'][19:cmd - 1])
    normal = before[len(before) // 2]
    for update in range(cmd, len(run['rx']) + 1):
        if run['rx'][update - 1] < 0.2 * normal:
            return update
    return None


def leading(updates, same):
    """'updates a-b' for the leading updates in which same holds."""
    n = next((i for i, s in enumerate(same) if not s), len(same))
    return f'updates {updates[0]}-{updates[n - 1]}' if n else 'none'


def main():
    out, seeds = Path(sys.argv[1]), [int(s) for s in sys.argv[2:]]
    ok, checks, ppl, delta = True, [], {c: [] for c in CONDITIONS}, {c: [] for c in CONDITIONS}
    window = ''
    for seed in seeds:
        runs = {c: load(out, f'{c}_{seed}') for c in CONDITIONS}
        missing = [c for c in CONDITIONS if runs[c] is None]
        if missing:
            print(f'seed {seed}: missing runs {", ".join(missing)}')
            ok = False
            continue
        base = runs['VNF']
        caps = sorted(base['grad'])
        window = f'updates {caps[0]}-{caps[-1]}' if caps else 'no updates'
        nf_same, lost, hit_same, loss_prefix, reduce_ok, params_ok = False, [], [], [], [], []
        print(f'===== seed {seed}: every run against VNF ({LABEL["VNF"]}) =====')
        fmt = '{:<5} {:<11} {:<17} {:<15} {:>11} {:>8} {:>9} {:>8}'
        print(fmt.format('run', 'NIC down', 'identical to VNF:', '', 'max', 'max', 'test PPL', 'vs VNF'))
        print(fmt.format('', 'from', 'reduced gradient', 'training loss', '|loss-VNF|', 'E_rel', '', '').rstrip())
        for c in CONDITIONS:
            run, sm = runs[c], runs[c]['summary']
            eff = failure_update(run)
            same_loss = [a == b for a, b in zip(run['loss'], base['loss'])]
            same_grad = [run['grad'].get(u) == base['grad'][u] for u in caps]
            e_rel = max((x['e_rel'] for x in sm['captures']), default=float('nan'))
            rel = math.exp(sm['test_nll'] - base['summary']['test_nll']) - 1
            ppl[c].append(sm['test_ppl'])
            delta[c].append(rel)
            if c == 'VNF':
                cells = ['-', '-', '-']
            else:
                cells = [leading(caps, same_grad), leading(list(range(1, len(same_loss) + 1)), same_loss),
                         f'{max(abs(a - b) for a, b in zip(run["loss"], base["loss"])):.2e}'.replace('0.00e+00', '0')]
            print(fmt.format(c, f'update {eff}' if eff else '-', *cells, f'{e_rel:.1e}', f'{sm["test_ppl"]:.4f}',
                             '-' if c == 'VNF' else f'{100 * rel:+.3f}%'))

            if c == 'NF':
                nf_same = (all(same_loss) and len(run['loss']) == len(base['loss'])
                           and sm['test_nll'] == base['summary']['test_nll'] and run['grad'] == base['grad']
                           and run['params'] == base['params'])
            if c in ('BALF', 'ARF'):
                lost.append(eff is not None)
                hit_same.append(eff is not None and eff in base['grad'] and run['grad'].get(eff) == base['grad'][eff])
                loss_prefix.append(eff is not None and all(same_loss[:eff]))
            reduce_ok.append(bool(sm['captures']) and all(x['out_digests_agree'] and x['e_rel'] < FP32_E_REL
                                                          for x in sm['captures']))
            params_ok.append(bool(sm['param_digests']) and all(x['agree'] for x in sm['param_digests']))
        print(f'reduced gradient: SHA-256 of the output of the gradient AllReduce, compared for {window}')
        print(f'E_rel: |AllReduce output - FP64 sum of the inputs| / |FP64 sum|, largest over {window}')
        print()
        checks += [(f'seed {seed}: NF is bit-identical to VNF: training loss of every update, test loss, '
                    f'reduced gradients, parameters', nf_same),
                   (f'seed {seed}: BALF and ARF lost mlx5_2 during the run', all(lost)),
                   (f'seed {seed}: BALF and ARF: the AllReduce that the failure hit gives the same reduced gradient '
                    f'as VNF', all(hit_same)),
                   (f'seed {seed}: BALF and ARF: the training loss is identical to VNF up to the failure',
                    all(loss_prefix)),
                   (f'seed {seed}: all runs: every AllReduce of {window} is identical on all ranks and has '
                    f'E_rel < {FP32_E_REL:g}', all(reduce_ok)),
                   (f'seed {seed}: all runs: the parameters are identical on all ranks at every check point',
                    all(params_ok))]

    done = [s for s in seeds if all(load(out, f'{c}_{s}') for c in CONDITIONS)]
    if done:
        which = f'seed {done[0]}' if len(done) == 1 else f'mean of seeds {", ".join(map(str, done))}'
        print(f'===== Training quality: GPT-2 (124M), WikiText-103, mlx5_2 of node-1 cut at update 400 ({which}) =====')
        print(f'{"Condition":<26} {"Test PPL":>8}   Max. paired Δ*')
        for c in CONDITIONS:
            worst = max(delta[c])
            cell = '--' if c == 'VNF' else ('0.000%' if worst == 0 else f'{100 * worst:+.3f}%')
            print(f'{LABEL[c]:<26} {sum(ppl[c]) / len(ppl[c]):>8.3f}   {cell}')
        print('* the increase of the test perplexity over the NCCL run with the same seed, largest over the seeds')
        print()

    print('===== checks =====')
    for text, passed in checks:
        print(f'{"yes" if passed else "NO ":<4} {text}')
        ok = ok and passed
    print('RESULT: ' + ('all checks passed' if ok else 'at least one check failed'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
