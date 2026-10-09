#!/usr/bin/env python3
"""GPT-2 (124M) training on WikiText-103 for 09.training_with_nic_failure.sh.

One run = 1000 optimizer updates of data-parallel training on six ranks (three servers with two GPUs each), a full
validation every 100 updates and one full test evaluation at the end. With --fail-at-step N, rank 0 cuts node-1's
mlx5_2 on the SmartNIC (nic/disconnect_nic1.sh) at update N and leaves it down until the run ends.

Every run registers the same pass-through DDP comm hook. Inside the capture window (updates 400-408) the hook keeps
the input and the output of the gradient AllReduce; after training, the inputs of all ranks are gathered to rank 0
over Gloo and summed in FP64 in rank order, which gives the error (E_max, E_rel) of the AllReduces around the failure.
Rank 0 writes train_log.csv and summary.json to <out>/<name>; training/compare.py reads them.
"""
import argparse, atexit, csv, datetime, hashlib, json, math, os, random, signal, socket, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

NIC_DIR = Path(__file__).resolve().parent.parent / 'nic'
RX_COUNTER = Path('/sys/class/infiniband/mlx5_2/ports/1/counters/port_rcv_data')
BLOCK, VOCAB, DIM, HEADS, LAYERS = 1024, 50304, 768, 12, 12
disconnect_process, restore_required = None, False


def restore_nic():
    global restore_required
    if not restore_required:
        return
    if disconnect_process is not None:
        try:
            disconnect_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(disconnect_process.pid, signal.SIGKILL)
            disconnect_process.wait()
    subprocess.run([str(NIC_DIR / 'connect_nic1.sh')], check=False, timeout=60,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    restore_required = False
    print('NIC_RESTORED ' + json.dumps(dict(wall_time_unix_s=time.time())), flush=True)


atexit.register(restore_nic)


class Attention(nn.Module):
    def __init__(self, dim, heads, block):
        super().__init__()
        self.heads = heads
        self.c_attn = nn.Linear(dim, 3 * dim)
        self.c_proj = nn.Linear(dim, dim)
        self.register_buffer('mask', torch.tril(torch.ones(block, block, dtype=torch.bool)).view(1, 1, block, block), persistent=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.c_attn(x).split(c, dim=2)
        q, k, v = [a.view(b, t, self.heads, c // self.heads).transpose(1, 2) for a in (q, k, v)]
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(~self.mask[:, :, :t, :t], float('-inf'))
        att = F.softmax(att, dim=-1)
        return self.c_proj((att @ v).transpose(1, 2).contiguous().view(b, t, c))


class Block(nn.Module):
    def __init__(self, dim, heads, block):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim); self.attn = Attention(dim, heads, block)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = nn.ModuleDict(dict(c_fc=nn.Linear(dim, 4 * dim), c_proj=nn.Linear(4 * dim, dim)))

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp.c_proj(F.gelu(self.mlp.c_fc(self.ln_2(x)), approximate='tanh'))


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOCAB, DIM); self.wpe = nn.Embedding(BLOCK, DIM)
        self.blocks = nn.ModuleList([Block(DIM, HEADS, BLOCK) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(DIM); self.lm_head = nn.Linear(DIM, VOCAB, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init)
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * LAYERS))

    def _init(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, y=None):
        h = self.wte(x) + self.wpe(torch.arange(x.size(1), device=x.device))
        for b in self.blocks:
            h = b(h)
        logits = self.lm_head(self.ln_f(h))
        if y is None:
            return logits
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


def sliding_windows(n_tokens, context=BLOCK, stride=512):
    """HF fixed-length-model protocol: every target token is scored exactly once; total = n_tokens - 1."""
    windows, prev_end, begin = [], 0, 0
    while True:
        end = min(begin + context + 1, n_tokens)
        trg = end - max(prev_end, begin + 1)
        if trg > 0:
            windows.append((begin, end, trg))
        prev_end = end
        if end == n_tokens:
            break
        begin += stride
    return windows


@torch.no_grad()
def evaluate(model, data, windows, rank, world, local, gloo):
    """FP32, no autocast, micro-batch 1, unwrapped model. Rank r takes windows where id % world == r."""
    model.eval()
    mine = []
    for wid, (begin, end, trg) in enumerate(windows):
        if wid % world != rank:
            continue
        ids = torch.from_numpy(data[begin:end].astype(np.int64)).to(local, non_blocking=True)
        x, y = ids[:-1].unsqueeze(0), ids[1:].clone().unsqueeze(0)
        y[0, :-trg] = -100
        logits = model(x)
        nll = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100, reduction='sum')
        n = int((y != -100).sum().item())
        assert n == trg, (wid, n, trg)
        mine.append((wid, float(nll.item()), n))
    model.train()
    bucket = [None] * world
    dist.all_gather_object(bucket, mine, group=gloo)
    merged = sorted([t for part in bucket for t in part])
    assert [t[0] for t in merged] == list(range(len(windows)))
    total_nll, total_n = 0.0, 0
    for _, nll, n in merged:          # ordered accumulation by global window id
        total_nll += nll; total_n += n
    return total_nll / total_n, total_n


def param_digest(model):
    h = hashlib.sha256()
    for _, p in sorted(model.state_dict().items()):
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fp64_reference(inp_fp32_cpu, out_fp32_cpu, rank, world, gloo, chunk=8 * 1024 * 1024):
    """Gathers each rank's FP32 input (taken before the hook divides it by the world size) to rank 0 and sums
    x_r / world in rank order in FP64. The FP64 rounding (about 1e-16) is far below the FP32 rounding of the
    AllReduce (about 6e-8), so the sum serves as the exact result. Gloo is used only to build this reference; the
    AllReduce being measured is the NCCL / R2CC one.
    """
    n = inp_fp32_cpu.numel()
    acc = torch.zeros(n, dtype=torch.float64) if rank == 0 else None
    for off in range(0, n, chunk):
        piece = inp_fp32_cpu[off:off + chunk].contiguous()
        if rank == 0:
            bufs = [torch.empty_like(piece) for _ in range(world)]
            dist.gather(piece, bufs, dst=0, group=gloo)
            s = bufs[0].double() / world
            for r in range(1, world):
                s = s + bufs[r].double() / world
            acc[off:off + piece.numel()] = s
            del bufs, s
        else:
            dist.gather(piece, dst=0, group=gloo)
    if rank != 0:
        return None
    y = out_fp32_cpu.double()
    diff = y - acc
    res = dict(e_max=float(diff.abs().max().item()),
               e_rel=float(diff.norm().item() / max(acc.norm().item(), 1e-30)),
               ref_norm=float(acc.norm().item()),
               out_nonfinite=bool(not torch.isfinite(y).all().item()))
    del acc, y, diff
    return res


def main():
    global disconnect_process, restore_required
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--fail-at-step', type=int, default=0)      # 0 = no failure; else the update that issues the command
    p.add_argument('--capture-lo', type=int, default=400)
    p.add_argument('--capture-hi', type=int, default=408)
    p.add_argument('--micro-batch', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--data', default='/proj/softmeasure-PG0/r2cc_ae/data')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    rank, world, local = [int(os.environ['OMPI_COMM_WORLD_' + k]) for k in ['RANK', 'SIZE', 'LOCAL_RANK']]
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(local))
    assert world == 6, f'expected 6 ranks (three servers x two GPUs), got {world}'
    assert os.environ.get('R2CC_MODE', '0') == '0'
    assert not any(k.startswith('R2CC_FAILED_') for k in os.environ), 'the library must discover the failure itself'
    # The VNF runs load upstream NCCL instead of R2CC; the expected library is named explicitly so that a run can
    # never silently pick up the wrong one.
    expect = os.environ['EXPECT_LIBNCCL']
    if args.fail_at_step:
        assert os.environ['R2CC_AR_AFTER_REPAIR'] in ('2', '3')
        if rank == 0:
            restore_required = True

    torch.cuda.set_device(local); torch.set_num_threads(8)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('nccl', init_method='env://', timeout=datetime.timedelta(minutes=30))
    gloo = dist.new_group(backend='gloo')                       # metrics, digests, eval merge, FP64 reference
    assert torch.cuda.nccl.version() == (2, 23, 4)
    libs = sorted({l.split()[-1] for l in open('/proc/self/maps') if 'libnccl.so' in l})
    assert libs and all(os.path.realpath(x) == os.path.realpath(expect) for x in libs), (libs, expect)

    data_dir = Path(args.data)
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    train = np.memmap(data_dir / 'train.bin', dtype=np.uint16, mode='r')
    val = np.memmap(data_dir / 'val.bin', dtype=np.uint16, mode='r')
    test = np.memmap(data_dir / 'test.bin', dtype=np.uint16, mode='r')
    val_win, test_win = sliding_windows(len(val)), sliding_windows(len(test))
    records = np.random.RandomState(args.seed).permutation((len(train) - 1) // (BLOCK + 1))

    def batch(step):
        base = (step * world + rank) * args.micro_batch
        offs = records[np.arange(base, base + args.micro_batch) % len(records)] * (BLOCK + 1)
        tok = np.stack([train[o:o + BLOCK + 1] for o in offs]).astype(np.int64)
        return (torch.from_numpy(tok[:, :-1].copy()).pin_memory().to(local, non_blocking=True),
                torch.from_numpy(tok[:, 1:].copy()).pin_memory().to(local, non_blocking=True))

    model = GPT().cuda(local)
    ddp = DDP(model, device_ids=[local], bucket_cap_mb=512, gradient_as_bucket_view=True, broadcast_buffers=False)

    cap = dict(update=0, lo=args.capture_lo, hi=args.capture_hi, store={})

    def hook(state, bucket):
        # The copies stay on the GPU; hashing and the FP64 reference are computed after training, so the updates
        # around the failure keep their normal step time.
        u = state['update']
        take = state['lo'] <= u <= state['hi']
        if take:
            state['store'][u] = [bucket.buffer().detach().clone(), None]   # before the hook's in-place div_
        fut = default_hooks.allreduce_hook(None, bucket)
        if not take:
            return fut

        def done(f):
            out = f.value()
            state['store'][u][1] = out.detach().clone()
            return out
        return fut.then(done)

    ddp.register_comm_hook(cap, hook)

    decay = [v for v in model.parameters() if v.ndim >= 2]
    no_decay = [v for v in model.parameters() if v.ndim < 2]
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=no_decay, weight_decay=0.0)],
                                  lr=6e-4, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler('cuda', init_scale=128.0)
    digest_steps = {399, 402, 403, 404, 405, 406, args.steps}

    meta = dict(vars(args), world_size=world, global_batch=world * args.micro_batch, block_size=BLOCK,
                vocab=VOCAB, layers=LAYERS, dim=DIM, heads=HEADS, dropout=0.0,
                parameters=sum(x.numel() for x in model.parameters()), learning_rate=6e-4, min_learning_rate=6e-5,
                warmup_steps=50, weight_decay=0.1, betas=[0.9, 0.95], grad_clip=1.0, bucket_cap_mb=512,
                gradient_as_bucket_view=True, initial_grad_scale=128.0, amp='fp16',
                eval_protocol=dict(context=BLOCK, stride=512, precision='fp32', micro_batch=1,
                                   val_windows=len(val_win), test_windows=len(test_win),
                                   val_targets=len(val) - 1, test_targets=len(test) - 1),
                data_manifest=manifest, torch=torch.__version__, nccl=list(torch.cuda.nccl.version()),
                libnccl=libs, libnccl_expected=expect, r2cc_after_repair=os.environ.get('R2CC_AR_AFTER_REPAIR'),
                env={k: os.environ.get(k) for k in ['R2CC_MODE', 'R2CC_AR_AFTER_REPAIR', 'NCCL_IB_TIMEOUT',
                                                    'NCCL_IB_RETRY_CNT', 'NCCL_R2CC_FAILOVER_TIMEOUT_MS',
                                                    'NCCL_IB_HCA', 'NCCL_ALGO', 'NCCL_DEBUG']},
                first_train_records=records[:world * args.micro_batch].tolist())
    print('RANK_CONFIG ' + json.dumps(dict(rank=rank, host=socket.gethostname(), local_rank=local,
                                           torch=torch.__version__, nccl=list(torch.cuda.nccl.version()),
                                           libnccl=libs, seed=args.seed, name=args.name,
                                           env=meta['env'])), flush=True)

    out_dir, log, writer = Path(args.out) / args.name, None, None
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'training_resolved.json').write_text(json.dumps(meta, indent=2) + '\n')
        log = open(out_dir / 'train_log.csv', 'w', buffering=1)
        writer = csv.DictWriter(log, fieldnames=['update', 'wall_time_s', 'step_time_ms', 'train_loss', 'target_tokens',
                                                 'tokens_per_s', 'lr', 'grad_norm', 'grad_scale', 'step_applied',
                                                 'eval_loss', 'mlx5_2_rx_mb', 'nic_state', 'nic_cmd_unix_s'])
        writer.writeheader()
        print('MODEL_CONFIG ' + json.dumps(meta), flush=True)

    events, captures, digests = [], [], []

    def full_eval(tag, update, data, wins):
        st = (torch.get_rng_state(), torch.cuda.get_rng_state(local), random.getstate(), np.random.get_state())
        loss, n = evaluate(model, data, wins, rank, world, local, gloo)
        torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1], local)
        random.setstate(st[2]); np.random.set_state(st[3])
        if rank == 0:
            print('EVAL ' + json.dumps(dict(split=tag, update=update, nll=loss, ppl=math.exp(loss), targets=n)), flush=True)
        return loss, n

    dist.barrier(group=gloo)
    eval_pts = {0: full_eval('validation', 0, val, val_win)[0]}
    torch.cuda.synchronize(); dist.barrier(group=gloo)
    start, skipped, nic_cmd = time.perf_counter(), 0, ''

    for step in range(args.steps):
        update = step + 1
        cap['update'] = update
        tic = time.perf_counter()
        rx0 = int(RX_COUNTER.read_text()) if rank == 0 else 0
        if rank == 0 and update == args.fail_at_step:
            nic_cmd = time.time()
            disconnect_process = subprocess.Popen([str(NIC_DIR / 'disconnect_nic1.sh')], start_new_session=True,
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print('NIC_DISCONNECT ' + json.dumps(dict(update=update, wall_time_unix_s=nic_cmd,
                                                      elapsed_s=time.perf_counter() - start)), flush=True)
        warm = 50
        lr = 6e-4 * update / warm if step < warm else 6e-5 + 0.5 * (6e-4 - 6e-5) * (1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm - 1)))
        for g in optimizer.param_groups:
            g['lr'] = lr
        x, y = batch(step)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            loss = ddp(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        old = scaler.get_scale()
        scaler.step(optimizer); scaler.update()
        applied = int(scaler.get_scale() >= old)
        skipped += 1 - applied
        mean_loss = loss.detach().float().cpu()
        dist.all_reduce(mean_loss, group=gloo); mean_loss /= world
        torch.cuda.synchronize()
        el = torch.tensor(time.perf_counter() - tic, dtype=torch.float64)
        dist.all_reduce(el, op=dist.ReduceOp.MAX, group=gloo)
        elapsed = el.item()
        assert math.isfinite(mean_loss.item()), 'non-finite training loss'

        if update in digest_steps:
            d = param_digest(model)
            alld = [None] * world
            dist.all_gather_object(alld, d, group=gloo)
            if rank == 0:
                rec = dict(update=update, agree=len(set(alld)) == 1, digest=d)
                digests.append(rec)
                print('PARAM_DIGEST ' + json.dumps(rec), flush=True)

        ev = ''
        if update % args.eval_every == 0 or update == args.steps:
            ev = full_eval('validation', update, val, val_win)[0]
            eval_pts[update] = ev
        if rank == 0:
            rx1 = int(RX_COUNTER.read_text())
            writer.writerow(dict(update=update, wall_time_s=time.perf_counter() - start, step_time_ms=1000 * elapsed,
                                 train_loss=repr(mean_loss.item()), target_tokens=world * args.micro_batch * BLOCK,
                                 tokens_per_s=world * args.micro_batch * BLOCK / elapsed, lr=repr(lr),
                                 grad_norm=repr(float(gnorm)), grad_scale=scaler.get_scale(), step_applied=applied,
                                 eval_loss=(repr(ev) if ev != '' else ''), mlx5_2_rx_mb=(rx1 - rx0) * 4 / 1e6,
                                 nic_state=('down' if nic_cmd != '' else 'up'), nic_cmd_unix_s=nic_cmd))

    # Capture analysis runs after the training loop so the injection window stays at normal step time.
    for u in sorted(cap['store']):
        inp_g, out_g = cap['store'][u]
        assert out_g is not None, u
        inp_cpu, out_cpu = inp_g.float().cpu(), out_g.float().cpu()
        cap['store'][u] = None
        del inp_g, out_g
        torch.cuda.empty_cache()
        dg = hashlib.sha256(out_cpu.numpy().tobytes()).hexdigest()
        allo = [None] * world
        dist.all_gather_object(allo, dg, group=gloo)
        ref = fp64_reference(inp_cpu, out_cpu, rank, world, gloo)
        if rank == 0:
            rec = dict(update=u, numel=int(inp_cpu.numel()), out_digests_agree=len(set(allo)) == 1,
                       out_digest=dg, **ref)
            captures.append(rec)
            print('CAPTURE ' + json.dumps(rec), flush=True)
        del inp_cpu, out_cpu

    test_nll, test_n = full_eval('test', args.steps, test, test_win)
    val_nll = eval_pts[args.steps]
    if rank == 0:
        summary = dict(name=args.name, seed=args.seed, steps=args.steps, fail_at_step=args.fail_at_step,
                       after_repair=os.environ.get('R2CC_AR_AFTER_REPAIR'), skipped_optimizer_steps=skipped,
                       final_val_nll=val_nll, final_val_ppl=math.exp(val_nll),
                       test_nll=test_nll, test_ppl=math.exp(test_nll), test_targets=test_n,
                       eval_points={str(k): v for k, v in eval_pts.items()}, captures=captures,
                       param_digests=digests, nic_cmd_unix_s=nic_cmd,
                       peak_cuda_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                       total_wall_time_s=time.perf_counter() - start,
                       ddp=ddp._get_ddp_logging_data())
        (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print('FINISHED ' + json.dumps({k: v for k, v in summary.items() if k not in ('eval_points', 'captures', 'param_digests', 'ddp')}), flush=True)
        log.close()
        restore_nic()
    dist.barrier(group=gloo)
    dist.destroy_process_group()


if __name__ == '__main__':
    try:
        main()
    finally:
        restore_nic()
