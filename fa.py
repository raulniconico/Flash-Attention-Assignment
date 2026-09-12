from functools import lru_cache

from tile_model import Hardware, VCPU, SA1, SA2, divup, DMA, SRAM

QK_LEAD = 2  # Arrays queue QK(p+QK_LEAD) before PV(p): PV lags two key blocks.
PACK_LOOKAHEAD = QK_LEAD + 1  # QK operands are packed this many blocks ahead.
UPDATE_LAG = 1  # U/l/m update of block p is queued after the pack work of p+1.


@lru_cache(maxsize=4096)
def _array_matmul(m, n, depth, rows_sa1, R, hw, diag=None, axis='cols'):
    """Schedule the physical jobs of one GEMM; return completion and traffic.

    Rows split m between SA1 (the first rows_sa1) and SA2. One job costs
    command latency + stage (operands in / MACs) + output drain; the drain
    overlaps the next stage and both arrays share one command interface
    ('pipelined' assumes timed early issue, a user assumption). With diag
    given, a job is skipped when every entry it produces (axis='cols',
    scores) or consumes (axis='depth', P.V) is causally masked.
    """
    arrays = [SA1(compute_model=hw.array_model), SA2(compute_model=hw.array_model)]
    split = min(m, rows_sa1)
    stages, packed_bytes, raw_bytes, skipped = [], 0, 0, 0
    for sa, base, rows in zip(arrays, (0, split), (split, m - split)):
        if rows:
            padded_rows = divup(rows, sa.physical_rows) * sa.physical_rows
            packed_bytes += hw.digits * (padded_rows + divup(n, 16) * 16) * depth
        stream = []  # stage (input/compute) cycles of each job, in issue order
        for r in range(0, rows, sa.physical_rows):
            last_row = base + r + min(sa.physical_rows, rows-r) - 1
            for c in range(0, n, 16):
                for k in range(0, depth, R):
                    if diag is not None and (c if axis == 'cols' else k) - last_row > diag:
                        skipped += hw.digits**2
                        continue
                    job = type(sa)(min(sa.physical_rows, rows-r), min(R, depth-k),
                                   min(16, n-c), hw.array_model)
                    stream += [job.stage_cycles()] * hw.digits**2
        stages.append(stream)
        raw_bytes += len(stream) * sa.output_bytes()

    advance = hw.array_latency if hw.control == 'pipelined' else 0
    interval = hw.issue_interval if hw.control == 'pipelined' else hw.array_latency
    cpu, pos, ready, drain, first = 0, [0, 0], [0, 0], [0, 0], [None, None]
    while True:
        waiting = [(max(cpu, ready[a] - advance), a)
                   for a in (0, 1) if pos[a] < len(stages[a])]
        if not waiting:
            break
        issue, a = min(waiting)
        if first[a] is None:
            first[a] = issue
        # If the drain is occupied, hold the new result until it becomes free.
        ready[a] = max(issue + hw.array_latency + stages[a][pos[a]], drain[a])
        drain[a] = ready[a] + arrays[a].output_cycles()
        cpu = issue + interval
        pos[a] += 1
    return dict(start=first, end=drain, jobs=[len(s) for s in stages],
                skipped=skipped, rows_sa1=split,
                packed_bytes=packed_bytes, raw_bytes=raw_bytes)


@lru_cache(maxsize=4096)
def _balanced_split(m, n, depth, R, hw, diag=None, axis='cols'):
    """Rows for SA1 (multiple of 16) that minimize the two-array makespan."""
    best = None
    for rows in sorted(set(range(0, m+1, 16)) | {m}):
        t = _array_matmul(m, n, depth, rows, R, hw, diag, axis)
        key = (max(t['end']), abs(t['end'][0]-t['end'][1]), rows)
        best = key if best is None or key < best else best
    return best[2]


class _Queue:
    """Three in-order engine queues: DMA, VCPU and the SA1+SA2 command pair.

    A command starts when its engine is free and every producer it depends on
    has finished. overlap=False also chains each command to the previous one,
    i.e. the fully serial schedule. Every task declares deps/alloc/free, so
    the schedule alone determines SRAM occupancy (see _replay_sram).
    """
    def __init__(self, hw, overlap=True):
        self.hw, self.overlap = hw, overlap
        self.tasks, self.free_at = [], {'DMA': 0, 'VCPU': 0, 'ARRAYS': 0}

    def task(self, name, unit, duration, deps=(), alloc=(), free=(), info=None):
        deps = [d for d in deps if d is not None]
        if not self.overlap and self.tasks:
            deps.append(self.tasks[-1]['id'])
        start = max([self.free_at[unit]] + [self.tasks[d]['end'] for d in deps])
        rec = dict(id=len(self.tasks), name=name, unit=unit, start=start,
                   end=start+duration, cycles=duration, deps=deps,
                   alloc=list(alloc), free=list(free), info=info)
        self.free_at[unit] = rec['end']
        self.tasks.append(rec)
        return rec['id']

    def vector(self, name, reads, writes, **kw):
        return self.task(name, 'VCPU', VCPU(reads, writes, self.hw).launch_cycle(), **kw)

    def dma(self, name, nbytes, direction, **kw):
        return self.task(name, 'DMA', DMA(nbytes, direction, self.hw).cycle(), **kw)

    def arrays(self, name, timing, dep, raw, dig):
        """One GEMM on the shared command interface; frees its packed operands."""
        return self.task(name, 'ARRAYS', max(timing['end']), deps=[dep],
                         alloc=[(raw, timing['raw_bytes'])], free=[dig], info=timing)


def _replay_sram(tasks, hw):
    """Replay the timed allocs/frees; record used_bytes per task, return peak.

    Buffers of in-flight blocks coexist, so occupancy follows the schedule and
    not the emission order. Over-subscription raises MemoryError.
    """
    sram = SRAM(hw=hw)
    order = []
    for t in tasks:
        order += [(t['start'], 1, t['id'], name, nb) for name, nb in t['alloc']]
        order += [(t['end'], 0, t['id'], name, 0) for name in t['free']]
        order.append((t['start'], 2, t['id'], None, 0))
    for _, kind, tid, name, nb in sorted(order, key=lambda e: e[:3]):
        if kind == 1:
            sram.allocate(name, nb, 8)
        elif kind == 0:
            sram.free(name)
        else:
            tasks[tid]['used_bytes'] = sram.used_bytes
    peak = sram.peak_bytes
    for name in list(sram.buffers):
        sram.free(name)
    return sram, peak


def _summarize(tasks, counts, hw, verbose):
    """Busy cycles per queue, job counts, Gantt events and the step table."""
    breakdown, events, steps = {}, [], []
    for t in sorted(tasks, key=lambda t: (t['start'], t['id'])):
        step = dict(name=t['name'], unit=t['unit'], start=t['start'], end=t['end'],
                    cycles=t['cycles'], used_bytes=t['used_bytes'])
        timing = t['info']
        if t['unit'] != 'ARRAYS':
            counts[t['unit']] += 1
            breakdown[t['unit']] = breakdown.get(t['unit'], 0) + t['cycles']
            events.append(step.copy())
        else:
            step['unit'] = 'SA1+SA2'
            breakdown['SA1+SA2'] = breakdown.get('SA1+SA2', 0) + t['cycles']
            counts['skipped_jobs'] += timing['skipped']
            for a in range(2):
                counts[f'SA{a+1}_jobs'] += timing['jobs'][a]
                if timing['jobs'][a]:
                    events.append(dict(name=t['name'], unit=f'SA{a+1}',
                                       start=t['start']+timing['start'][a],
                                       end=t['start']+timing['end'][a]))
        steps.append(step)
        if verbose:
            print(f'{t["name"]:42s} {t["unit"]:6s} @{t["start"]:>11,d} '
                  f'{t["cycles"]:9,d} cycles {hw.time_us(t["cycles"]):9.3f} us  '
                  f'SRAM {t["used_bytes"]/1024:9.2f} KiB')
            if t['unit'] == 'ARRAYS':
                for a in range(2):
                    if timing['jobs'][a]:
                        elapsed = timing['end'][a] - timing['start'][a]
                        print(f'  SA{a+1}: {timing["jobs"][a]:,} commands '
                              f'({timing["rows_sa1"] if a == 0 else "rest"} rows), '
                              f'{elapsed:,} cycles ({hw.time_us(elapsed):.3f} us); '
                              f'starts +{timing["start"][a]} cycles in GEMM')
    return breakdown, events, steps


def _report(result, verbose):
    """Print the totals shared by both kernels and return the result dict."""
    if verbose:
        cycles, sram = result['cycles'], result['sram']
        print(f'\nTotal: {result["time_ms"]:.6f} ms ({cycles:,} cycles)')
        print('Busy: ' + ', '.join(f'{u} {100*b/cycles:.1f}%'
                                   for u, b in result['breakdown'].items()))
        print(f'Peak SRAM: {result["peak_sram_bytes"]:,} bytes '
              f'({result["max_sram_utilization_percent"]:.3f}%)')
    return result


def flash_attention(T=2048, S=2048, H=128, G=4, Br=256, Bc=256,
                    Br_sa1=None, R=128, causal=True, q_offset=0,
                    hw=Hardware(), overlap=True, skip_masked=True,
                    verbose=True, gantt_path=None):
    """Time ONE KV-head group of prefill, in ONE layer and ONE batch element.

    G Q heads share K/V, read and quantized once into SRAM; for the whole
    model the serial group count is B * L * K (K = KV heads). R is the array
    reduction chunk. Causal positions: key <= q_offset + query; short tail
    blocks are supported.

    Pipeline (overlap=True) over the (query tile, key block) pairs p: the
    arrays run QK(p+QK_LEAD) before PV(p), the vector CPU packs QK operands
    PACK_LOOKAHEAD blocks ahead and lags the U/l/m update by UPDATE_LAG, so
    softmax hides under the score GEMMs and the in-order vector queue never
    stalls on PV(p). K is quantized before V, Q of the next tile is
    prefetched into a second buffer, and O16 stores overlap the next tile.

    Work split. Br_sa1 rows of every GEMM go to SA1 and the rest to SA2;
    None picks, per geometry, the multiple of 16 minimizing the two-array
    makespan. With skip_masked, causal blocks are trimmed to 16-column
    granularity and fully masked jobs are skipped.

    Quantization. safe-int8 packs two signed base-16 digit planes per operand
    (four products, reduction <= 128); native-int8 is a timing comparison that
    assumes a suitable output mechanism. Q/K/V use scalar INT8 scales, P a
    fixed 1/127. VCPU byte formulas state their passes explicitly. Timing and
    metadata only: no numerical attention.
    """
    for value in (T, S, H, G, Br, Bc, R):
        if not isinstance(value, int) or value <= 0:
            raise ValueError('Dimensions, G and R must be positive integers')
    if hw.arithmetic not in ('safe-int8', 'native-int8'):
        raise ValueError('Unknown arithmetic mode')
    if hw.control not in ('serialized', 'pipelined') or hw.issue_interval < 1:
        raise ValueError('Invalid control mode or issue interval')
    if not isinstance(q_offset, int) or q_offset < 0:
        raise ValueError('q_offset must be a nonnegative integer')
    if Br_sa1 is not None and (not isinstance(Br_sa1, int) or not 0 <= Br_sa1 <= Br):
        raise ValueError('Br_sa1 must be None (auto) or between 0 and Br')

    def split(m, n, depth, diag, axis):
        if Br_sa1 is None:
            return _balanced_split(m, n, depth, R, hw, diag, axis)
        return min(Br_sa1, m)

    # ---- work list -------------------------------------------------------
    tiles = [(g, i, min(Br, T-i)) for g in range(G) for i in range(0, T, Br)]
    pairs = []
    for t, (g, i, m) in enumerate(tiles):
        key_end = min(S, q_offset+i+m) if causal else S
        for j in range(0, key_end, Bc):
            c, diag = min(Bc, S-j), None
            if causal and skip_masked:
                c = min(c, divup(key_end-j, 16)*16)
                if j+c-1 > q_offset+i:  # block touches the diagonal
                    diag = q_offset+i-j
            pairs.append(dict(tile=t, j=j, m=m, c=c, diag=diag,
                              tag=f'h{g} q{i} k{j}'))
    P = len(pairs)
    last_pair_of_tile = {pr['tile']: p for p, pr in enumerate(pairs)}

    q = _Queue(hw, overlap)
    tasks = q.tasks

    # K/V loaded once for all G query heads.
    dK = q.dma('Load K16', 2*S*H, 'read', alloc=[('K16', 2*S*H)])
    qK = q.vector('Quantize K', 4*S*H, S*H+4, deps=[dK],  # Two A16 reads.
                  alloc=[('K8', S*H), ('K_scale', 4)], free=['K16'])
    dV = q.dma('Load V16', 2*S*H, 'read', alloc=[('V16', 2*S*H)])

    tile_ready, packed, qk_done, pv_done = {}, {}, {}, {}
    prev_softmax, prev_update = {}, {}

    def ensure_tile(t):
        if t in tile_ready:
            return
        g, i, m = tiles[t]
        tag = f'h{g} q{i}'
        d = q.dma(tag+' load Q16', 2*m*H, 'read', deps=[tile_ready.get(t-1)],
                  alloc=[(f'Q16@t{t}', 2*m*H)])
        tile_ready[t] = q.vector(
            tag+' quantize Q + init U,l,m', 4*m*H, 5*m*H+8*m+4, deps=[d],
            alloc=[(f'Q8@t{t}', m*H), (f'Q_scale@t{t}', 4), (f'U@t{t}', 4*m*H),
                   (f'l@t{t}', 4*m), (f'm@t{t}', 4*m)],
            free=[f'Q16@t{t}'])

    def emit_pack(p):
        if p >= P or p in packed:
            return
        pr = pairs[p]
        ensure_tile(pr['tile'])
        m, c = pr['m'], pr['c']
        timing = _array_matmul(m, c, H, split(m, c, H, pr['diag'], 'cols'),
                               R, hw, pr['diag'], 'cols')
        # Read each logical operand once; write digit planes including padding.
        pid = q.vector(pr['tag']+' QK pack', (m+c)*H, timing['packed_bytes'],
                       deps=[tile_ready[pr['tile']], qK],
                       alloc=[(f'QKdig@{p}', timing['packed_bytes'])])
        packed[p] = (pid, timing)

    def emit_qk(p):
        emit_pack(p)  # already packed ahead when overlapping
        pid, timing = packed[p]
        qk_done[p] = q.arrays(pairs[p]['tag']+' QK', timing, pid,
                              f'QKraw@{p}', f'QKdig@{p}')

    def emit_softmax_pv(p):
        pr = pairs[p]
        t, m, c, tag = pr['tile'], pr['m'], pr['c'], pr['tag']
        # Reconstruct in vector registers, discard padding, then scale to FP32.
        rec = q.vector(tag+' QK reconstruct', packed[p][1]['raw_bytes']+8, 4*m*c,
                       deps=[qk_done[p]], alloc=[(f'scores@{p}', 4*m*c)],
                       free=[f'QKraw@{p}'])
        # Two FP32 score reads: rowmax, then exp/sum/P8 encoding.
        # m_new=max(m,rowmax(S)); alpha=exp(m-m_new).
        # l_new=alpha*l+sum(exp(S-m_new)); masked P entries are zero.
        sm = q.vector(tag+' softmax + P8', 8*m*c+8*m, m*c+12*m,
                      deps=[rec, prev_softmax.get(t)],
                      alloc=[(f'P8@{p}', m*c), (f'stats@{p}', 12*m)],
                      free=[f'scores@{p}'])
        prev_softmax[t] = sm
        timing = _array_matmul(m, H, c, split(m, H, c, pr['diag'], 'depth'),
                               R, hw, pr['diag'], 'depth')
        pk = q.vector(tag+' PV pack', (m+H)*c, timing['packed_bytes'],
                      deps=[sm, qV], alloc=[(f'PVdig@{p}', timing['packed_bytes'])],
                      free=[f'P8@{p}'])
        return pk, timing

    def emit_update(p):
        pr = pairs[p]
        t, m, tag = pr['tile'], pr['m'], pr['tag']
        rec = q.vector(tag+' PV reconstruct', tasks[pv_done[p]]['info']['raw_bytes']+4,
                       4*m*H, deps=[pv_done[p]], alloc=[(f'C@{p}', 4*m*H)],
                       free=[f'PVraw@{p}'])
        # U <- alpha*U + C; l <- l_new; m <- m_new.
        prev_update[t] = q.vector(tag+' update U,l,m', 8*m*H+12*m, 4*m*H+8*m,
                                  deps=[rec, prev_update.get(t), prev_softmax[t]],
                                  free=[f'C@{p}', f'stats@{p}'])
        if last_pair_of_tile[t] == p:
            g, i, m = tiles[t]
            ttag = f'h{g} q{i}'
            norm = q.vector(ttag+' normalize + encode O16', 4*m*H+4*m, 2*m*H,
                            deps=[prev_update[t]], alloc=[(f'O16@t{t}', 2*m*H)],
                            free=[f'Q8@t{t}', f'Q_scale@t{t}', f'U@t{t}',
                                  f'l@t{t}', f'm@t{t}'])
            q.dma(ttag+' store O16', 2*m*H, 'write', deps=[norm],
                  free=[f'O16@t{t}'])

    # Serial schedule = the same loop with no lead, lookahead or lag.
    lead, ahead, lag = (QK_LEAD, PACK_LOOKAHEAD, UPDATE_LAG) if overlap else (0, 0, 0)
    for p in range(ahead):
        emit_pack(p)
    for p in range(min(lead, P)):
        emit_qk(p)
    qV = q.vector('Quantize V + transpose', 4*S*H, S*H+4, deps=[dV],
                  alloc=[('V8T', S*H), ('V_scale', 4)], free=['V16'])
    for p in range(P):
        if p+lead < P:
            emit_qk(p+lead)
        pk, timing = emit_softmax_pv(p)
        emit_pack(p+ahead)
        pv_done[p] = q.arrays(pairs[p]['tag']+' PV', timing, pk,
                              f'PVraw@{p}', f'PVdig@{p}')
        if p >= lag:
            emit_update(p-lag)
    for p in range(max(0, P-lag), P):
        emit_update(p)
    cycles = max(t['end'] for t in tasks)

    sram, peak = _replay_sram(tasks, hw)
    counts = dict(query_tiles=len(tiles), pairs=P, DMA=0, VCPU=0,
                  SA1_jobs=0, SA2_jobs=0, skipped_jobs=0)
    breakdown, events, steps = _summarize(tasks, counts, hw, verbose)
    full = pairs[0]
    result = dict(cycles=cycles, time_us=hw.time_us(cycles),
                  time_ms=hw.time_us(cycles)/1000,
                  kv_setup_cycles=tasks[qV]['end'],
                  query_work_cycles=cycles-tasks[qV]['end'], counts=counts,
                  breakdown=breakdown,
                  utilization={u: b/cycles for u, b in breakdown.items()},
                  steps=steps, events=events, sram=sram, peak_sram_bytes=peak,
                  max_sram_utilization_percent=100*peak/sram.capacity_bytes,
                  first_tile_end=next(t['end'] for t in tasks
                                      if t['name'].endswith('store O16')),
                  hw=hw,
                  config=dict(T=T,S=S,H=H,G=G,Br=Br,Bc=Bc,Br_sa1=Br_sa1,R=R,
                              overlap=overlap, skip_masked=skip_masked,
                              split_qk=split(full['m'], full['c'], H, None, 'cols'),
                              split_pv=split(full['m'], H, full['c'], None, 'depth')))
    _report(result, verbose)
    if gantt_path is not None:
        plot_flash_attention(result, gantt_path)
    return result


def flash_decode(S=2048, H=128, G=4, Bc=2048, R=256, hw=Hardware(),
                 overlap=True, verbose=True, gantt_path=None):
    """Time ONE decode step of ONE KV-head group: G query heads, one token each.

    For the whole model, serial group count = B * L * K per generated token.
    Three changes make the prefill kernel fit decode:

    1. Orientation. With only G query rows the prefill layout leaves one array
       idle, so the LONG dimension drives the rows instead:
           scores  P[c,G] = K8[c,H]  . Q8[H,G]    rows = the c keys
           output  O[H,G] = V8T[H,c] . P8[c,G]    rows = the head dimension
       The G queries still fill G of the 16 output columns of every job, the
       irreducible cost of one token per sequence.
    2. KV cache. It holds INT8 K and pre-transposed INT8 V with per-block
       scales, so a block is used straight from DRAM: no quantize pass, half
       the bytes. Blocks stream with a two-block prefetch.
    3. Softmax. m, l and the FP32 accumulator U[H,G] are updated once per key
       block; the last update normalizes and stores O16.
    """
    for value in (S, H, G, Bc, R):
        if not isinstance(value, int) or value <= 0:
            raise ValueError('S, H, G, Bc and R must be positive integers')

    q = _Queue(hw, overlap)
    tasks = q.tasks

    # Q of this step: G rows, transposed to (H,G) for the column operand.
    dQ = q.dma('Load Q16', 2*G*H, 'read', alloc=[('Q16', 2*G*H)])
    qQ = q.vector('Quantize Q + transpose, init U,l,m', 4*G*H, G*H+4*H*G+8*G+4,
                  deps=[dQ], free=['Q16'],
                  alloc=[('Q8', G*H), ('Q_scale', 4), ('U', 4*H*G),
                         ('l', 4*G), ('m', 4*G)])

    packs, update = {}, None
    for b, j in enumerate(range(0, S, Bc)):
        c, tag = min(Bc, S-j), f'k{j}'
        # INT8 cache blocks; load b waits for the pack of b-2 (two slots).
        dk = q.dma(f'{tag} load K8', c*H, 'read', deps=[packs.get(b-2)],
                   alloc=[(f'K8@{b}', c*H), (f'K_scale@{b}', 4)])
        dv = q.dma(f'{tag} load V8T', c*H, 'read', deps=[packs.get(b-2)],
                   alloc=[(f'V8T@{b}', c*H), (f'V_scale@{b}', 4)])

        t_qk = _array_matmul(c, G, H, _balanced_split(c, G, H, R, hw), R, hw)
        # Read each logical operand once; write digit planes including padding.
        packs[b] = q.vector(f'{tag} QK pack', (c+G)*H, t_qk['packed_bytes'],
                            deps=[dk, qQ], free=[f'K8@{b}', f'K_scale@{b}'],
                            alloc=[(f'{tag}QKdig', t_qk['packed_bytes'])])
        qk = q.arrays(f'{tag} QK', t_qk, packs[b], f'{tag}QKraw', f'{tag}QKdig')

        # Reconstruct to FP32, then a second pass for exp/sum and the P8 encode.
        # m_new=max(m,colmax(P)); alpha=exp(m-m_new); l_new=alpha*l+sum(exp).
        sm = q.vector(f'{tag} softmax + P8', t_qk['raw_bytes']+4*c*G+8*G,
                      4*c*G+c*G+12*G, deps=[qk, update], free=[f'{tag}QKraw'],
                      alloc=[(f'P8@{b}', c*G), (f'stats@{b}', 12*G)])

        t_pv = _array_matmul(H, G, c, _balanced_split(H, G, c, R, hw), R, hw)
        pk = q.vector(f'{tag} PV pack', (H+G)*c, t_pv['packed_bytes'],
                      deps=[sm, dv], alloc=[(f'{tag}PVdig', t_pv['packed_bytes'])],
                      free=[f'P8@{b}', f'V8T@{b}', f'V_scale@{b}'])
        pv = q.arrays(f'{tag} PV', t_pv, pk, f'{tag}PVraw', f'{tag}PVdig')

        # U <- alpha*U + C; l <- l_new; m <- m_new.
        update = q.vector(f'{tag} accumulate U,l,m', t_pv['raw_bytes']+4*H*G+12*G,
                          4*H*G+8*G, deps=[pv, update, sm],
                          free=[f'{tag}PVraw', f'stats@{b}'])

    norm = q.vector('normalize + encode O16', 4*H*G+4*G, 2*G*H, deps=[update],
                    alloc=[('O16', 2*G*H)],
                    free=['U', 'l', 'm', 'Q8', 'Q_scale'])
    q.dma('store O16', 2*G*H, 'write', deps=[norm], free=['O16'])
    cycles = max(t['end'] for t in tasks)

    sram, peak = _replay_sram(tasks, hw)
    counts = dict(key_blocks=len(packs), DMA=0, VCPU=0,
                  SA1_jobs=0, SA2_jobs=0, skipped_jobs=0)
    breakdown, events, steps = _summarize(tasks, counts, hw, verbose)
    macs = 2 * G * H * S                        # scores + output, one token
    result = dict(cycles=cycles, time_us=hw.time_us(cycles),
                  time_ms=hw.time_us(cycles)/1000, counts=counts,
                  breakdown=breakdown, macs=macs, mac_per_cycle=macs/cycles,
                  dram_bytes=2*S*H + 2*G*H + 2*G*H,   # INT8 cache + Q16 + O16
                  utilization={u: b/cycles for u, b in breakdown.items()},
                  steps=steps, events=events, sram=sram, peak_sram_bytes=peak,
                  max_sram_utilization_percent=100*peak/sram.capacity_bytes,
                  hw=hw, first_tile_end=next(t['end'] for t in tasks
                                             if t['name'].endswith('accumulate U,l,m')),
                  panel_titles=['Complete decode step', 'First key block'],
                  config=dict(S=S, H=H, G=G, Bc=Bc, R=R, overlap=overlap))
    result['title'] = (
        f'FlashAttention decode | {hw.array_model} | {hw.control} control | '
        f'{"overlapped engines" if overlap else "serial"}\n'
        f'S={S}, H={H}, G={G}, Bc={Bc}, R={R} | {result["time_us"]:.2f} us | '
        f'{result["mac_per_cycle"]:.1f} MAC/cycle | peak SRAM '
        f'{result["max_sram_utilization_percent"]:.2f}%')
    _report(result, verbose)
    if gantt_path is not None:
        plot_flash_attention(result, gantt_path)
    return result


def flash_attention_vanilla(T=2048, S=2048, H=128, G=4, Br=128, Bc=256, R=128,
                            causal=True, hw=Hardware(), verbose=False):
    """Fully serial FA2 baseline: one command at a time, nothing overlapped."""
    sram, cycles, busy = SRAM(hw=hw), 0, {'DMA': 0, 'VCPU': 0, 'SA': 0}
    digits, pairs = hw.digits, hw.digits**2

    def step(name, unit, n):
        nonlocal cycles
        cycles += n
        busy[unit] += n
        if verbose:
            print(f'{name:38s} {unit:5s} {n:9,d} cycles  SRAM {sram.used_bytes/1024:8.1f} KiB')

    def vcpu(name, reads, writes):
        step(name, 'VCPU', VCPU(reads, writes, hw).launch_cycle())

    def gemm(name, m, n, depth):
        """CPU issues SA1/SA2 commands for the physical tiles; wait for all.

        Rows split half/half; each array runs its commands back to back.
        One command = one padded output tile x one reduction chunk x digit pair.
        """
        rows = [divup(m, 2), m - divup(m, 2)]
        stream = 0
        for sa, r in zip((SA1(compute_model=hw.array_model), SA2(compute_model=hw.array_model)), rows):
            commands = divup(r, sa.physical_rows) * divup(n, 16) * divup(depth, R) * pairs
            stream = max(stream, commands * sa.compute(sa.physical_rows, min(R, depth), 8, 16, hw=hw))
        step(name, 'SA', stream)
        return 2 * m * n * divup(depth, R) * pairs  # raw INT16 partial bytes

    # DMA load K16, V16
    step('DMA load K16', 'DMA', DMA(hw=hw).transfer((S, H), 16, 'read', sram, 'K16'))
    step('DMA load V16', 'DMA', DMA(hw=hw).transfer((S, H), 16, 'read', sram, 'V16'))
    # VCPU K16 -> K8, V16 -> packed transposed V8 ; SRAM write K8, V8, sK, sV
    sram.allocate('K8', (S, H), 8); sram.allocate('V8T', (H, S), 8); sram.allocate('sKV', 2, 'fp32')
    vcpu('VCPU quantize K,V + transpose V', 8*S*H, 2*S*H + 8)
    sram.free('K16'); sram.free('V16')

    for g in range(G):
        for i in range(0, T, Br):                       # FOR i
            m = min(Br, T - i)
            step('DMA load Qi16', 'DMA', DMA(hw=hw).transfer((m, H), 16, 'read', sram, 'Q16'))
            sram.allocate('Q8', (m, H), 8); sram.allocate('U', (m, H), 'fp32')
            sram.allocate('l', m, 'fp32'); sram.allocate('m', m, 'fp32')
            vcpu('VCPU Qi16 -> Qi8; init U,l,m', 4*m*H, 5*m*H + 8*m + 4)
            sram.free('Q16')

            key_end = min(S, i + m) if causal else S
            for j in range(0, key_end, Bc):             # FOR j
                c = min(Bc, S - j)
                # VCPU pack Qi8, Kj8 digits
                sram.allocate('dig', digits * (m + c) * H, 8)
                vcpu('VCPU pack Q,K digits', (m + c) * H, digits * (m + c) * H)
                # SA1 & SA2 compute S16 partials ; CPU wait
                raw = gemm('SA1+SA2 S = Q K^T', m, c, H)
                sram.allocate('raw', raw, 8); sram.free('dig')
                # VCPU reconstruct S16 -> S_FP32
                sram.allocate('S32', (m, c), 'fp32')
                vcpu('VCPU reconstruct S -> FP32', raw + 8, 4*m*c)
                sram.free('raw')
                # VCPU softmax: mask, new_m, alpha, P, b, P8, sP
                sram.allocate('P8', (m, c), 8); sram.allocate('stats', 3*m, 'fp32')
                vcpu('VCPU softmax -> P8', 8*m*c + 8*m, m*c + 12*m)
                sram.free('S32')
                # VCPU pack P8, Vj8 digits
                sram.allocate('dig', digits * (m + H) * c, 8)
                vcpu('VCPU pack P,V digits', (m + H) * c, digits * (m + H) * c)
                sram.free('P8')
                # SA1 & SA2 compute C16 partials ; CPU wait
                raw = gemm('SA1+SA2 C = P V', m, H, c)
                sram.allocate('raw', raw, 8); sram.free('dig')
                # VCPU reconstruct C16 -> C_FP32
                sram.allocate('C32', (m, H), 'fp32')
                vcpu('VCPU reconstruct C -> FP32', raw + 4, 4*m*H)
                sram.free('raw')
                # VCPU U = alpha*U + C ; l = alpha*l + b ; swap m
                vcpu('VCPU update U,l', 8*m*H + 12*m, 4*m*H + 8*m)
                sram.free('C32'); sram.free('stats')

            # VCPU Oi = U / l -> Oi16 ; DMA store Oi16
            sram.allocate('O16', (m, H), 16)
            vcpu('VCPU normalize -> Oi16', 4*m*H + 4*m, 2*m*H)
            for name in ('Q8', 'U', 'l', 'm'):
                sram.free(name)
            step('DMA store Oi16', 'DMA', DMA(hw=hw).transfer((m, H), 16, 'write', sram, 'O16', release=True))

    for name in ('K8', 'V8T', 'sKV'):
        sram.free(name)
    result = dict(cycles=cycles, time_ms=hw.time_us(cycles) / 1000, busy=busy,
                  peak_sram_bytes=sram.peak_bytes,
                  sram_utilization_percent=100 * sram.max_utilization())
    if verbose:
        print(f'\nTotal {result["time_ms"]:.3f} ms; busy {busy}; '
              f'peak SRAM {result["sram_utilization_percent"]:.2f}%')
    return result


def plot_flash_attention(result, path):
    """Save a Gantt PNG; matplotlib is only needed when plotting.

    Array bars are stream envelopes including command gaps and draining,
    not claims of continuous arithmetic. Zoom shows cold setup + first Q tile
    (or result['zoom_end']); result['title'] / ['panel_titles'] override text.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import re

    hw, cfg = result['hw'], result['config']
    colors = {'DMA':'#4979b6', 'VCPU':'#dd9338', 'SA1':'#329b80', 'SA2':'#8965ba'}
    lanes = list(colors)
    zoom_end = result.get('zoom_end', result['first_tile_end'])
    zoom_events = sorted((e for e in result['events'] if e['start'] < zoom_end),
                         key=lambda e: (e['start'], lanes.index(e['unit'])))
    ncol, per_col = 3, -(-len(zoom_events) // 3)
    fig, axes = plt.subplots(3, 1, figsize=(15, 11 + 0.16*per_col),
                             height_ratios=[2, 3, 0.18*per_col + 0.4],
                             layout='constrained')
    panel_titles = result.get('panel_titles',
                              ['Complete KV-head group', 'Cold setup + first query tile'])
    for ax, end, title, zoom in zip(axes[:2], [result['cycles'], zoom_end],
                                    panel_titles, [False, True]):
        shown = [e for e in result['events'] if e['start'] < end]
        if not zoom:  # one bar collection per lane: fast for long schedules
            for unit in lanes:
                ranges = [(hw.time_us(e['start']),
                           hw.time_us(min(e['end'], end)-e['start']))
                          for e in shown if e['unit'] == unit]
                ax.broken_barh(ranges, (lanes.index(unit)-0.35, 0.7),
                               facecolors=colors[unit], edgecolors='white',
                               linewidth=0.35)
            shown = []
        for e in shown:
            x = hw.time_us(e['start'])
            width = hw.time_us(min(e['end'], end)-e['start'])
            y = lanes.index(e['unit'])
            ax.broken_barh([(x, width)], (y-0.35, 0.7), facecolors=colors[e['unit']],
                           edgecolors='white', linewidth=0.35)
            # Zoom panel: wide blocks carry task + cycles; every block carries
            # its index into the table below.
            k = zoom_events.index(e) + 1
            cyc = e['end']-e['start']
            if width > hw.time_us(end)*0.08:
                label = re.sub(r'^h\d+ ', '', e['name'])
                ax.text(x+width/2, y, f'[{k}] {label}\n{cyc:,} cyc', ha='center',
                        va='center', fontsize=7.5, color='white')
            elif width > hw.time_us(end)*0.012:
                ax.text(x+width/2, y, str(k), ha='center', va='center',
                        fontsize=6.5, color='white')
            else:
                ax.text(x+width/2, y-0.37, str(k), ha='center', va='bottom',
                        fontsize=6, color=colors[e['unit']])
        ax.set_yticks(range(4), lanes)
        ax.invert_yaxis()
        ax.set_xlim(0, hw.time_us(end))
        ax.set_xlabel('Time (microseconds)')
        ax.set_title(title, loc='left')
        ax.grid(axis='x', alpha=0.2)
    tab = axes[2]
    tab.axis('off')
    for k, e in enumerate(zoom_events):
        col, row = divmod(k, per_col)
        tab.text(col/ncol, 1 - row/per_col,
                 f'[{k+1:>2}] {e["unit"]:4s} @{hw.time_us(e["start"]):7.2f} us  '
                 f'{e["end"]-e["start"]:7,d} cyc  {e["name"]}',
                 transform=tab.transAxes, fontsize=7, family='monospace',
                 va='top', color=colors[e['unit']])
    mode = 'overlapped engines' if cfg['overlap'] else 'serial'
    title = result.get('title') or (
        f'FlashAttention | {hw.array_model} | {hw.control} control | {mode}\n'
        f'T={cfg["T"]}, S={cfg["S"]}, H={cfg["H"]}, G={cfg["G"]}, '
        f'Br={cfg["Br"]}, Bc={cfg["Bc"]}, R={cfg["R"]}, '
        f'SA1 rows QK/PV={cfg["split_qk"]}/{cfg["split_pv"]} | '
        f'{result["time_ms"]:.3f} ms | peak SRAM '
        f'{result["max_sram_utilization_percent"]:.2f}%')
    fig.suptitle(title, fontsize=13)
    fig.legend(handles=[Patch(color=c, label=u) for u,c in colors.items()],
               loc='outside lower center', ncol=4)
    fig.savefig(path, dpi=160)
    plt.close(fig)
