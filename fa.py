from functools import lru_cache

from tile_model import Hardware, VCPU, SA1, SA2, divup, DMA, SRAM, data_bytes

QK_LEAD = 2  # Arrays queue QK(p+QK_LEAD) before PV(p): PV lags two key blocks.
PACK_LOOKAHEAD = QK_LEAD + 1  # QK operands are packed this many blocks ahead.
UPDATE_LAG = 1  # U/l/m update of block p is queued after the pack work of p+1.


@lru_cache(maxsize=4096)
def _array_matmul(m, n, depth, rows_sa1, R, hw, diag=None, axis='cols'):
    """Schedule physical jobs; return per-array completion and traffic.

    Inputs are packed as row-dot operands. Each array has one compute stage
    and one output drain. The next compute can overlap the previous drain.
    Commands share one CPU interface. 'pipelined' assumes timed early issue;
    the issue interval is a user assumption, not a hardware specification.

    Causal skipping: with diag given, entry (row, x) is masked when
    x - row > diag, where x is the column (axis='cols', scores) or the
    reduction index (axis='depth', P.V with zero probabilities). A physical
    job is skipped when every entry it produces (or consumes) is masked.
    """
    arrays = [SA1(compute_model=hw.array_model), SA2(compute_model=hw.array_model)]
    row_counts = [min(m, rows_sa1), max(0, m - rows_sa1)]
    row_bases = [0, min(m, rows_sa1)]
    jobs, packed_bytes, raw_bytes, skipped = [], 0, 0, 0
    for sa, rows, base in zip(arrays, row_counts, row_bases):
        padded_rows = divup(rows, sa.physical_rows) * sa.physical_rows
        padded_cols = divup(n, 16) * 16
        packed_bytes += hw.digits * (padded_rows + padded_cols) * depth if rows else 0
        stream = []
        for r in range(0, rows, sa.physical_rows):
            last_row = base + r + min(sa.physical_rows, rows-r) - 1
            for c in range(0, n, 16):
                for k in range(0, depth, R):
                    first = c if axis == 'cols' else k
                    if diag is not None and first - last_row > diag:
                        skipped += hw.digits**2
                        continue
                    for _ in range(hw.digits**2):
                        stream.append((min(sa.physical_rows, rows-r),
                                       min(R, depth-k), min(16, n-c)))
        jobs.append(stream)
        raw_bytes += len(stream) * sa.output_bytes()

    cpu = 0
    ready, drain, pos, first = [0, 0], [0, 0], [0, 0], [None, None]
    while any(pos[a] < len(jobs[a]) for a in range(2)):
        candidates = [a for a in range(2) if pos[a] < len(jobs[a])]
        def issue_time(a):
            advance = hw.array_latency if hw.control == 'pipelined' else 0
            return max(cpu, ready[a] - advance)
        a = min(candidates, key=lambda a: (issue_time(a), a))
        issue = issue_time(a)
        arrival = issue + hw.array_latency
        first[a] = issue if first[a] is None else first[a]
        rows, k, cols = jobs[a][pos[a]]
        # compute() includes command + compute/input + final output drain.
        isolated = arrays[a].compute(rows, k, 8, cols, hw=hw)
        out = arrays[a].output_cycles()
        stage = isolated - hw.array_latency - out
        # If the drain is occupied, hold the new result until it becomes free.
        ready[a] = max(arrival + stage, drain[a])
        drain[a] = ready[a] + out
        cpu = issue + (hw.issue_interval if hw.control == 'pipelined'
                       else hw.array_latency)
        pos[a] += 1
    return dict(start=first, end=drain, jobs=[len(x) for x in jobs],
                skipped=skipped, rows_sa1=row_counts[0],
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


def flash_attention(T=2048, S=2048, H=128, G=4, Br=256, Bc=256,
                    Br_sa1=None, R=128, causal=True, q_offset=0,
                    hw=Hardware(), overlap=True, skip_masked=True,
                    verbose=True, gantt_path=None):
    """Time ONE KV-head group in ONE layer and ONE batch element.

    G Q heads share K/V, which are read/quantized once and kept in SRAM.
    For the whole model, serial group count = B * L * K (K = KV heads).
    Here R is the array reduction chunk, not the number of KV heads.
    Tail blocks are supported. Causal positions: key <= q_offset + query.

    Execution model. The control CPU only issues commands; DMA, the vector
    CPU and the array pair (SA1+SA2 interleaved on the shared command
    interface) are three engines with in-order queues. A command starts
    when its engine is free and every producer it depends on has finished
    (event-based ordering). With overlap=False each command also waits for
    the previous one, i.e. the fully serial schedule of the earlier model.

    Software pipeline (overlap=True), for the global sequence of
    (query tile, key block) pairs p:
        arrays : QK(p+QK_LEAD) is queued before PV(p), so the softmax of
                 block p is hidden under QK_LEAD score GEMMs (short causal
                 diagonal blocks included); PV lags QK_LEAD key blocks.
        vector : per block p: reconstruct/softmax/pack P(p), pack the QK
                 operands of p+PACK_LOOKAHEAD, then reconstruct/update
                 U,l,m of block p-UPDATE_LAG (so the in-order vector queue
                 does not stall on PV(p) before packing the next blocks).
                 Q of the next tile is loaded and quantized when its first
                 pack is issued (two Q16 buffers).
        setup  : K is quantized first so QK(0) can start; V is converted
                 while QK(0) runs; O16 stores overlap the next tile.
    Data dependencies are explicit; row statistics chain per tile.

    Array work split. Br_sa1 rows of every GEMM go to SA1 and the rest to
    SA2. Br_sa1=None picks, per GEMM geometry, the multiple of 16 that
    minimizes the two-array makespan (this also balances the causal
    diagonal blocks and tail tiles). Causal blocks are trimmed to the
    needed columns (16-column granularity) and physical jobs whose whole
    output (QK) or whole P input (PV) is masked are skipped when
    skip_masked is set; the VCPU traffic shrinks with the block width.

    safe-int8: quantize A16 to A8, then pack two signed base-16 digits,
    four products and INT32 reconstruction, reduction <=128. Digit operands
    have magnitude <=15; raw INT16 partials are safe. native-int8 is only a
    timing comparison and requires a suitable hardware output mechanism.
    Scaled INT16 inputs have known external scales. Q/K/V use scalar INT8
    scales; P uses fixed 1/127. Final INT16 uses a caller-supplied output scale.

    VCPU routines fuse row operations with row data kept in vector registers;
    byte counts below explicitly state passes. V transpose is fused with
    conversion. Packed operands duplicate the right operand across arrays;
    full physical raw outputs are retained until reconstruction. Registers
    and unknown array-local storage are excluded from SRAM utilization.
    SRAM occupancy is replayed from the timed schedule (buffers of
    in-flight blocks coexist). This executes metadata and timing only,
    not numerical attention.
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

    # ---- in-order engine queues -----------------------------------------
    tasks = []
    unit_free = {'DMA': 0, 'VCPU': 0, 'ARRAYS': 0}

    def task(name, unit, duration, deps=(), alloc=(), free=(), info=None):
        deps = [d for d in deps if d is not None]
        if not overlap and tasks:
            deps.append(tasks[-1]['id'])
        start = max([unit_free[unit]] + [tasks[d]['end'] for d in deps])
        rec = dict(id=len(tasks), name=name, unit=unit, start=start,
                   end=start+duration, cycles=duration, deps=deps,
                   alloc=list(alloc), free=list(free), info=info)
        unit_free[unit] = rec['end']
        tasks.append(rec)
        return rec['id']

    def vector(name, reads, writes, **kw):
        return task(name, 'VCPU', VCPU(reads, writes, hw).launch_cycle(), **kw)

    def dma(name, shape, bits, direction, **kw):
        return task(name, 'DMA', DMA(hw=hw).transfer(shape, bits, direction), **kw)

    def nbytes(shape, dtype):
        return data_bytes(shape, dtype)

    # K/V loaded once for all G query heads.
    dK = dma('Load K16', (S,H), 16, 'read', alloc=[('K16', nbytes((S,H),16))])
    qK = vector('Quantize K', 4*S*H, S*H+4, deps=[dK],  # Two A16 reads.
                alloc=[('K8', nbytes((S,H),8)), ('K_scale', 4)], free=['K16'])
    qV = None

    tile_ready, packed, qk_done, pv_done = {}, {}, {}, {}
    prev_softmax, prev_update = {}, {}

    def ensure_tile(t):
        if t in tile_ready:
            return
        g, i, m = tiles[t]
        tag = f'h{g} q{i}'
        d = dma(tag+' load Q16', (m,H), 16, 'read', deps=[tile_ready.get(t-1)],
                alloc=[(f'Q16@t{t}', nbytes((m,H),16))])
        tile_ready[t] = vector(
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
        pid = vector(pr['tag']+' QK pack', (m+c)*H, timing['packed_bytes'],
                     deps=[tile_ready[pr['tile']], qK],
                     alloc=[(f'QKdig@{p}', timing['packed_bytes'])])
        packed[p] = (pid, timing)

    def emit_arrays(p, label, timing, deps, digits, raw):
        return task(pairs[p]['tag']+' '+label, 'ARRAYS', max(timing['end']),
                    deps=deps, alloc=[(raw, timing['raw_bytes'])],
                    free=[digits], info=timing)

    def emit_qk(p):
        pid, timing = packed[p]
        qk_done[p] = emit_arrays(p, 'QK', timing, [pid], f'QKdig@{p}', f'QKraw@{p}')

    def emit_softmax_pv(p):
        pr = pairs[p]
        t, m, c, tag = pr['tile'], pr['m'], pr['c'], pr['tag']
        raw_qk = tasks[qk_done[p]]['info']['raw_bytes']
        # Reconstruct in vector registers, discard padding, then scale to FP32.
        rec = vector(tag+' QK reconstruct', raw_qk+8, 4*m*c, deps=[qk_done[p]],
                     alloc=[(f'scores@{p}', 4*m*c)], free=[f'QKraw@{p}'])
        # Two FP32 score reads: rowmax, then exp/sum/P8 encoding.
        # m_new=max(m,rowmax(S)); alpha=exp(m-m_new).
        # l_new=alpha*l+sum(exp(S-m_new)); masked P entries are zero.
        sm = vector(tag+' softmax + P8', 8*m*c+8*m, m*c+12*m,
                    deps=[rec, prev_softmax.get(t)],
                    alloc=[(f'P8@{p}', m*c), (f'stats@{p}', 12*m)],
                    free=[f'scores@{p}'])
        prev_softmax[t] = sm
        timing = _array_matmul(m, H, c, split(m, H, c, pr['diag'], 'depth'),
                               R, hw, pr['diag'], 'depth')
        pk = vector(tag+' PV pack', (m+H)*c, timing['packed_bytes'],
                    deps=[sm, qV], alloc=[(f'PVdig@{p}', timing['packed_bytes'])],
                    free=[f'P8@{p}'])
        return pk, timing

    def emit_pv(p, pk, timing):
        pv_done[p] = emit_arrays(p, 'PV', timing, [pk], f'PVdig@{p}', f'PVraw@{p}')

    def emit_update(p):
        pr = pairs[p]
        t, m, tag = pr['tile'], pr['m'], pr['tag']
        raw_pv = tasks[pv_done[p]]['info']['raw_bytes']
        rec = vector(tag+' PV reconstruct', raw_pv+4, 4*m*H, deps=[pv_done[p]],
                     alloc=[(f'C@{p}', 4*m*H)], free=[f'PVraw@{p}'])
        # U <- alpha*U + C; l <- l_new; m <- m_new.
        prev_update[t] = vector(tag+' update U,l,m', 8*m*H+12*m, 4*m*H+8*m,
                                deps=[rec, prev_update.get(t), prev_softmax[t]],
                                free=[f'C@{p}', f'stats@{p}'])
        if last_pair_of_tile[t] == p:
            g, i, m = tiles[t]
            ttag = f'h{g} q{i}'
            norm = vector(ttag+' normalize + encode O16', 4*m*H+4*m, 2*m*H,
                          deps=[prev_update[t]], alloc=[(f'O16@t{t}', 2*m*H)],
                          free=[f'Q8@t{t}', f'Q_scale@t{t}', f'U@t{t}',
                                f'l@t{t}', f'm@t{t}'])
            dma(ttag+' store O16', (m,H), 16, 'write', deps=[norm],
                free=[f'O16@t{t}'])

    dV = dma('Load V16', (S,H), 16, 'read', alloc=[('V16', nbytes((S,H),16))])

    def emit_v():
        nonlocal qV
        qV = vector('Quantize V + transpose', 4*S*H, S*H+4, deps=[dV],
                    alloc=[('V8T', nbytes((H,S),8)), ('V_scale', 4)], free=['V16'])

    lead, lag = (QK_LEAD, UPDATE_LAG) if overlap else (0, 0)
    if overlap:
        for q in range(min(PACK_LOOKAHEAD, P)):
            emit_pack(q)
        for q in range(min(lead, P)):
            emit_qk(q)
    emit_v()
    for p in range(P):
        if not overlap:
            emit_pack(p)
        if p+lead < P:
            emit_qk(p+lead)
        pk, timing = emit_softmax_pv(p)
        if overlap:
            emit_pack(p+PACK_LOOKAHEAD)
        emit_pv(p, pk, timing)
        if p-lag >= 0:
            emit_update(p-lag)
    for p in range(max(0, P-lag), P):
        emit_update(p)
    kv_setup_cycles = tasks[qV]['end']
    cycles = max(t['end'] for t in tasks)

    # ---- replay SRAM occupancy in time order (frees, allocs, probes) ----
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

    # ---- bookkeeping ------------------------------------------------------
    counts = dict(query_tiles=len(tiles), pairs=P, DMA=0, VCPU=0,
                  SA1_jobs=0, SA2_jobs=0, skipped_jobs=0)
    breakdown, events, steps = {}, [], []
    for t in sorted(tasks, key=lambda t: (t['start'], t['id'])):
        step = dict(name=t['name'], unit=t['unit'], start=t['start'], end=t['end'],
                    cycles=t['cycles'], used_bytes=t['used_bytes'])
        if t['unit'] != 'ARRAYS':
            counts[t['unit']] += 1
            breakdown[t['unit']] = breakdown.get(t['unit'], 0) + t['cycles']
            events.append(step.copy())
        else:
            timing = t['info']
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
                timing = t['info']
                for a in range(2):
                    if timing['jobs'][a]:
                        elapsed = timing['end'][a] - timing['start'][a]
                        print(f'  SA{a+1}: {timing["jobs"][a]:,} commands '
                              f'({timing["rows_sa1"] if a == 0 else "rest"} rows), '
                              f'{elapsed:,} cycles ({hw.time_us(elapsed):.3f} us); '
                              f'starts +{timing["start"][a]} cycles in GEMM')
    first_tile_end = next(t['end'] for t in tasks if t['name'].endswith('store O16'))
    full = pairs[0]
    result = dict(cycles=cycles, time_us=hw.time_us(cycles),
                  time_ms=hw.time_us(cycles)/1000, kv_setup_cycles=kv_setup_cycles,
                  query_work_cycles=cycles-kv_setup_cycles, counts=counts,
                  breakdown=breakdown,
                  utilization={u: b/cycles for u, b in breakdown.items()},
                  steps=steps, events=events, sram=sram, peak_sram_bytes=peak,
                  max_sram_utilization_percent=100*peak/sram.capacity_bytes,
                  first_tile_end=first_tile_end, hw=hw,
                  config=dict(T=T,S=S,H=H,G=G,Br=Br,Bc=Bc,Br_sa1=Br_sa1,R=R,
                              overlap=overlap, skip_masked=skip_masked,
                              split_qk=split(full['m'], full['c'], H, None, 'cols'),
                              split_pv=split(full['m'], H, full['c'], None, 'depth')))
    if verbose:
        print(f'\nTotal: {result["time_ms"]:.6f} ms ({cycles:,} cycles)')
        print('Busy: ' + ', '.join(f'{u} {100*b/cycles:.1f}%'
                                    for u, b in breakdown.items()))
        print(f'Peak SRAM: {peak:,} bytes ({100*peak/sram.capacity_bytes:.3f}%)')
    if gantt_path is not None:
        plot_flash_attention(result, gantt_path)
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
