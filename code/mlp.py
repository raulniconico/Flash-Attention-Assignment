from fa import _Queue, _array_matmul, _balanced_split, plot_flash_attention
from tile_model import Hardware, VCPU, DMA, SRAM

DOWN_LAG = 1  # Arrays queue gate+up(f+1) before down(f): Z(f) is hidden.
ACC_LAG = 1   # Y accumulation of f is queued after the gate+up work of f+2.


def _replay(tasks, hw):
    """Replay the timed allocs/frees/probes; record used_bytes, return peak."""
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


def _collect(tasks, counts, hw, verbose):
    """Busy cycles per queue, job counts, Gantt events and the step table."""
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
            for a in range(2):
                counts[f'SA{a+1}_jobs'] += timing['jobs'][a]
                if timing['jobs'][a]:
                    events.append(dict(name=t['name'], unit=f'SA{a+1}',
                                       start=t['start']+timing['start'][a],
                                       end=t['start']+timing['end'][a]))
        steps.append(step)
        if verbose:
            print(f'{t["name"]:42s} {t["unit"]:6s} @{t["start"]:>13,d} '
                  f'{t["cycles"]:9,d} cycles {hw.time_us(t["cycles"]):10.3f} us  '
                  f'SRAM {t["used_bytes"]/1024:9.2f} KiB')
    return breakdown, events, steps


def mlp(M=2048, D=4096, F=14336, Mt=192, Ft=256, Mt_sa1=None, R=256,
        residual=True, hw=Hardware(), overlap=True, verbose=True,
        gantt_path=None):

    for value in (M, D, F, Mt, Ft, R):
        if not isinstance(value, int) or value <= 0:
            raise ValueError('Dimensions and R must be positive integers')
    if Mt_sa1 is not None and not 0 <= Mt_sa1 <= Mt:
        raise ValueError('Mt_sa1 must be None (auto) or between 0 and Mt')

    def split(m, n, depth):
        if Mt_sa1 is None:
            return _balanced_split(m, n, depth, R, hw)
        return min(Mt_sa1, m)

    tiles = [(i, min(Mt, M-i)) for i in range(0, M, Mt)]
    ftiles = [(j, min(Ft, F-j)) for j in range(0, F, Ft)]
    nF = len(ftiles)
    pairs = [(t, f) for t in range(len(tiles)) for f in range(nF)]

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

    def dma(name, nbytes, direction, **kw):
        return task(name, 'DMA', DMA(nbytes, direction, hw).cycle(), **kw)

    def arrays(name, timing, **kw):
        return task(name, 'ARRAYS', max(timing['end']), info=timing, **kw)

    x_ready, w_gu, w_d, gateup_done, down_done, z_ready = {}, {}, {}, {}, {}, {}
    acc_done, down_raw, last_gateup = {}, {}, [None]

    def ensure_tile(t):
        if t in x_ready:
            return
        i, m = tiles[t]
        d = dma(f'x{i} load X16', 2*m*D, 'read',
                deps=[x_ready.get(t-1), last_gateup[0]],
                alloc=[(f'X16@{t}', 2*m*D)])
        x_ready[t] = vector(f'x{i} quantize X -> X8 operands', 2*m*D, m*D,
                            deps=[d], alloc=[(f'X8@{t}', m*D)], free=[f'X16@{t}'])

    def prev_pair(p, back):
        return p-back if p-back >= 0 else None

    def load_gu(p):
        if p >= len(pairs) or p in w_gu:
            return
        t, f = pairs[p]
        j, n = ftiles[f]
        w_gu[p] = dma(f'x{tiles[t][0]} f{j} load Wg|Wu', hw.digits*D*2*n, 'read',
                      deps=[gateup_done.get(prev_pair(p, 2))],
                      alloc=[(f'Wgu@{p}', hw.digits*D*2*n), (f'sW@{p}', 8*n+4*D)])

    def load_d(p):
        if p >= len(pairs) or p in w_d:
            return
        t, f = pairs[p]
        j, n = ftiles[f]
        w_d[p] = dma(f'x{tiles[t][0]} f{j} load Wd', hw.digits*n*D, 'read',
                     deps=[down_done.get(prev_pair(p, 2))],
                     alloc=[(f'Wd@{p}', hw.digits*n*D)])

    def emit_gateup(p):
        """Chunked GEMM: arrays chunk k, then a vector accumulate of chunk k."""
        t, f = pairs[p]
        i, m = tiles[t]
        j, n = ftiles[f]
        ensure_tile(t)
        load_gu(p)
        chunks = list(range(0, D, R))
        acc = None
        for ci, k in enumerate(chunks):
            kc, last = min(R, D-k), ci == len(chunks)-1
            timing = _array_matmul(m, 2*n, kc, split(m, 2*n, kc), R, hw)
            a = arrays(f'x{i} f{j} gate+up k{k}', timing,
                       deps=[x_ready[t], w_gu[p]],
                       alloc=[(f'GUraw@{p}k{k}', timing['raw_bytes'])],
                       free=[f'Wgu@{p}'] if last else [])
            last_gateup[0] = a
            if last:
                # Final pass: add last partials, apply scales, SiLU(G)*U,
                # static-scale quantize, write Z8 in operand layout.
                gu = 0 if ci == 0 else 4*m*2*n
                acc = vector(f'x{i} f{j} G,U -> SiLU(G)*U -> Z8',
                             timing['raw_bytes']+gu+8*n, m*n, deps=[a, acc],
                             alloc=[(f'Z8@{p}', m*n)],
                             free=[f'GUraw@{p}k{k}'] + ([] if ci == 0 else [f'GU32@{p}']))
            else:
                acc = vector(f'x{i} f{j} G,U32 {"=" if ci == 0 else "+="} k{k}',
                             timing['raw_bytes']+(0 if ci == 0 else 4*m*2*n),
                             4*m*2*n, deps=[a, acc],
                             alloc=[(f'GU32@{p}', 4*m*2*n)] if ci == 0 else [],
                             free=[f'GUraw@{p}k{k}'])
        gateup_done[p], z_ready[p] = a, acc

    def emit_down(p):
        t, f = pairs[p]
        i, m = tiles[t]
        j, n = ftiles[f]
        load_d(p)
        for ci, k in enumerate(range(0, n, R)):
            kc, last = min(R, n-k), k+R >= n
            timing = _array_matmul(m, D, kc, split(m, D, kc), R, hw)
            down_raw[(p, k)] = timing['raw_bytes']
            down_done[p] = arrays(
                f'x{i} f{j} down k{k}', timing, deps=[z_ready[p], w_d[p]],
                alloc=[(f'Yraw@{p}k{k}', timing['raw_bytes'])],
                free=[f'Z8@{p}', f'Wd@{p}'] if last else [])

    def emit_accumulate(p):
        t, f = pairs[p]
        i, m = tiles[t]
        j, n = ftiles[f]
        for ci, k in enumerate(range(0, n, R)):
            new = f == 0 and ci == 0
            last = k+R >= n
            raw = down_raw[(p, k)]
            acc_done[t] = vector(
                f'x{i} f{j} Y32 {"=" if new else "+="} k{k}',
                raw+(0 if new else 4*m*D)+(4*D if last else 0), 4*m*D,
                deps=[down_done[p], acc_done.get(t)],
                alloc=[(f'Y32@{t}', 4*m*D)] if new else [],
                free=[f'Yraw@{p}k{k}', f'sW@{p}'] if last else [f'Yraw@{p}k{k}'])
        if f == nF-1:
            deps = [acc_done[t]]
            if residual:
                deps.append(dma(f'x{i} load residual X16', 2*m*D, 'read',
                                alloc=[(f'Xres@{t}', 2*m*D)]))
            enc = vector(f'x{i} encode Y16{" + residual" if residual else ""}',
                         4*m*D+4+(2*m*D if residual else 0), 2*m*D, deps=deps,
                         alloc=[(f'Y16@{t}', 2*m*D)],
                         free=[f'Y32@{t}', f'X8@{t}'] + ([f'Xres@{t}'] if residual else []))
            dma(f'x{i} store Y16', 2*m*D, 'write', deps=[enc], free=[f'Y16@{t}'])

    lead, lag = (DOWN_LAG, ACC_LAG) if overlap else (0, 0)
    zoom_end = None
    for p in range(len(pairs)):
        if p == 0:
            for q in range(min(lead, len(pairs))):
                emit_gateup(q)
        if p+lead < len(pairs):
            emit_gateup(p+lead)
        if overlap:
            load_d(p+lead)  # prefetch into the slot freed by down(p-2+lead)
        emit_down(p)
        if p-lag >= 0:
            emit_accumulate(p-lag)
            zoom_end = tasks[-1]['end'] if zoom_end is None else zoom_end
    for p in range(max(0, len(pairs)-lag), len(pairs)):
        emit_accumulate(p)
    cycles = max(t['end'] for t in tasks)

    sram, peak = _replay(tasks, hw)
    counts = dict(token_tiles=len(tiles), feature_tiles=nF, DMA=0, VCPU=0,
                  SA1_jobs=0, SA2_jobs=0)
    breakdown, events, steps = _collect(tasks, counts, hw, verbose)
    macs = 3*M*D*F
    weight_bytes = hw.digits*3*D*F*len(tiles)
    first_tile_end = next(t['end'] for t in tasks if t['name'].endswith('store Y16'))
    m0, n0 = tiles[0][1], ftiles[0][1]
    result = dict(cycles=cycles, time_us=hw.time_us(cycles),
                  time_ms=hw.time_us(cycles)/1000, counts=counts, breakdown=breakdown,
                  utilization={u: b/cycles for u, b in breakdown.items()},
                  macs=macs, mac_per_cycle=macs/cycles,
                  weight_dram_bytes=weight_bytes, steps=steps, events=events,
                  sram=sram, peak_sram_bytes=peak,
                  max_sram_utilization_percent=100*peak/sram.capacity_bytes,
                  first_tile_end=first_tile_end, zoom_end=zoom_end, hw=hw,
                  panel_titles=['Complete MLP layer',
                                'Cold setup + first down projection'],
                  config=dict(M=M, D=D, F=F, Mt=Mt, Ft=Ft, Mt_sa1=Mt_sa1, R=R,
                              residual=residual, overlap=overlap,
                              split_gateup=split(m0, 2*n0, min(R, D)),
                              split_down=split(m0, D, min(R, n0))))
    cfg = result['config']
    result['title'] = (
        f'SwiGLU MLP | {hw.array_model} | {hw.control} control | '
        f'{"overlapped engines" if overlap else "serial"}\n'
        f'M={M}, D={D}, F={F}, Mt={Mt}, Ft={Ft}, R={R}, SA1 rows gate+up/down='
        f'{cfg["split_gateup"]}/{cfg["split_down"]} | {result["time_ms"]:.3f} ms | '
        f'{macs/cycles:,.0f} MAC/cycle | peak SRAM '
        f'{result["max_sram_utilization_percent"]:.2f}%')
    if verbose:
        print(f'\nTotal: {result["time_ms"]:.6f} ms ({cycles:,} cycles), '
              f'{macs/cycles:,.0f} MAC/cycle')
        print('Busy: ' + ', '.join(f'{u} {100*b/cycles:.1f}%'
                                    for u, b in breakdown.items()))
        print(f'Weights streamed: {weight_bytes/2**20:,.0f} MiB '
              f'({len(tiles)} token tiles)')
        print(f'Peak SRAM: {peak:,} bytes ({100*peak/sram.capacity_bytes:.3f}%)')
    if gantt_path is not None:
        plot_flash_attention(result, gantt_path)
    return result


def mlp_decode(M=16, D=4096, F=14336, Ft=256, Nt_sa1=None, R=256,
               residual=True, hw=Hardware(), overlap=True, verbose=True,
               gantt_path=None):

    for value in (M, D, F, Ft, R):
        if not isinstance(value, int) or value <= 0:
            raise ValueError('Dimensions and R must be positive integers')
    if R > hw.max_reduction:
        raise ValueError(f'R must be <= {hw.max_reduction} for {hw.arithmetic}')
    if Nt_sa1 is not None and (not isinstance(Nt_sa1, int) or Nt_sa1 < 0):
        raise ValueError('Nt_sa1 must be None (auto) or a row count >= 0')

    def split(m, n, depth):
        if Nt_sa1 is None:
            return _balanced_split(m, n, depth, R, hw)
        return min(Nt_sa1, m)

    ftiles = [(j, min(Ft, F-j)) for j in range(0, F, Ft)]
    nF = len(ftiles)

    q = _Queue(hw, overlap)
    tasks = q.tasks

    def arrays(name, timing, **kw):
        return q.task(name, 'ARRAYS', max(timing['end']), info=timing, **kw)

    # X16 [M, D] arrives once and is quantized into the transposed operand
    # layout X8^T [D, M]; both are a few hundred KiB at decode batch sizes.
    dX = q.dma('load X16', 2*M*D, 'read', alloc=[('X16', 2*M*D)])
    x8 = q.vector('quantize X -> X8^T operands', 2*M*D, M*D, deps=[dX],
                  alloc=[('X8', M*D)], free=['X16'])

    w_gu, w_d, gateup_done, down_done, z_ready, down_raw = {}, {}, {}, {}, {}, {}
    acc_done = [None]

    def load_gu(f):
        if f >= nF or f in w_gu:
            return
        j, n = ftiles[f]
        w_gu[f] = q.dma(f'f{j} load Wg|Wu^T', hw.digits*2*n*D, 'read',
                        deps=[gateup_done.get(f-2)],
                        alloc=[(f'Wgu@{f}', hw.digits*2*n*D), (f'sW@{f}', 8*n+4*D)])

    def load_d(f):
        if f >= nF or f in w_d:
            return
        j, n = ftiles[f]
        w_d[f] = q.dma(f'f{j} load Wd^T', hw.digits*D*n, 'read',
                       deps=[down_done.get(f-2)],
                       alloc=[(f'Wd@{f}', hw.digits*D*n)])

    def emit_gateup(f):
        """2Ft rows of G|U for the M tokens, chunked over the D reduction."""
        j, n = ftiles[f]
        load_gu(f)
        chunks = list(range(0, D, R))
        acc = None
        for ci, k in enumerate(chunks):
            kc, last = min(R, D-k), ci == len(chunks)-1
            timing = _array_matmul(2*n, M, kc, split(2*n, M, kc), R, hw)
            a = arrays(f'f{j} gate+up k{k}', timing, deps=[x8, w_gu[f]],
                       alloc=[(f'GUraw@{f}k{k}', timing['raw_bytes'])],
                       free=[f'Wgu@{f}'] if last else [])
            if last:
                # Final pass: add last partials, apply scales, SiLU(G)*U,
                # static-scale quantize, write Z8^T in operand layout.
                gu = 0 if ci == 0 else 4*2*n*M
                acc = q.vector(f'f{j} G,U -> SiLU(G)*U -> Z8^T',
                               timing['raw_bytes']+gu+8*n, n*M, deps=[a, acc],
                               alloc=[(f'Z8@{f}', n*M)],
                               free=[f'GUraw@{f}k{k}'] + ([] if ci == 0 else [f'GU32@{f}']))
            else:
                acc = q.vector(f'f{j} G,U32 {"=" if ci == 0 else "+="} k{k}',
                               timing['raw_bytes']+(0 if ci == 0 else 4*2*n*M),
                               4*2*n*M, deps=[a, acc],
                               alloc=[(f'GU32@{f}', 4*2*n*M)] if ci == 0 else [],
                               free=[f'GUraw@{f}k{k}'])
        gateup_done[f], z_ready[f] = a, acc

    def emit_down(f):
        """All D rows of Y for the M tokens, reduction = this Ft tile."""
        j, n = ftiles[f]
        load_d(f)
        for k in range(0, n, R):
            kc, last = min(R, n-k), k+R >= n
            timing = _array_matmul(D, M, kc, split(D, M, kc), R, hw)
            down_raw[(f, k)] = timing['raw_bytes']
            down_done[f] = arrays(f'f{j} down k{k}', timing,
                                  deps=[z_ready[f], w_d[f]],
                                  alloc=[(f'Yraw@{f}k{k}', timing['raw_bytes'])],
                                  free=[f'Z8@{f}', f'Wd@{f}'] if last else [])

    def emit_accumulate(f):
        j, n = ftiles[f]
        for ci, k in enumerate(range(0, n, R)):
            new = f == 0 and ci == 0
            last = k+R >= n
            raw = down_raw[(f, k)]
            acc_done[0] = q.vector(
                f'f{j} Y32 {"=" if new else "+="} k{k}',
                raw+(0 if new else 4*D*M)+(4*D if last else 0), 4*D*M,
                deps=[down_done[f], acc_done[0]],
                alloc=[('Y32', 4*D*M)] if new else [],
                free=[f'Yraw@{f}k{k}', f'sW@{f}'] if last else [f'Yraw@{f}k{k}'])
        if f == nF-1:
            deps = [acc_done[0]]
            if residual:
                deps.append(q.dma('load residual X16', 2*M*D, 'read',
                                  alloc=[('Xres', 2*M*D)]))
            enc = q.vector(f'encode Y16{" + residual" if residual else ""}',
                           4*D*M+4+(2*M*D if residual else 0), 2*M*D, deps=deps,
                           alloc=[('Y16', 2*M*D)],
                           free=['Y32', 'X8'] + (['Xres'] if residual else []))
            q.dma('store Y16', 2*M*D, 'write', deps=[enc], free=['Y16'])

    lead, lag = (DOWN_LAG, ACC_LAG) if overlap else (0, 0)
    zoom_end = None
    for f in range(nF):
        if f == 0:
            for g in range(min(lead, nF)):
                emit_gateup(g)
        if f+lead < nF:
            emit_gateup(f+lead)
        if overlap:
            load_d(f+lead)  # prefetch into the slot freed by down(f-2+lead)
        emit_down(f)
        if f-lag >= 0:
            emit_accumulate(f-lag)
            zoom_end = tasks[-1]['end'] if zoom_end is None else zoom_end
    for f in range(max(0, nF-lag), nF):
        emit_accumulate(f)
    cycles = max(t['end'] for t in tasks)

    sram, peak = _replay(tasks, hw)
    counts = dict(token_tiles=1, feature_tiles=nF, DMA=0, VCPU=0,
                  SA1_jobs=0, SA2_jobs=0)
    breakdown, events, steps = _collect(tasks, counts, hw, verbose)
    macs = 3*M*D*F
    weight_bytes = hw.digits*3*D*F
    n0 = ftiles[0][1]
    result = dict(cycles=cycles, time_us=hw.time_us(cycles),
                  time_ms=hw.time_us(cycles)/1000, counts=counts, breakdown=breakdown,
                  utilization={u: b/cycles for u, b in breakdown.items()},
                  macs=macs, mac_per_cycle=macs/cycles,
                  weight_dram_bytes=weight_bytes,
                  dma_bound_cycles=weight_bytes//hw.dma_bw,
                  us_per_token=hw.time_us(cycles)/M, steps=steps, events=events,
                  sram=sram, peak_sram_bytes=peak,
                  max_sram_utilization_percent=100*peak/sram.capacity_bytes,
                  first_tile_end=cycles, zoom_end=zoom_end, hw=hw,
                  panel_titles=['Complete MLP decode step',
                                'Cold setup + first down projection'],
                  config=dict(M=M, D=D, F=F, Mt=M, Ft=Ft, Mt_sa1=Nt_sa1, R=R,
                              residual=residual, overlap=overlap,
                              split_gateup=split(2*n0, M, min(R, D)),
                              split_down=split(D, M, min(R, n0))))
    cfg = result['config']
    result['title'] = (
        f'SwiGLU MLP decode step | {hw.array_model} | {hw.control} control | '
        f'{"overlapped engines" if overlap else "serial"}\n'
        f'M={M}, D={D}, F={F}, Ft={Ft}, R={R}, SA1 rows gate+up/down='
        f'{cfg["split_gateup"]}/{cfg["split_down"]} | {result["time_ms"]:.3f} ms | '
        f'{macs/cycles:,.0f} MAC/cycle | peak SRAM '
        f'{result["max_sram_utilization_percent"]:.2f}%')
    if verbose:
        print(f'\nTotal: {result["time_ms"]:.6f} ms ({cycles:,} cycles), '
              f'{macs/cycles:,.0f} MAC/cycle, '
              f'{result["us_per_token"]:.1f} us/token/layer')
        print('Busy: ' + ', '.join(f'{u} {100*b/cycles:.1f}%'
                                    for u, b in breakdown.items()))
        print(f'Weights streamed: {weight_bytes/2**20:,.0f} MiB per step '
              f'-> DMA floor {hw.time_us(result["dma_bound_cycles"])/1000:.3f} ms')
        print(f'Peak SRAM: {peak:,} bytes ({100*peak/sram.capacity_bytes:.3f}%)')
    if gantt_path is not None:
        plot_flash_attention(result, gantt_path)
    return result


plot_mlp = plot_flash_attention
