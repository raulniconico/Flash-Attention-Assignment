from functools import lru_cache

from tile_model import Hardware, VCPU, SA1, divup, DMA, SRAM, SA2


@lru_cache(maxsize=512)
def _array_matmul(m, n, depth, rows_sa1, R, hw):
    """Schedule physical jobs; return per-array completion and traffic.

    Inputs are packed as row-dot operands. Each array has one compute stage
    and one output drain. The next compute can overlap the previous drain.
    Commands share one CPU interface. 'pipelined' assumes timed early issue;
    the issue interval is a user assumption, not a hardware specification.
    """
    arrays = [SA1(compute_model=hw.array_model), SA2(compute_model=hw.array_model)]
    row_counts = [min(m, rows_sa1), max(0, m - rows_sa1)]
    jobs, packed_bytes, raw_bytes = [], 0, 0
    for sa, rows in zip(arrays, row_counts):
        padded_rows = divup(rows, sa.physical_rows) * sa.physical_rows
        padded_cols = divup(n, 16) * 16
        packed_bytes += hw.digits * (padded_rows + padded_cols) * depth if rows else 0
        stream = []
        for r in range(0, rows, sa.physical_rows):
            for c in range(0, n, 16):
                for k in range(0, depth, R):
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
                packed_bytes=packed_bytes, raw_bytes=raw_bytes)


def flash_attention(T=2048, S=2048, H=128, G=4, Br=256, Bc=256,
                    Br_sa1=None, R=128, causal=True, q_offset=0,
                    hw=Hardware(), verbose=True, gantt_path=None):
    """Time ONE KV-head group in ONE layer and ONE batch element.

    G Q heads share K/V, which are read/quantized once and kept in SRAM.
    For the whole model, serial group count = B * L * K (K = KV heads).
    Here R is the array reduction chunk, not the number of KV heads.
    Br_sa1 rows go to SA1 and Br-Br_sa1 to SA2; default is Br//2.
    Tail blocks are supported. Causal positions: key <= q_offset + query.

    Phases are serial; the two arrays overlap inside each GEMM, including
    computation/output draining. No DMA/VCPU/inter-block overlap is modeled.
    Each physical array command pays its configured control cost.

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
    This executes metadata and timing only, not numerical attention.
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
    # if R > hw.max_reduction:
    #     raise ValueError(f'R must be <= {hw.max_reduction} for {hw.arithmetic}')
    Br_sa1 = Br//2 if Br_sa1 is None else Br_sa1
    if not isinstance(Br_sa1, int) or not 0 <= Br_sa1 <= Br:
        raise ValueError('Br_sa1 must be between 0 and Br')

    dma, sram = DMA(hw=hw), SRAM(hw=hw)
    cycles, steps, events, breakdown = 0, [], [], {}
    counts = dict(query_tiles=0, pairs=0, DMA=0, VCPU=0, SA1_jobs=0, SA2_jobs=0)
    first_tile_end = None

    def record(name, unit, duration):
        nonlocal cycles
        step = dict(name=name, unit=unit, start=cycles, end=cycles+duration,
                    cycles=duration, used_bytes=sram.used_bytes)
        steps.append(step)
        events.append(step.copy())
        breakdown[unit] = breakdown.get(unit, 0) + duration
        if unit in ('DMA', 'VCPU'):
            counts[unit] += 1
        cycles += duration
        if verbose:
            print(f'{name:42s} {duration:10,d} cycles  '
                  f'{hw.time_us(duration):11.3f} us  '
                  f'SRAM {sram.used_bytes/1024:9.2f} KiB')

    def vector(name, reads, writes):
        record(name, 'VCPU', VCPU(reads, writes, hw).launch_cycle())

    def free(*names):
        for name in names:
            sram.free(name)

    def matmul(label, m, n, depth, output, scale_bytes):
        nonlocal cycles
        timing = _array_matmul(m, n, depth, Br_sa1, R, hw)
        sram.allocate('digits', timing['packed_bytes'], 8)
        # Read each logical operand once; write digit planes including padding.
        vector(label+' pack', (m+n)*depth, timing['packed_bytes'])
        sram.allocate('raw', timing['raw_bytes'], 8)
        start = cycles
        record(label+' arrays', 'SA1+SA2', max(timing['end']))
        events.pop()  # Replace aggregate with concurrent per-array envelopes.
        for a in range(2):
            counts[f'SA{a+1}_jobs'] += timing['jobs'][a]
            if timing['jobs'][a]:
                if verbose:
                    elapsed = timing['end'][a] - timing['start'][a]
                    print(f'  SA{a+1}: {timing["jobs"][a]:,} commands, '
                          f'{elapsed:,} cycles ({hw.time_us(elapsed):.3f} us); '
                          f'starts +{timing["start"][a]} cycles in GEMM')
                events.append(dict(name=label, unit=f'SA{a+1}',
                                   start=start+timing['start'][a],
                                   end=start+timing['end'][a]))
        free('digits')
        sram.allocate(output, (m, n), 'fp32')
        # Reconstruct in vector registers, discard padding, then scale to FP32.
        vector(label+' reconstruct', timing['raw_bytes']+scale_bytes, 4*m*n)
        free('raw')

    # Load/convert K and V once for all G query heads.
    for name in ('K16', 'V16'):
        record('Load '+name, 'DMA', dma.transfer((S,H), 16, 'read', sram, name))
    sram.allocate('K8', (S,H), 8)
    sram.allocate('V8T', (H,S), 8)
    sram.allocate('KV_scales', 2, 'fp32')
    vector('Quantize K/V + transpose V', 8*S*H, 2*S*H+8)  # Two A16 reads.
    free('K16', 'V16')
    kv_setup_cycles = cycles

    for g in range(G):
        for i in range(0, T, Br):
            m = min(Br, T-i)
            tag = f'h{g} q{i}'
            counts['query_tiles'] += 1
            record(tag+' load Q16', 'DMA', dma.transfer((m,H), 16, 'read', sram, 'Q16'))
            sram.allocate('Q8', (m,H), 8)
            sram.allocate('Q_scale', 1, 'fp32')
            sram.allocate('U', (m,H), 'fp32')
            sram.allocate('l', m, 'fp32')
            sram.allocate('m', m, 'fp32')
            vector(tag+' quantize Q + init U,l,m', 4*m*H, 5*m*H+8*m+4)
            free('Q16')

            key_end = min(S, q_offset+i+m) if causal else S
            for j in range(0, key_end, Bc):
                # Full valid rectangle is computed; future positions are masked.
                c = min(Bc, S-j)
                pair = tag+f' k{j}'
                counts['pairs'] += 1
                matmul(pair+' QK', m, c, H, 'scores', 8)
                sram.allocate('P8', (m,c), 8)
                for name in ('alpha', 'm_new', 'l_new'):
                    sram.allocate(name, m, 'fp32')
                # Two FP32 score reads: rowmax, then exp/sum/P8 encoding.
                # m_new=max(m,rowmax(S)); alpha=exp(m-m_new).
                # l_new=alpha*l+sum(exp(S-m_new)); masked P entries are zero.
                vector(pair+' softmax + P8', 8*m*c+8*m, m*c+12*m)
                free('scores')
                matmul(pair+' PV', m, H, c, 'C', 4)
                free('P8')
                # U <- alpha*U + C; l <- l_new; m <- m_new.
                vector(pair+' update U,l,m', 8*m*H+12*m, 4*m*H+8*m)
                free('C', 'alpha', 'm_new', 'l_new')

            sram.allocate('O16', (m,H), 16)
            vector(tag+' normalize + encode O16', 4*m*H+4*m, 2*m*H)
            free('Q8', 'Q_scale', 'U', 'l', 'm')
            record(tag+' store O16', 'DMA', dma.transfer((m,H), 16, 'write',
                                                        sram, 'O16', release=True))
            if first_tile_end is None:
                first_tile_end = cycles

    free('K8', 'V8T', 'KV_scales')
    result = dict(cycles=cycles, time_us=hw.time_us(cycles),
                  time_ms=hw.time_us(cycles)/1000, kv_setup_cycles=kv_setup_cycles,
                  query_work_cycles=cycles-kv_setup_cycles, counts=counts,
                  breakdown=breakdown, steps=steps, events=events, sram=sram,
                  peak_sram_bytes=sram.peak_bytes,
                  max_sram_utilization_percent=100*sram.max_utilization(),
                  first_tile_end=first_tile_end, hw=hw,
                  config=dict(T=T,S=S,H=H,G=G,Br=Br,Bc=Bc,Br_sa1=Br_sa1,R=R))
    if verbose:
        print(f'\nTotal: {result["time_ms"]:.6f} ms ({cycles:,} cycles)')
        print(f'Peak SRAM: {sram.peak_bytes:,} bytes '
              f'({100*sram.max_utilization():.3f}%)')
    if gantt_path is not None:
        plot_flash_attention(result, gantt_path)
    return result


def plot_flash_attention(result, path):
    """Save a Gantt PNG; matplotlib is only needed when plotting.

    Array bars are stream envelopes including command gaps and draining,
    not claims of continuous arithmetic. Zoom shows cold setup + first Q tile.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import re

    hw, cfg = result['hw'], result['config']
    colors = {'DMA':'#4979b6', 'VCPU':'#dd9338', 'SA1':'#329b80', 'SA2':'#8965ba'}
    lanes = list(colors)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), layout='constrained')
    for ax, end, title in zip(axes, [result['cycles'], result['first_tile_end']],
                              ['Complete KV-head group', 'Cold setup + first query tile']):
        shown = [e for e in result['events'] if e['start'] < end]
        for e in shown:
            x = hw.time_us(e['start'])
            width = hw.time_us(min(e['end'], end)-e['start'])
            y = lanes.index(e['unit'])
            ax.broken_barh([(x, width)], (y-0.3, 0.6), facecolors=colors[e['unit']],
                           edgecolors='white', linewidth=0.35)
            if len(shown) <= 24 and width > hw.time_us(end)*0.035:
                label = re.sub(r'^h\d+ q\d+(?: k\d+)? ', '', e['name'])
                label = label.replace('Quantize K/V + transpose V', 'K/V conversion')
                ax.text(x+width/2, y, f'{label}\n{e["end"]-e["start"]:,} cyc',
                        ha='center', va='center', fontsize=8, color='white')
        ax.set_yticks(range(4), lanes)
        ax.invert_yaxis()
        ax.set_xlim(0, hw.time_us(end))
        ax.set_xlabel('Time (microseconds)')
        ax.set_title(title, loc='left')
        ax.grid(axis='x', alpha=0.2)
    fig.suptitle(f'FlashAttention | {hw.array_model} | {hw.control} control\n'
                 f'T={cfg["T"]}, S={cfg["S"]}, H={cfg["H"]}, G={cfg["G"]}, '
                 f'Br={cfg["Br"]}, Bc={cfg["Bc"]}, R={cfg["R"]} | '
                 f'{result["time_ms"]:.3f} ms | peak SRAM '
                 f'{result["max_sram_utilization_percent"]:.2f}%', fontsize=13)
    fig.legend(handles=[Patch(color=c, label=u) for u,c in colors.items()],
               loc='outside lower center', ncol=4)
    fig.savefig(path, dpi=160)
    plt.close(fig)
