from fa import flash_attention, flash_decode
from helper import *
from mlp import mlp, mlp_decode
from tile_model import Host, Hardware, divup
from math import inf


def _shard(r, reps, n, hw):
    """reps serial kernel runs per accelerator, n accelerators at once on ONE host."""
    interval = hw.issue_interval if hw.control == 'pipelined' else hw.array_latency
    jobs = reps * (r['counts']['SA1_jobs'] + r['counts']['SA2_jobs'])
    return max((reps * r['cycles'], 'accelerator kernel'),                # own SA1/SA2/VCPU
               (n * jobs * interval, 'host command interface'),           # shared command interface
               (n * reps * r['breakdown']['DMA'], 'host DMA'))            # shared host DMA


def _all_reduce(system, rows, tiles=1):
    """Ring all-reduce of a (rows, D) FP16 tile, repeated for tiles: 2(n-1) steps,
    each step n neighbour transfers issued one by one."""
    n = len(system)
    return 2 * (n-1) * n * tiles * system.transfer(0, 1, (rows, D // n), 16) if n > 1 else 0


def _prefill_layer(n, Mt, hw, link_bw, sharing):
    """One prefill layer on a tensor-parallel group of n accelerators.

    sharing = accelerators running at the same time on the ONE host; all of them
    compete for its command interface and DMA (see _shard).
    """
    system = Host(n=n, hw=hw, link_latency=hw.array_latency + hw.dma_setup, link_bw=link_bw)
    att = flash_attention(T=T, S=S, H=H, G=G, Br=192, Bc=256, Br_sa1=None, R=256, hw=hw, verbose=False)
    ffn = mlp(M=T, D=D, F=F // n, Mt=Mt, Ft=256, Mt_sa1=None, R=256, hw=hw, verbose=False)
    ring = _all_reduce(system, Mt, divup(T, Mt))
    (a, a_by), (m, m_by) = _shard(att, K // n, sharing, hw), _shard(ffn, 1, sharing, hw)
    return dict(att=a, mlp=m, allreduce=2*ring, cycles=a + m + 2*ring, bound=(a_by, m_by))


def tensor_parallel_prefill(n, Mt, hw = Hardware(), link_bw=inf):
    """One layer, one T-token sequence, Megatron tensor parallel over n accelerators.

    Attention: the K KV-head groups (each with its G Q heads) split K/n per accelerator.
    MLP: gate/up columns and down rows split F/n per accelerator.
    Each block ends in a ring all-reduce of its (T, D) FP16 output, streamed in Mt-token tiles.
    All accelerators hang off ONE host: its command interface issues every array job
    (one per array_latency when serialized) and its one DMA serves every SRAM.
    Links have infinite bandwidth, but every transfer is still a host command
    (array_latency) set up by the DMA (dma_setup), one transfer at a time.
    """
    return _prefill_layer(n, Mt, hw, link_bw, sharing=n)


def pipeline_tensor_parallel_prefill(nt, np, Mt, m=B, hw=Hardware(), link_bw=inf, one_host=True):
    """Prefill of m T-token sequences through all L layers, pipeline x tensor parallel.

    nt accelerators form one tensor-parallel group; np such groups are the pipeline
    stages, so nt * np accelerators in total. Stage s runs its share of the L layers
    (L // np, one more for the first L % np stages), each layer timed as in
    tensor_parallel_prefill over its nt accelerators.

    Every sequence is one micro-batch (GPipe, forward only): sequence i starts on
    stage s once stage s-1 has handed it over and stage s has finished sequence i-1.
    The hand-off sends the (T, D) FP16 activations from each of the nt ranks to the
    matching rank of the next stage, in Mt-token tiles; like the all-reduce, every
    transfer is one host command (array_latency + dma_setup).

    one_host=True: all accelerators hang off ONE host, so the stages running at once
    (up to min(np, m)) share its command interface and DMA. This is the steady state,
    so warm-up and drain are timed conservatively. one_host=False: a host per stage.

    Returns cycles: ttft (first sequence done), cycles (whole batch), bubble (idle
    share of stage time), speedup over one accelerator running the m sequences
    one after another, and the per-layer bound.
    """
    if not (isinstance(nt, int) and isinstance(np, int) and nt >= 1 and np >= 1):
        raise ValueError('nt and np must be positive integers')
    if K % nt:
        raise ValueError(f'nt must divide the {K} KV heads')
    if np > L:
        raise ValueError(f'np cannot exceed the {L} layers')

    sharing = nt * min(np, m) if one_host else nt
    layer = _prefill_layer(nt, Mt, hw, link_bw, sharing)
    layers = [L // np + (s < L % np) for s in range(np)]
    stage = [n * layer['cycles'] for n in layers]
    link = Host(n=2, hw=hw, link_latency=hw.array_latency + hw.dma_setup, link_bw=link_bw)
    send = nt * divup(T, Mt) * link.transfer(0, 1, (Mt, D), 16) if np > 1 else 0

    done = [0] * np                  # cycle at which stage s finished its latest sequence
    for i in range(m):
        ready = 0                    # every sequence is in DRAM from cycle 0
        for s in range(np):
            done[s] = max(ready, done[s]) + stage[s]
            ready = done[s] + send
        if i == 0:
            ttft = done[-1]

    cycles = done[-1]
    single = m * L * _prefill_layer(1, Mt, hw, link_bw, sharing=1)['cycles']
    return dict(stages=layers, layer=layer, stage=stage, send=send, ttft=ttft,
                cycles=cycles, bubble=1 - m * sum(stage) / (np * cycles),
                speedup=single / cycles, bound=layer['bound'])


def tensor_parallel_decode(n, hw = Hardware(), link_bw=inf):
    """One layer, one decode step of B sequences (one token each) at S context, over n accelerators.

    Attention: every sequence's K KV-head groups split K/n per accelerator (B*K/n flash_decode runs).
    MLP: one mlp_decode of the B-token batch with F/n hidden features per accelerator.
    Each block ends in a ring all-reduce of its (B, D) FP16 output. Host and links as in prefill.
    """
    system = Host(n=n, hw=hw, link_latency=hw.array_latency + hw.dma_setup, link_bw=link_bw)
    att = flash_decode(S=S, H=H, G=G, Bc=1024, R=256, hw=hw, verbose=False)
    ffn = mlp_decode(M=B, D=D, F=F // n, Ft=256, R=256, hw=hw, verbose=False)
    ring = _all_reduce(system, B)
    (a, a_by), (m, m_by) = _shard(att, B * K // n, n, hw), _shard(ffn, 1, n, hw)
    return dict(att=a, mlp=m, allreduce=2*ring, cycles=a + m + 2*ring, bound=(a_by, m_by))
