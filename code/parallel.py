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
    system = Host(n=n, hw=hw, link_latency=hw.array_latency + hw.dma_setup, link_bw=link_bw)
    att = flash_attention(T=T, S=S, H=H, G=G, Br=192, Bc=256, Br_sa1=None, R=256, hw=hw, verbose=False)
    ffn = mlp(M=T, D=D, F=F // n, Mt=Mt, Ft=256, Mt_sa1=None, R=256, hw=hw, verbose=False)
    ring = _all_reduce(system, Mt, divup(T, Mt))
    (a, a_by), (m, m_by) = _shard(att, K // n, n, hw), _shard(ffn, 1, n, hw)
    return dict(att=a, mlp=m, allreduce=2*ring, cycles=a + m + 2*ring, bound=(a_by, m_by))


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
