"""Vanilla (fully serial) FlashAttention-2 prefill timing, one KV-head group.

Follows the pseudocode of 3.flash_attention.ipynb line by line: every step
waits for the previous one, only SA1 and SA2 overlap inside a GEMM. Timing
only; no numerical attention is computed.
"""
from tile_model import Hardware, SRAM, DMA, VCPU, SA1, SA2, divup


def flash_attention_vanilla(T=2048, S=2048, H=128, G=4, Br=128, Bc=256, R=128,
                            causal=True, hw=Hardware(), verbose=False):
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


if __name__ == '__main__':
    flash_attention_vanilla(verbose=True)
