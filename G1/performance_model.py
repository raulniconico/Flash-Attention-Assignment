#!/usr/bin/env python3
"""Conditional analytical model; not a cycle-accurate simulator.
Run: python3 performance_model.py --output results.json
All bandwidths are aggregate bytes/cycle. A MAC is one multiply-add.
Primary backend: literal memory-only vector CPU, 16-bit stored activations,
8-bit weights, FP32 running state, >=12 KiB vector-local working storage.
"""
from dataclasses import dataclass, replace
import argparse, json, math

@dataclass(frozen=True)
class Hardware:
    hz: float = 1e9
    dram_bw: float = 64.0
    dma_sram_bw: float = 64.0
    vector_bw: float = 128.0
    dma_setup: float = 200.0
    read_latency: float = 200.0
    write_latency: float = 150.0
    vector_launch: float = 300.0
    @property
    def path_bw(self):
        return min(self.dram_bw, self.dma_sram_bw)
    def read(self, n):
        return self.dma_setup + self.read_latency + math.ceil(n/self.path_bw)
    def write(self, n):
        return self.dma_setup + self.write_latency + math.ceil(n/self.path_bw)
    def vector(self, n):
        return self.vector_launch + math.ceil(n/self.vector_bw)

L, D, F, HQ, HKV, HD, VOCAB = 32, 4096, 14336, 32, 8, 128, 128256
BATCH, PROMPT, OUTPUT = 16, 2048, 256

def matrices(tp=1):
    assert HQ % tp == HKV % tp == F % tp == 0
    return [('q', D, D//tp), ('k', D, HKV*HD//tp),
            ('v', D, HKV*HD//tp), ('o', D//tp, D),
            ('gate', D, F//tp), ('up', D, F//tp),
            ('down', F//tp, D)]

def dense_cycles(m, k, n, hw, out_bytes=2):
    """256-row DRAM macrotiles, 16x128 register microtiles.
    Keep full A macrotile in SRAM; double-buffer full-K weight panels.
    For each macro, serialize input/output DMA with its weight pipeline.
    Quantization scale: one FP32 number/output column; read once per panel.
    """
    assert n % 128 == 0 and m % 16 == 0
    total = 0.0
    panels = n//128
    for i in range(0, m, 256):
        rows = min(256, m-i)
        # Worst SRAM: A + two weight panels + full output + workspace.
        allocated = rows*k*2 + 2*k*128 + rows*n*out_bytes + 1024*1024
        assert allocated <= 16*1024*1024, (m,k,n,allocated)
        r = hw.read(k*128 + 128*4)
        # W reread per 16 rows; A read per output panel; FP16 output write.
        io = (rows//16)*k*128 + rows*k*2 + rows*128*out_bytes + 128*4
        v = hw.vector(io)
        total += hw.read(rows*k*2)
        total += r + v + (panels-1)*max(r,v)
        total += hw.write(rows*n*out_bytes)
    return total

def elementwise_cycles(m, tp, hw):
    # Conservative unfused traffic budget for 2 RMSNorm, 2 residual adds,
    # RoPE on local Q/K, and local SiLU*gate. 5 launched macro programs.
    traffic = m*(20*D + 4*(D+HKV*HD)/tp + 6*F/tp)
    # Budget an entire read/write DMA pass and vector pass serially.
    # 5 read/write pairs cover the aggregate bytes; no compute-time term.
    dma = traffic/hw.path_bw + 5*(2*hw.dma_setup+hw.read_latency+hw.write_latency)
    vec = traffic/hw.vector_bw + 5*hw.vector_launch
    return dma+vec

def prefill_attention_cycles(hw):
    groups = BATCH*HKV
    query_blocks = PROMPT//16
    visited = sum(math.ceil((i+16)/128) for i in range(0,PROMPT,16))
    # One QK tile, online update, then PV tile. FP32 S/P/O, FP16 Q/K/V.
    qk = 16*128*2 + 128*128*2 + 16*128*4
    online = 2*16*128*4 + 2*16*128*4 + 2*16*2*4
    pv = 16*128*4 + 128*128*2 + 2*16*128*4
    tile_bytes = qk+online+pv
    # Initialize O,m,l; final O read + FP16 write for every query block.
    boundary_bytes = 16*128*4 + 16*2*4 + 16*128*6
    vec = hw.vector(4*(visited*tile_bytes+query_blocks*boundary_bytes))
    r = hw.read(2*PROMPT*HD*2) + hw.read(4*PROMPT*HD*2)
    w = hw.write(4*PROMPT*HD*2)
    # Double-buffer full head-group bundles; conservative boundary allowance.
    layer = groups*max(r+w,vec)+r+vec+w
    return L*layer, {'visited_tiles_per_head':visited,'tile_vector_bytes':tile_bytes,
                     'vector_cycles_per_group':vec,'dma_cycles_per_group':r+w}

def decode_attention_cycles(context, tp, hw):
    groups = BATCH*HKV//tp
    kv = 2*context*HD*2
    qout = 4*HD*2
    # Four Q heads share K/V. Q,m,l,O stay local during the full KV sweep.
    r = hw.read(kv) + hw.read(qout)
    w = hw.write(qout)
    v = hw.vector(kv+2*qout)
    return L*(groups*max(r+w,v)+r+v+w)

def reduction_staging_cycles(hw, payload_bytes=4):
    x = BATCH*D*payload_bytes
    # Per rank, two collectives/layer. Outgoing SRAM->DRAM and reverse.
    return 2*L*(hw.write(x)+hw.read(x))

def head_and_sampling_cycles(hw, tp=1):
    # 128256/4 is not divisible by 128: pad last vocabulary panel.
    n = math.ceil(VOCAB/tp/128)*128
    c = dense_cycles(BATCH,D,n,hw)
    # Read local FP16 logits once for exact local greedy argmax.
    c += hw.read(BATCH*n*2)+hw.vector(BATCH*n*2)
    return c

def decode(context, hw=Hardware(), tp=1, batch=BATCH):
    assert batch == BATCH, 'This TP model holds global batch at 16.'
    dense = L*sum(dense_cycles(BATCH,k,n,hw,4 if tp>1 and name in ('o','down') else 2) for name,k,n in matrices(tp))
    dense += head_and_sampling_cycles(hw,tp)
    attn = decode_attention_cycles(context,tp,hw)
    aux = L*elementwise_cycles(BATCH,tp,hw)
    # New KV token writes and embedding fetch, conservative separate commands.
    cache_write = L*hw.write(BATCH*2*HKV*HD*2/tp)
    embedding = hw.read(BATCH*D*2)
    staging = reduction_staging_cycles(hw) if tp>1 else 0
    cycles = dense+attn+aux+cache_write+embedding+staging
    return {'dense_s':dense/hw.hz,'attention_s':attn/hw.hz,'auxiliary_s':(aux+cache_write+embedding)/hw.hz,
            'collective_staging_s':staging/hw.hz,'step_s':cycles/hw.hz,
            'per_sequence_tokens_s':hw.hz/cycles,'aggregate_tokens_s':BATCH*hw.hz/cycles}

def prefill(hw=Hardware()):
    m = BATCH*PROMPT
    dense = L*sum(dense_cycles(m,k,n,hw) for _,k,n in matrices())
    head = head_and_sampling_cycles(hw)
    attn, details = prefill_attention_cycles(hw)
    aux = L*elementwise_cycles(m,1,hw)
    # QKV projections already write K/V to DRAM; no second cache write here.
    embed = hw.read(m*D*2)
    total = dense+head+attn+aux+embed
    return {'dense_s':dense/hw.hz,'head_and_sampling_s':head/hw.hz,
            'attention_s':attn/hw.hz,'auxiliary_s':(aux+embed)/hw.hz,
            'warm_ttft_s':total/hw.hz,'attention_details':details}

def elasticity(fn, hw, field, delta=0.01):
    base = fn(hw)
    modified = fn(replace(hw,**{field:getattr(hw,field)*(1+delta)}))
    return -math.log(modified/base)/math.log(1+delta)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output'); a=p.parse_args()
    hw=Hardware()
    weight_layer=sum(k*n for _,k,n in matrices())
    weights=L*weight_layer+D*VOCAB
    contexts=list(range(PROMPT+1,PROMPT+OUTPUT)) # 255 decode forwards
    dec=[decode(c,hw) for c in contexts]
    tp=[decode(c,hw,4) for c in contexts]
    avg=sum(x['step_s'] for x in dec)/len(dec)
    avg4=sum(x['step_s'] for x in tp)/len(tp)
    # Host reduction: 4 gather + 4 scatter transfers per collective.
    comm=2*L*8*BATCH*D*4
    # Greedy vocab reduction: FP32 score + uint32 token ID per rank,
    # then broadcast uint32 winners to all ranks: 768 bytes/step.
    comm+=4*BATCH*8+4*BATCH*4
    raw_kv=2*L*BATCH*HKV*HD*2176*2
    floor=(weights+raw_kv)/(4*hw.path_bw*hw.hz)
    report={
      'assumptions':{'backend':'memory-only vector CPU; >=12 KiB local working storage',
                     'sampling':'greedy','resident_model':True,'reduction_payload_bytes':4,
                     'host_collective_latency_s':0,'quant_scale':'FP32 per output column',
                     'array_compute_timing':'not needed by primary backend'},
      'counts':{'layer_linear_weights':weight_layer,'streamed_weight_bytes_per_step':weights,
                'total_weight_bytes_including_input_embedding':weights+D*VOCAB,
                'kv_bytes_at_2048':2*L*BATCH*HKV*HD*2048*2,
                'kv_bytes_at_2304':2*L*BATCH*HKV*HD*2304*2,
                'prefill_dense_macs':BATCH*PROMPT*L*weight_layer,
                'prefill_causal_attention_macs':L*BATCH*HQ*HD*PROMPT*(PROMPT+1),
                'decode_dense_macs':BATCH*weights,'decode_attention_macs_at_2176':2*L*BATCH*HQ*HD*2176},
      'prefill':prefill(hw),
      'decode_first':dec[0],'decode_average_context':decode(2176,hw),'decode_last':dec[-1],
      'generation':{'decode_forwards':len(contexts),'decode_total_s':sum(x['step_s'] for x in dec),
                    'average_step_s':avg,'per_sequence_tokens_s':1/avg,'aggregate_tokens_s':BATCH/avg},
      'four_way':{'average_step_excluding_host_network_s':avg4,'conditional_per_sequence_tokens_s':1/avg4,
                  'conditional_aggregate_tokens_s':BATCH/avg4,
                  'traffic_floor_s':floor,'traffic_ceiling_per_sequence_tokens_s':1/floor,
                  'traffic_ceiling_aggregate_tokens_s':BATCH/floor,
                  'host_aggregate_bytes_per_step':comm,
                  'host_GB_s_70pct_schedule_serial':comm/((1/0.7-1)*avg4)/1e9,
                  'host_GB_s_70pct_traffic_ceiling_serial':comm/(floor/0.7-avg4)/1e9 if floor/0.7>avg4 else None,
                  'host_GB_s_70pct_traffic_ceiling_perfect_overlap':comm/(floor/0.7)/1e9},
      'sensitivity':{},
      'improvements':{'double_memory_path':decode(2176,replace(hw,dram_bw=128,dma_sram_bw=128)),
                      'double_vector_bandwidth':decode(2176,replace(hw,vector_bw=256)),
                      'ideal_dma_descriptor_and_latency_hiding':decode(2176,replace(hw,dma_setup=0,read_latency=0,write_latency=0))}}
    for name,fn in [('prefill_attention',lambda h:prefill_attention_cycles(h)[0]/h.hz),
                    ('decode_attention',lambda h:decode_attention_cycles(2176,1,h)/h.hz),
                    ('decode_total',lambda h:decode(2176,h)['step_s'])]:
        report['sensitivity'][name]={field:elasticity(fn,hw,field) for field in
              ['dram_bw','dma_sram_bw','vector_bw','dma_setup','vector_launch']}
        both=replace(hw,dram_bw=64*1.01,dma_sram_bw=64*1.01)
        report['sensitivity'][name]['both_memory_links']= -math.log(fn(both)/fn(hw))/math.log(1.01)
    out=json.dumps(report,indent=2)
    if a.output:
        with open(a.output,'w') as f:f.write(out+'\n')
    print(out)

if __name__=='__main__':main()
