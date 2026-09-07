#!/usr/bin/env python3
"""Pure-Python numerical reference, not accelerator executable code.
Validates causal/noncausal GQA, online softmax and split-KV state merging.
Uses Python float to isolate algorithmic correctness from quantization error.
"""
import math, random

def empty_state(d):
    return (-math.inf, 0.0, [0.0]*d)

def update(state, scores, values):
    m, ell, out = state
    valid = [(s,v) for s,v in zip(scores,values) if s != -math.inf]
    if not valid:
        return state
    new_m = max(m, max(s for s,_ in valid))
    alpha = 0.0 if ell == 0 else math.exp(m-new_m)
    p = [math.exp(s-new_m) for s,_ in valid]
    return (new_m, alpha*ell+sum(p),
            [alpha*out[d]+sum(w*v[d] for w,(_,v) in zip(p,valid))
             for d in range(len(out))])

def merge(a,b):
    ma,la,ua=a; mb,lb,ub=b
    if la==0:return b
    if lb==0:return a
    m=max(ma,mb); aa=math.exp(ma-m); ab=math.exp(mb-m)
    return m, aa*la+ab*lb, [aa*x+ab*y for x,y in zip(ua,ub)]

def finish(state):
    _,ell,out=state
    return [x/ell for x in out] if ell else [0.0]*len(out)

def flash_gqa(q,k,v,block=4,causal=True,query_positions=None):
    # q [B,Hq,Nq,D], k/v [B,Hkv,Nkv,D].
    bsz,hq,nq,d=len(q),len(q[0]),len(q[0][0]),len(q[0][0][0])
    hkv,nkv=len(k[0]),len(k[0][0]); assert hq%hkv==0
    group=hq//hkv
    positions=list(range(nq)) if query_positions is None else query_positions
    result=[]
    for b in range(bsz):
        heads=[]
        for h in range(hq):
            rows=[]; g=h//group
            for i in range(nq):
                state=empty_state(d)
                for j in range(0,nkv,block):
                    values=v[b][g][j:j+block]
                    scores=[sum(x*y for x,y in zip(q[b][h][i],k[b][g][t]))/math.sqrt(d)
                            if not causal or t<=positions[i] else -math.inf
                            for t in range(j,min(j+block,nkv))]
                    state=update(state,scores,values)
                rows.append(finish(state))
            heads.append(rows)
        result.append(heads)
    return result

def dense_gqa(q,k,v,causal=True,query_positions=None):
    # Independently materialize each complete score row for reference.
    hq,hkv=len(q[0]),len(k[0]); d=len(q[0][0][0]); result=[]
    for b in range(len(q)):
        heads=[]
        for h in range(hq):
            g=h//(hq//hkv); rows=[]
            for i,row in enumerate(q[b][h]):
                pos=i if query_positions is None else query_positions[i]
                ids=[j for j in range(len(k[b][g])) if not causal or j<=pos]
                if not ids:
                    rows.append([0.0]*d); continue
                scores=[sum(a*c for a,c in zip(row,k[b][g][j]))/math.sqrt(d) for j in ids]
                mx=max(scores); probs=[math.exp(s-mx) for s in scores]; den=sum(probs)
                rows.append([sum(p*v[b][g][j][x] for p,j in zip(probs,ids))/den for x in range(d)])
            heads.append(rows)
        result.append(heads)
    return result

def flattened(x):
    for item in x:
        if isinstance(item,list):yield from flattened(item)
        else:yield item

def main():
    rng=random.Random(22)
    def tensor(b,h,n,d,scale=1):
        return [[[[rng.uniform(-scale,scale) for _ in range(d)] for _ in range(n)]
                 for _ in range(h)] for _ in range(b)]
    cases=0; max_error=0.0
    for scale in [1,50]:
        q=tensor(2,4,9,5,scale); k=tensor(2,2,9,5,scale); v=tensor(2,2,9,5)
        for causal in [True,False]:
            expected=dense_gqa(q,k,v,causal)
            for block in [1,3,4,16]:
                actual=flash_gqa(q,k,v,block,causal)
                err=max(abs(a-b) for a,b in zip(flattened(actual),flattened(expected)))
                assert err<1e-10,err
                max_error=max(max_error,err); cases+=1
    q=tensor(2,4,1,5); k=tensor(2,2,13,5); v=tensor(2,2,13,5)
    for pos in [-1,0,12]:
        actual=flash_gqa(q,k,v,4,True,[pos]); expected=dense_gqa(q,k,v,True,[pos])
        err=max(abs(a-b) for a,b in zip(flattened(actual),flattened(expected)))
        assert err<1e-10; max_error=max(max_error,err); cases+=1
    scores=[1000,-1000,999,1001,-2000]; values=[[rng.random() for _ in range(5)] for _ in scores]
    all_state=update(empty_state(5),scores,values)
    joined=merge(update(empty_state(5),scores[:2],values[:2]),
                 update(empty_state(5),scores[2:],values[2:]))
    assert max(abs(a-b) for a,b in zip(finish(all_state),finish(joined)))<1e-12
    assert merge(empty_state(5),all_state)==all_state
    assert update(empty_state(5),[-math.inf],[[1]*5])==empty_state(5)
    print(f'PASS: {cases} attention comparisons; max abs error={max_error:.3g}; split-KV and empty-state checks passed.')
    print('Not tested: actual 16-bit rounding, W8 quantization quality, device ISA, queue timing, host collectives.')

if __name__=='__main__':main()
