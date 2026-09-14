#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf, prod
from operator import index


def divup(a: int, b: int) -> int:
    return (a + b - 1) // b


DTYPE_BYTES = {
    'int8': 1, 'uint8': 1,
    'int16': 2, 'uint16': 2, 'float16': 2, 'bfloat16': 2,
    'int32': 4, 'uint32': 4, 'float32': 4
}


def dtype_name(dtype: str | int) -> str:
    """Integer dtype arguments are bit widths: 8 -> int8, 16 -> int16."""

    if isinstance(dtype, int):
        dtype = f'int{dtype}'
    dtype = str(dtype).lower()
    dtype = {'fp16': 'float16', 'bf16': 'bfloat16',
             'fp32': 'float32'}.get(dtype, dtype)
    if dtype not in DTYPE_BYTES:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return dtype


def data_shape(data):
    """Accept shape tuples/lists, """
    if hasattr(data, 'shape'):
        data = data.shape
    if isinstance(data, int):
        data = (data,)
    try:
        shape = tuple(index(d) for d in data)
    except TypeError as exc:
        raise ValueError('Give integer dimensions or an object with .shape') from exc
    if any(d < 0 for d in shape):
        raise ValueError('Dimensions must be nonnegative')
    return shape


def data_bytes(data, dtype) -> int:
    """Bytes = product(shape) x bytes per element; no tensor data is read."""
    return prod(data_shape(data)) * DTYPE_BYTES[dtype_name(dtype)]


@dataclass(frozen=True)
class Hardware:
    sram_bytes: int = 16 * 1024**2
    clock_hz: int = 1_000_000_000
    dma_bw: int = 64
    vector_bw: int = 128
    dma_setup: int = 200
    dram_read_latency: int = 200
    dram_write_latency: int = 150
    vector_launch: int = 300
    array_latency: int = 100
    control: str = 'serialized' # serialized/pipelined
    issue_interval: int = 1
    array_model: str = 'mac' # mac / bandwidth
    arithmetic: str = 'native-int8' #safe-int8 / native-int8

    @property
    def digits(self) -> int:
        return 2 if self.arithmetic == 'safe-int8' else 1

    @property
    def max_reduction(self) -> int:
        return 128 if self.arithmetic == 'safe-int8' else 256

    def time_us(self, cycles: int) -> float:
        """Microseconds = cycles / clock_hz x 1e6."""
        return cycles / self.clock_hz * 1e6


class SRAM:
    """Track allocated buffers, not arithmetic or transfer time.
    """
    def __init__(self, capacity_bytes: int | None = None,
                 hw: Hardware = Hardware()):
        self.capacity_bytes = hw.sram_bytes if capacity_bytes is None else capacity_bytes
        if self.capacity_bytes <= 0:
            raise ValueError('SRAM capacity must be positive')
        self.buffers = {}
        self.peak_bytes = 0
        self.history = []

    @property
    def used_bytes(self) -> int:
        return sum(buf['bytes'] for buf in self.buffers.values())

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def utilization(self) -> float:
        """Fraction occupied, from 0 to 1."""
        return self.used_bytes / self.capacity_bytes

    def allocate(self,
                 name: str,
                 data, dtype) -> int:
        """Reserve a new named buffer. Return allocated bytes, not cycles."""
        if name in self.buffers:
            raise ValueError(f'Buffer already exists: {name}')
        shape, dtype = data_shape(data), dtype_name(dtype)
        nbytes = data_bytes(shape, dtype)
        if nbytes > self.free_bytes:
            raise MemoryError(f'{name} needs {nbytes} bytes; {self.free_bytes} bytes free')
        self.buffers[name] = dict(shape=shape, dtype=dtype, bytes=nbytes)
        self.peak_bytes = max(self.peak_bytes, self.used_bytes)
        self.history.append(dict(action='allocate', name=name, used_bytes=self.used_bytes))
        return nbytes

    def free(self,
             name: str) -> None:

        """Release one buffer. Missing names raise KeyError."""
        del self.buffers[name]
        self.history.append(dict(action='free', name=name, used_bytes=self.used_bytes))


    def status(self) -> dict:
        return dict(used_bytes=self.used_bytes, free_bytes=self.free_bytes,
                    peak_bytes=self.peak_bytes,
                    utilization_percent=100*self.utilization())

    def max_utilization(self) -> float:
        """Peak SRAM usage as a fraction of total capacity."""
        return self.peak_bytes / self.capacity_bytes


class SystolicArray:
    """One physical job; partial tiles are padded for traffic and timing."""
    physical_rows: int
    BW_in: int
    BW_out = 8

    def __init__(self, rows = None,
                 K: int = 128,
                 cols: int = 16,
                 compute_model: str = 'mac'):
        rows = self.physical_rows if rows is None else rows
        if not (1 <= rows <= self.physical_rows and 1 <= K <= 256
                and 1 <= cols <= 16):
            raise ValueError('invalid tile')
        if compute_model not in ('bandwidth', 'mac', 'wavefront'):
            raise ValueError('Unknown array model')
        self.rows, self.K, self.cols = rows, K, cols
        self.compute_model = compute_model

    def input_bytes(self) -> int:
        return (self.physical_rows + 16) * self.K

    def output_bytes(self) -> int:
        return self.physical_rows * 16 * 2

    def input_cycles(self) -> int:
        return divup(self.input_bytes(), self.BW_in)

    def output_cycles(self) -> int:
        return divup(self.output_bytes(), self.BW_out)

    def compute_cycles(self) -> int:
        if self.compute_model == 'bandwidth':
            return 0
        if self.compute_model == 'mac':
            return self.K
        return self.K + self.physical_rows + 16 - 2

    def stage_cycles(self) -> int:
        return max(self.input_cycles(), self.compute_cycles())

    def bandwidth_cycle(self) -> int:
        return max(self.input_cycles(), self.output_cycles())

    def cycle(self) -> int:
        return max(self.stage_cycles(), self.output_cycles())

    def latency(self) -> int:
        return self.stage_cycles() + self.output_cycles()

    def macs(self) -> int:
        return self.rows * self.cols * self.K

    def mac_rate(self) -> float:
        return self.macs() / self.cycle()

    def compute(self,
                rows: int,
                K: int,
                dtype = 8,
                cols: int = 16,
                hw: Hardware = Hardware(),
                sram = None,
                output = None) -> int:

        if dtype_name(dtype) not in ('int8', 'uint8'):
            raise ValueError('Array inputs must be 8-bit integers in this model')
        job = type(self)(rows, K, cols, self.compute_model)
        if (sram is None) != (output is None):
            raise ValueError('Provide both sram and output, or neither')
        if sram is not None:
            sram.allocate(output, (self.physical_rows, 16), 'int16')
        return hw.array_latency + job.latency()


class SA1(SystolicArray):
    physical_rows = 16
    BW_in = 64


class SA2(SystolicArray):
    physical_rows = 32
    BW_in = 96


class VCPU:
    def __init__(self,
                 in_bytes: int = 0,
                 out_bytes: int = 0,
                 hw: Hardware = Hardware()):
        self.in_bytes, self.out_bytes, self.hw = in_bytes, out_bytes, hw

    def cycle(self) -> int:
        return divup(self.in_bytes + self.out_bytes, self.hw.vector_bw)

    def launch_cycle(self) -> int:
        return self.hw.vector_launch + self.cycle()

    def compute(self,
                data,
                in_dtype,
                out_dtype,
                read_passes: int = 1,
                write_passes: int = 1) -> int:
        """
        Same-shape operation: launch + ceil((reads + writes)/vector_bw).

        Example: VCPU().compute((16,128), 16, 8, read_passes=2).
        """
        if (not isinstance(read_passes, int) or not isinstance(write_passes, int)
                or read_passes < 0 or write_passes < 0):
            raise ValueError('Pass counts must be nonnegative integers')
        reads = data_bytes(data, in_dtype) * read_passes
        writes = data_bytes(data, out_dtype) * write_passes
        return VCPU(reads, writes, self.hw).launch_cycle()

    def transform(self,
                  sram: SRAM,
                  source: str,
                  output: str,
                  out_dtype,
                  read_passes: int = 1,
                  free_input: bool = False) -> int:


        src = sram.buffers[source]
        cycles = self.compute(src['shape'], src['dtype'], out_dtype, read_passes)
        sram.allocate(output, src['shape'], out_dtype)
        if free_input:
            sram.free(source)
        return cycles



class DMA:
    def __init__(self,
                 nbytes: int = 0,
                 direction: str = 'read',
                 hw: Hardware = Hardware()):
        if direction not in ('read', 'write'):
            raise ValueError('DMA direction must be read or write')
        self.nbytes, self.direction, self.hw = nbytes, direction, hw

    def cycle(self) -> int:
        latency = (self.hw.dram_read_latency if self.direction == 'read'
                   else self.hw.dram_write_latency)
        return self.hw.dma_setup + latency + divup(self.nbytes, self.hw.dma_bw)

    def transfer(self, data, dtype: str | int, direction = None,
                 sram= None, name = None,
                 release: bool = False) -> int:
        """Cycles = setup + DRAM first-byte latency + ceil(bytes/dma_bw).
        """
        direction = self.direction if direction is None else direction
        shape, dtype = data_shape(data), dtype_name(dtype)
        cycles = DMA(data_bytes(shape, dtype), direction, self.hw).cycle()
        if (sram is None) != (name is None):
            raise ValueError('Provide both sram and name, or neither')
        if release and (direction != 'write' or sram is None):
            raise ValueError('release requires a tracked write')
        if sram is not None:
            if direction == 'read':
                sram.allocate(name, shape, dtype)
            else:
                src = sram.buffers[name]
                if src['shape'] != shape or src['dtype'] != dtype:
                    raise ValueError('Transfer shape/dtype must match the SRAM buffer')
                if release:
                    sram.free(name)
        return cycles


class Accelerator:


    def __init__(self,
                 hw: Hardware = Hardware(),
                 name: str = 'acc0'):
        self.hw, self.name = hw, name
        self.sa1 = SA1(compute_model=hw.array_model)
        self.sa2 = SA2(compute_model=hw.array_model)
        self.sram = SRAM(hw=hw)
        self.vcpu = VCPU(hw=hw)
        self.links: dict[str, Link] = {}

    @property
    def arrays(self) -> tuple[SA1, SA2]:
        return self.sa1, self.sa2

    def link(self, peer: Accelerator) -> Link:
        return self.links[peer.name]

    def __repr__(self) -> str:
        return f'Accelerator({self.name!r}, peers={sorted(self.links)})'


class Link:
    """
    Cycles = latency + ceil(bytes / bw)
    """
    def __init__(self,
                 a: Accelerator,
                 b: Accelerator,
                 bw = inf,
                 latency: int = 0):
        if a is b:
            raise ValueError('A link needs two different accelerators')
        if not bw > 0 or not isinstance(latency, int) or latency < 0:
            raise ValueError('Link bw must be positive and latency a nonnegative integer')
        self.a, self.b, self.bw, self.latency = a, b, bw, latency

    def peer(self, src: Accelerator) -> Accelerator:
        if src is self.a:
            return self.b
        if src is self.b:
            return self.a
        raise ValueError(f'{src.name} is not an end of this link')

    def cycle(self, nbytes: int) -> int:
        return self.latency + (0 if self.bw == inf else ceil(nbytes / self.bw))

    def transfer(self, data, dtype: str | int, src: Accelerator,
                 name: str | None = None, release: bool = False) -> int:
        """Copy from src's SRAM to the peer's SRAM.

        With name, the source buffer must match shape/dtype
        """
        dst = self.peer(src)
        shape, dtype = data_shape(data), dtype_name(dtype)
        if release and name is None:
            raise ValueError('release requires a tracked buffer name')
        if name is not None:
            buf = src.sram.buffers[name]
            if buf['shape'] != shape or buf['dtype'] != dtype:
                raise ValueError('Transfer shape/dtype must match the SRAM buffer')
            dst.sram.allocate(name, shape, dtype)
            if release:
                src.sram.free(name)
        return self.cycle(data_bytes(shape, dtype))


class Host:
    """One host CPU commanding n accelerators
    """
    def __init__(self,
                 n: int = 2,
                 hw: Hardware = Hardware(),
                 link_bw: float = inf,
                 link_latency: int = 0):

        if not isinstance(n, int) or n < 1:
            raise ValueError('Need at least one accelerator')
        self.hw = hw
        self.dma = DMA(hw=hw)
        self.accelerators = [Accelerator(hw, f'acc{i}') for i in range(n)]
        self.links: dict[tuple[int, int], Link] = {}
        for i, a in enumerate(self.accelerators):
            for j in range(i+1, n):
                b = self.accelerators[j]
                link = Link(a, b, link_bw, link_latency)
                self.links[(i, j)] = a.links[b.name] = b.links[a.name] = link

    def __getitem__(self, i: int) -> Accelerator:
        return self.accelerators[i]

    def __len__(self) -> int:
        return len(self.accelerators)

    def link(self,
             i: int,
             j: int) -> Link:
        return self.links[(min(i, j), max(i, j))]

    def dram(self,
             i: int,
             data,
             dtype,
             direction: str = 'read',
             name= None,
             release: bool = False) -> int:

        """DMA between host DRAM and accelerator i's SRAM; same cycles as DMA.transfer."""
        sram = self[i].sram if name is not None else None
        return self.dma.transfer(data, dtype, direction, sram, name, release)


    def transfer(self, i: int, j: int, data, dtype: str | int,
                 name: str | None = None, release: bool = False) -> int:
        """Move data from accelerator i to j over their link; return cycles."""
        return self.link(i, j).transfer(data, dtype, self[i], name, release)


    def command(self, i: int, array: str, rows: int, K: int,
                dtype: str | int = 8, cols: int = 16,
                output: str | None = None) -> int:
        """Issue one array job ('sa1' or 'sa2') on accelerator i.

        Same cycles as SystolicArray.compute; output reserves the INT16
        result buffer in that accelerator's SRAM.
        """
        if array not in ('sa1', 'sa2'):
            raise ValueError("array must be 'sa1' or 'sa2'")
        acc = self[i]
        sram = acc.sram if output is not None else None
        return getattr(acc, array).compute(rows, K, dtype, cols, self.hw, sram, output)


def sram_allocation(N: int,
                    d: int,
                    br: int,
                    bc: int,
                    hw: Hardware) -> dict[str, int]:

    m, c = min(br, N), min(bc, N)
    digits = hw.digits
    chunks_qk = divup(d, hw.max_reduction)
    chunks_pv = divup(c, hw.max_reduction)
    return {
        'K16 + V16': 4 * N * d,
        'K8 + packed V8 transpose': 2 * N * d,
        'Q16 + Q8': 3 * m * d,
        'FP32 output numerator U': 4 * m * d,
        'FP32 PV contribution': 4 * m * d,
        'INT16 final output': 2 * m * d,
        'FP32 scores + FP32 P + INT8 P': 9 * m * c,
        'row statistics and scales': 32 * m + 64,
        'reusable operand digit planes': (
            digits * max((m + c) * d, (m + d) * c) if digits == 2 else 0),
        'reusable INT16 raw partials': (
            2 * digits**2 * max(m * c * chunks_qk, m * d * chunks_pv)),
    }