#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from math import prod
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


def data_shape(data) -> tuple[int, ...]:
    """Accept shape tuples/lists, an element count, or an object with .shape."""
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


def data_bytes(data, dtype: str | int) -> int:
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

    SRAM has different bandwidths to each engine; DMA/VCPU/SA charge those
    transfers. Do not add another generic SRAM transfer time.
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

    def allocate(self, name: str, data, dtype: str | int) -> int:
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

    def free(self, name: str) -> None:
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

    def __init__(self, rows: int | None = None, K: int = 128, cols: int = 16,
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

    def compute(self, rows: int, K: int, dtype: str | int = 8, cols: int = 16,
                hw: Hardware = Hardware(), sram: SRAM | None = None,
                output: str | None = None) -> int:
        """Cycles = array_latency + max(input_cycles, compute_cycles) + output_cycles.

        Example: SA1().compute(16, 128, 8).
        Only 8-bit integer input is modeled; the physical output is INT16.
        Optional sram/output reserves a NEW full physical output buffer.
        Caller must allocate padded input buffers separately. Neither input
        conversion nor four digit-pair products are automatically included.
        This estimates the requested job without changing the instance's
        original cycle()/latency() parameters. compute_model comes from self.
        """
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
    def __init__(self, in_bytes: int = 0, out_bytes: int = 0,
                 hw: Hardware = Hardware()):
        self.in_bytes, self.out_bytes, self.hw = in_bytes, out_bytes, hw

    def cycle(self) -> int:
        return divup(self.in_bytes + self.out_bytes, self.hw.vector_bw)

    def launch_cycle(self) -> int:
        return self.hw.vector_launch + self.cycle()

    def compute(self, data, in_dtype: str | int, out_dtype: str | int,
                read_passes: int = 1, write_passes: int = 1) -> int:
        """Same-shape operation: launch + ceil((reads + writes)/vector_bw).

        Example: VCPU().compute((16,128), 16, 8, read_passes=2).
        Count separate buffers for scales/intermediates when needed; this
        method counts only the supplied input/output payload and passes.
        """
        if (not isinstance(read_passes, int) or not isinstance(write_passes, int)
                or read_passes < 0 or write_passes < 0):
            raise ValueError('Pass counts must be nonnegative integers')
        reads = data_bytes(data, in_dtype) * read_passes
        writes = data_bytes(data, out_dtype) * write_passes
        return VCPU(reads, writes, self.hw).launch_cycle()

    def transform(self, sram: SRAM, source: str, output: str,
                  out_dtype: str | int, read_passes: int = 1,
                  free_input: bool = False) -> int:
        """Model conversion/copy and track two simultaneously live buffers.

        Destination must have a new name. Out-of-place conversion: allocate
        output first, then optionally release input after the operation.
        This is metadata bookkeeping, not numerical quantization.
        """
        src = sram.buffers[source]
        cycles = self.compute(src['shape'], src['dtype'], out_dtype, read_passes)
        sram.allocate(output, src['shape'], out_dtype)
        if free_input:
            sram.free(source)
        return cycles



class DMA:
    def __init__(self, nbytes: int = 0, direction: str = 'read',
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


def sram_allocation(N: int, d: int, br: int, bc: int,
                    hw: Hardware) -> dict[str, int]:
    """Original static estimate. Independent of the live SRAM tracker."""
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