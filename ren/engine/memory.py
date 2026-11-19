from collections import defaultdict

from ren.dtype import dtypes
from ren.buffer import Buffer, BufferSpec
from ren.engine.realize import buffer_binding
from ren.ir import Op
from ren.ir.node import Node
from ren.engine.tlsf import TLSFAllocator, round_up
from ren.context import DEBUG


# Ren's memory planner adapts code from tinygrad.
def plan_memory(linear: Node, noopt_buffers: set[Buffer] | None = None) -> dict[Buffer, Buffer]:
    """Pack temporary storage into TLSF arenas using CALL binding lifetimes.

    Parameters, allocated storage, protected buffers, and COPY buffers are
    excluded so external storage stays fixed and transfers remain parallel.
    """
    assert linear.op is Op.LINEAR
    protected = set(noopt_buffers or ())
    first_use: dict[Buffer, int] = {}
    last_use: dict[Buffer, int] = {}
    buf_to_opt: set[Buffer] = set()
    storage: dict[Node, Buffer | None] = {}

    def binding_storage(node: Node) -> Buffer | None:
        if node not in storage:
            base = buffer_binding(node)
            while base.op is Op.BUFFER_VIEW:
                base = base.src[0]
            assert base.op in {Op.BUFFER, Op.PARAM}
            storage[node] = base.buffer if base.op is Op.BUFFER else None
        return storage[node]

    # Compute first and last use for each buffer.
    for ki, call in enumerate(linear.src):
        bufs = [buf for source in call.inputs if (buf := binding_storage(source)) is not None]
        if call.src[0].op is Op.COPY:
            protected.update(buf.base for buf in bufs)
        for buf in bufs:
            base = buf.base
            # Allocated and explicitly protected storage stays fixed.
            if base.is_allocated() or base in protected:
                continue
            if base not in first_use:
                first_use[base] = ki
            last_use[base] = ki
            buf_to_opt.add(buf)

    # A later COPY or retained view can protect storage encountered earlier.
    first_use = {buf: ki for buf, ki in first_use.items() if buf not in protected}
    if not first_use:
        return {}

    # Group by device only (single dtype-agnostic arena per device)
    by_device: dict[str, list[Buffer]] = defaultdict(list)
    for buf in first_use:
        by_device[buf.spec.device].append(buf)

    assigned: dict[Buffer, Buffer] = {}

    for device, bufs in by_device.items():
        min_block = 0x1000 if device != "python" else 1

        # Calculate TLSF virtual space (2x for fragmentation headroom)
        required_bytes = sum(round_up(b.nbytes, min_block) for b in bufs)
        virtual_bytes = round_up(required_bytes * 2, min_block)

        # Create fresh allocator over virtual space (no device allocation yet)
        allocator = TLSFAllocator(virtual_bytes, base=0, block_size=min_block, lv2_cnt=32)

        # Create timeline events sorted by (time, is_alloc) so frees precede allocs at the same step
        events: list[tuple[int, bool, Buffer]] = []
        for buf in bufs:
            start = first_use[buf]
            end = last_use[buf] + 1
            events.append((start, True, buf))
            events.append((end, False, buf))
        events.sort(key=lambda e: (e[0], e[1]))

        # Simulate allocation to find peak concurrent usage
        peak = 0
        offsets: dict[Buffer, int] = {}
        for _, is_alloc, buf in events:
            if is_alloc:
                req = round_up(buf.nbytes, min_block)
                off = allocator.alloc(req)
                offsets[buf] = off
                peak = max(peak, off + req)
            else:
                allocator.free(offsets[buf])

        # Allocate arena at peak usage (not full virtual space)
        arena_size = round_up(peak, min_block) if peak > 0 else min_block
        base_spec = BufferSpec(dtype=dtypes.uint8, size=arena_size, device=device)
        arena_base = Buffer(base_spec)

        if DEBUG >= 2:
            print(f"memory arena device={device} bytes={arena_size}")

        # Create view buffers pointing into arena
        for buf in bufs:
            off_bytes = offsets[buf]
            # Create a view into the arena (offset is in bytes)
            view = Buffer(buf.spec, base=arena_base, offset=off_bytes)
            assigned[buf] = view

    # Handle sub-buffers: if the base was remapped, update the view
    for buf in buf_to_opt:
        if buf._base is not None and buf not in assigned:
            base = buf.base
            if base in assigned:
                new_base = assigned[base]
                assigned[buf] = Buffer(buf.spec, base=new_base.base, offset=new_base._offset + buf._offset)

    if DEBUG >= 2:
        print(f"memory plan: optimized {len(assigned)} buffers")
    return assigned
