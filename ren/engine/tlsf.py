import collections
import functools
from typing import Dict, List


def round_up(x: int, align: int) -> int:
    if align <= 0:
        return x
    return ((x + align - 1) // align) * align


# Adapted from tinygrad's TLSFAllocator.


class TLSFAllocator:
    """
    Minimal TLSF-style allocator over a fixed-size address space.

    The allocator manages a set of free blocks using a two-level
    segregated fit scheme: a coarse first level indexed by the most
    significant bit of the block size, and a finer second level that
    subdivides each first-level range into `lv2_cnt` buckets.

    Blocks are identified by their starting offset (an integer in
    [base, base+size)). `alloc` returns such an offset; `free` releases
    the block and attempts to merge with neighboring free blocks.
    """

    def __init__(self, size: int, base: int = 0, block_size: int = 16, lv2_cnt: int = 16):
        self.size = size
        self.base = base
        self.block_size = block_size
        self.l2_cnt = lv2_cnt.bit_length()

        # storage[level1][level2] -> list of block starts
        self.storage: List[Dict[int, List[int]]] = [collections.defaultdict(list) for _ in range(size.bit_length() + 1)]
        self.lv1_entries: List[int] = [0] * len(self.storage)

        # blocks[start] = (size, next, prev, is_free)
        self.blocks: Dict[int, tuple[int, int | None, int | None, bool]] = {0: (size, None, None, True)}
        if size > 0:
            self._insert_block(0, size)

    def lv1(self, size: int) -> int:
        return size.bit_length()

    def lv2(self, size: int) -> int:
        return self._lv2(size, self.l2_cnt)

    @staticmethod
    @functools.cache
    def _lv2(size: int, l2_cnt: int) -> int:
        return (size - (1 << (size.bit_length() - 1))) // (1 << max(0, size.bit_length() - l2_cnt))

    def _insert_block(self, start: int, size: int, prev: int | None = None) -> TLSFAllocator:
        if prev is None:
            prev = self.blocks[start][2]
        l1 = self.lv1(size)
        l2 = self.lv2(size)
        self.storage[l1][l2].append(start)
        self.lv1_entries[l1] += 1
        self.blocks[start] = (size, start + size, prev, True)
        return self

    def _remove_block(self, start: int, size: int, prev: int | None = None) -> TLSFAllocator:
        if prev is None:
            prev = self.blocks[start][2]
        l1 = self.lv1(size)
        l2 = self.lv2(size)
        self.storage[l1][l2].remove(start)
        self.lv1_entries[l1] -= 1
        self.blocks[start] = (size, start + size, prev, False)
        return self

    def _split_block(self, start: int, size: int, new_size: int) -> TLSFAllocator:
        nxt = self.blocks[start][1]
        assert self.blocks[start][3], "block must be free"
        self._remove_block(start, size)
        self._insert_block(start, new_size)
        self._insert_block(start + new_size, size - new_size, prev=start)
        if nxt in self.blocks:
            nsz, nnxt, _, nfree = self.blocks[nxt]
            self.blocks[nxt] = (nsz, nnxt, start + new_size, nfree)
        return self

    def _merge_right(self, start: int) -> None:
        size, nxt, _, is_free = self.blocks[start]
        assert is_free, "block must be free"

        while is_free and nxt in self.blocks:
            blk = self.blocks[nxt]
            if not blk[3]:
                break
            self._remove_block(start, size)
            self._remove_block(nxt, blk[0])
            self._insert_block(start, size := size + blk[0])
            assert self.blocks[start][1] == blk[1]
            _, nxt, _, _ = self.blocks.pop(nxt)

        if nxt in self.blocks:
            nsz, nnxt, _, nfree = self.blocks[nxt]
            self.blocks[nxt] = (nsz, nnxt, start, nfree)

    def _merge_block(self, start: int) -> None:
        # Go left while blocks are free, then merge right.
        while (x := self.blocks[start][2]) is not None and self.blocks[x][3]:
            start = x
        self._merge_right(start)

    def alloc(self, req_size: int, align: int = 1) -> int:
        req_size = max(self.block_size, req_size)
        size = max(self.block_size, req_size + align - 1)

        # Round up allocation size to next bucket so any entry there fits.
        size = round_up(size, (1 << max(0, size.bit_length() - self.l2_cnt)))

        for l1 in range(self.lv1(size), len(self.storage)):
            if self.lv1_entries[l1] == 0:
                continue
            start_l2 = self.lv2(size) if l1 == size.bit_length() else 0
            for l2 in range(start_l2, (1 << self.l2_cnt)):
                bucket = self.storage[l1].get(l2)
                if not bucket:
                    continue
                start = bucket[0]
                nsize = self.blocks[start][0]
                assert nsize >= size, "block must be large enough"

                # Align start if needed by splitting off a prefix fragment.
                new_start = round_up(start, align)
                if new_start != start:
                    self._split_block(start, nsize, new_start - start)
                    start = new_start
                    nsize = self.blocks[start][0]

                # Split if block is larger than requested.
                if nsize > req_size:
                    self._split_block(start, nsize, req_size)
                self._remove_block(start, req_size)
                return start + self.base

        raise MemoryError(f"TLSFAllocator: cannot allocate {req_size} bytes")

    def free(self, start: int) -> None:
        rel = start - self.base
        size, _, _, _ = self.blocks[rel]
        self._insert_block(rel, size)
        self._merge_block(rel)


__all__ = ["TLSFAllocator", "round_up"]
