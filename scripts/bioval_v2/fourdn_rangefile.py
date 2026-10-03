# ruff: noqa: SIM115  (one-shot data-prep script; style-only, behaviour unchanged)
"""Sparse local mirror of a remote file, filled by parallel HTTP range requests (CPU/network only)."""
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import requests


class RangeMirror(io.RawIOBase):
    def __init__(self, url, local, size, block=1 << 20, workers=12):
        self.url, self.local, self.size, self.block = url, local, size, block
        self.pos = 0; self.lock = threading.Lock()
        self.state = local + ".blocks.json"
        self.have = set(json.load(open(self.state))) if os.path.exists(self.state) else set()
        mode = "r+b" if os.path.exists(local) else "w+b"
        self.fd = open(local, mode)
        if mode == "w+b": self.fd.truncate(size)
        self.ex = ThreadPoolExecutor(workers)
        self.sess = threading.local(); self.last = -10; self.streak = 0; self.ahead = 48
    def _get(self, b):
        s = getattr(self.sess, "s", None)
        if s is None: s = self.sess.s = requests.Session()
        a = b * self.block; z = min(self.size, a + self.block) - 1
        for k in range(8):
            try:
                r = s.get(self.url, headers={"Range": f"bytes={a}-{z}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == z - a + 1: return b, r.content
            except requests.RequestException:
                pass
        raise OSError(f"range fetch failed block {b}")
    def ensure(self, blocks):
        need = sorted(set(blocks) - self.have)
        # coalesce neighbouring blocks into larger requests would be faster; keep simple
        for b, data in self.ex.map(self._get, need):
            with self.lock:
                os.pwrite(self.fd.fileno(), data, b * self.block); self.have.add(b)
        if need: self.save()
    def save(self):
        with self.lock: json.dump(sorted(self.have), open(self.state, "w"))
    def prefetch_ranges(self, ranges):
        bl = set()
        for off, n in ranges:
            if n > 0: bl.update(range(off // self.block, (off + n - 1) // self.block + 1))
        self.ensure(bl)
    # file API
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, o, w=0):
        self.pos = o if w == 0 else self.pos + o if w == 1 else self.size + o
        return self.pos
    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0: return 0
        b0 = self.pos // self.block; b1 = (self.pos + n - 1) // self.block
        self.streak = self.streak + 1 if b0 in (self.last, self.last + 1) else 0
        self.last = b1
        hi = b1 + (self.ahead if self.streak >= 2 else 0)
        self.ensure(range(b0, min(hi, (self.size - 1) // self.block) + 1))
        data = os.pread(self.fd.fileno(), n, self.pos); buf[:len(data)] = data; self.pos += len(data)
        return len(data)
