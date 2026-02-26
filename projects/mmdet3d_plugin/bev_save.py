import lmdb
import torch
import zlib
import msgpack
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import threading
import concurrent.futures

class OptimizedFeatureStorage:
    """
    Feature storage with asynchronous write support.
    - Write: synchronous write() and asynchronous write_async()
    - Read: batch read (any size)
    """

    def __init__(self, db_path: str, map_size: int = 1 * 1024**3,
                 compression_level: int = 0, device: str = 'cuda',
                 write_workers: int = 1):
        """
        Args:
            db_path: LMDB database directory
            map_size: max size of database
            compression_level: zlib compression level (0 = no compression)
            device: device to load tensors ('cuda' or 'cpu')
            write_workers: number of threads for asynchronous writes
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.compression_level = compression_level
        self.device = device

        # LMDB environment
        self.env = lmdb.open(
            str(db_path), map_size=map_size, max_dbs=3,
            writemap=True, metasync=False, sync=False, map_async=True, lock=False
        )
        self.data_db = self.env.open_db(b'data')
        self.meta_db = self.env.open_db(b'meta')
        self.index_db = self.env.open_db(b'index')

        # Load index into memory
        self.index: Dict[str, dict] = {}
        self.index_lock = threading.Lock()
        with self.env.begin(db=self.index_db) as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                self.index[k.decode()] = msgpack.loads(v)

        # Thread pool for asynchronous writes
        self.write_executor = concurrent.futures.ThreadPoolExecutor(max_workers=write_workers)

    # ----------------------------------------------------------------------
    # Index helpers (thread-safe)
    # ----------------------------------------------------------------------
    def _update_index(self, key: str, info: dict):
        """Update in-memory index and persist to LMDB (thread-safe)."""
        with self.index_lock:
            self.index[key] = info
        with self.env.begin(db=self.index_db, write=True) as txn:
            txn.put(key.encode(), msgpack.dumps(info))

    # ----------------------------------------------------------------------
    # Compression / decompression
    # ----------------------------------------------------------------------
    def _compress_tensor(self, tensor: torch.Tensor) -> bytes:
        np_arr = tensor.detach().cpu().numpy()
        data = msgpack.dumps({
            'shape': np_arr.shape,
            'dtype': str(np_arr.dtype).encode(),
            'data': np_arr.tobytes()
        })
        return zlib.compress(data, level=self.compression_level) if self.compression_level > 0 else data

    def _decompress_tensor(self, data: bytes) -> torch.Tensor:
        if self.compression_level > 0:
            data = zlib.decompress(data)
        info = msgpack.loads(data)
        np_arr = np.frombuffer(info['data'], dtype=info['dtype']).reshape(info['shape'])
        return torch.from_numpy(np_arr).to(self.device)

    def _compress_composite(self, t1: torch.Tensor, t2: torch.Tensor, f: float) -> bytes:
        c1 = self._compress_tensor(t1)
        c2 = self._compress_tensor(t2)
        return msgpack.dumps({'t1': c1, 't2': c2, 'f': f})

    def _decompress_composite(self, data: bytes) -> Tuple[torch.Tensor, torch.Tensor, float]:
        packed = msgpack.loads(data)
        return (self._decompress_tensor(packed['t1']),
                self._decompress_tensor(packed['t2']),
                packed['f'])

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def write(self, key: str, tensor1: torch.Tensor, tensor2: torch.Tensor,
              float_val: float, metadata: dict = None):
        """Synchronously write composite feature (two tensors + float)."""
        compressed = self._compress_composite(tensor1, tensor2, float_val)
        with self.env.begin(write=True, db=self.data_db) as txn:
            txn.put(key.encode(), compressed)
        if metadata:
            with self.env.begin(write=True, db=self.meta_db) as txn:
                txn.put(key.encode(), msgpack.dumps(metadata))
        self._update_index(key, {
            't1_shape': tensor1.shape, 't1_dtype': str(tensor1.dtype),
            't2_shape': tensor2.shape, 't2_dtype': str(tensor2.dtype),
            'float': float_val
        })

    def write_async(self, key: str, tensor1: torch.Tensor, tensor2: torch.Tensor,
                    float_val: float = 0.0, metadata: dict = None) -> concurrent.futures.Future:
        """
        Asynchronously write composite feature.
        Returns a Future that can be used to wait for completion or check for exceptions.
        """
        return self.write_executor.submit(
            self.write, key, tensor1, tensor2, float_val, metadata
        )

    def read_batch(self, keys: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Read a batch of composite features."""
        result = []
        with self.env.begin(db=self.data_db) as txn:
            for key in keys:
                data = txn.get(key.encode())
                if data is None:
                    raise KeyError(f"Key not found: {key}")
                result.append(self._decompress_composite(data))
        return result

    def read(self, key: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Read a single composite feature."""
        return self.read_batch([key])[0]

    def exists(self, key: str) -> bool:
        """Check if key exists (thread-safe)."""
        with self.index_lock:
            return key in self.index

    def get_info(self, key: str) -> Optional[dict]:
        """Get metadata/index info for a key (thread-safe)."""
        with self.index_lock:
            return self.index.get(key)

    def delete(self, key: str):
        """Delete a key and its associated data."""
        with self.env.begin(write=True, db=self.data_db) as txn:
            txn.delete(key.encode())
        with self.env.begin(write=True, db=self.meta_db) as txn:
            txn.delete(key.encode())
        with self.env.begin(write=True, db=self.index_db) as txn:
            txn.delete(key.encode())
        with self.index_lock:
            self.index.pop(key, None)

    def close(self):
        """Close LMDB and shut down the thread pool (waits for pending writes)."""
        self.write_executor.shutdown(wait=True)
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ================== Example ==================
import time
if __name__ == "__main__":
    with OptimizedFeatureStorage('./History_BEV.lmdb', write_workers=2) as store:
        # Synchronous write
        t1 = torch.randn(256, 200, 200).to('cuda')
        t2 = torch.randn(256, 900).to('cuda')
        tm1 = time.time()
        store.write('sample_sync', t1, t2, 3.14, metadata={'desc': 'sync'})
        tm2 = time.time()
        # Asynchronous write
        future = store.write_async('sample_async', t1 * 2, t2 * 2, 6.28, metadata={'desc': 'async'})
        # Do other work while write is in background...
        # Optionally wait for completion
        tm3 = time.time()
        future.result()  # blocks until write finishes
        tm4 = time.time()
        # Read back
        t1_out, t2_out, f_out = store.read('sample_async')
        tm5 = time.time()
        print("Async write result:", (t1_out - t1*2).sum().item(),
              (t2_out - t2*2).sum().item(), f_out - 6.28)
        print(tm2-tm1,tm3-tm2,tm4-tm3,tm5-tm4)