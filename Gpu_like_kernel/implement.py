
import ctypes

import Metal
import numpy as np


# 1. Initialize Metal device and command queue
device = Metal.MTLCreateSystemDefaultDevice()

if device is None:
    raise RuntimeError("No compatible Metal device found")

command_queue = device.newCommandQueue()

if command_queue is None:
    raise RuntimeError("Failed to create Metal command queue")


# 2. Define GPU key-value store capacity
capacity = 1024 * 1024
buffer_size = capacity * np.dtype(np.uint32).itemsize

EMPTY_KEY = np.uint32(0xFFFFFFFF)


# 3. Initialize input keys
keys_np = np.array(
    [123, 456, 789],
    dtype=np.uint32
)

key_buffer = device.newBufferWithBytes_length_options_(
    keys_np.ctypes.data,
    keys_np.nbytes,
    Metal.MTLResourceStorageModeShared
)

if key_buffer is None:
    raise RuntimeError("Failed to allocate key buffer")


# 4. Allocate GPU hash-key storage
hash_keys_buffer = device.newBufferWithLength_options_(
    buffer_size,
    Metal.MTLResourceStorageModeShared
)

if hash_keys_buffer is None:
    raise RuntimeError("Failed to allocate hash-key buffer")


# 5. Initialize all hash slots to EMPTY_KEY
ctypes.memset(
    hash_keys_buffer.contents(),
    0xFF,
    buffer_size
)


print("Metal device:", device.name())
print("Hash capacity:", capacity)
print("Hash buffer size:", buffer_size)

