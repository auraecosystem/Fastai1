import torch
from torch.utils.benchmark import Timer
size = 128*512
nrep = 100
nbytes_read_write = 4 # this is number of bytes read + written by a kernel. Change this to fit your kernel.

for i in range(10):
    a=torch.empty(size).cuda().uniform_()
    torch.cuda.synchronize()
    out = a.uniform_()
    torch.cuda.synchronize()
    t = Timer(stmt="a.uniform_()", globals=globals())
    res = t.blocked_autorange()
    timec = res.median
    print("uniform, size, elements", size, "forward", timec, "bandwidth (GB/s)", size*(nbytes_read_write)*1e-9/timec)
    size *=2
