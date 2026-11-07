# Results: Measured

Empty until the kernels run on a real GPU. One command fills this folder:

```bash
bash scripts/run_all.sh      # on an A100 / L4 / H100 with PyTorch + CUDA
```

It writes `bench_<gpu>.csv`, the plots, `*_summary.md` with the headline (including % of
FlashAttention-2, which only a Measured run can produce), tile-sweep CSVs and Nsight reports.
Every number here is labeled **Measured**. Simulator numbers live in `../simulated/`.
