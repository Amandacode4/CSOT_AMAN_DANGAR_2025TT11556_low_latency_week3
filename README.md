# Week 3 Tick Aggregator

This is my submission for the Week 3 parallel tick aggregator. I focused on making sure the threads don't step on each other's toes and dividing the work efficiently.

## How to build and run
```bash
cmake -B build -DCSOT_AGG_SRC=aggregator.cpp
cmake --build build -j
./build/agg_runner data/large.ticks
```

## My Hardware
- **OS**: macOS
- **CPU**: Apple M4 (10 cores)

## Results
- **Throughput**: ~1678 M ticks/s (on my local machine)
- **Score on Judge**: 159.1
- **Speedup**: ~5.5x compared to the single-threaded version

## What I did to make it fast
1. **Unbalanced Chunks**: I realized the main thread was just sitting around waiting for the workers to finish before it could merge. So I gave the workers a bit more work (26%) and the main thread a bit less (22%). This way the main thread finishes early and starts merging right away while the others finish up.
2. **64-bit Packed Loads**: Since the `symbol_id` and `qty` are right next to each other in the `AggTick` struct (both 32-bit), I used a single 64-bit `memcpy` to load them both at the same time. This halved the memory reads in the tight loop.
3. **Cache Line Padding**: I added `alignas(64)` to the partial arrays so each thread writes to its own cache line. This stopped the false sharing issue where the cores kept invalidating each other's cache.
4. **Branch Prediction**: At first I tried to use branchless programming for the `min` and `max` updates, but it actually slowed things down. I switched to using `__builtin_expect` (branches) because it turns out the CPU's branch predictor is really good at guessing that the all-time min/max won't change, which lets it skip the write instruction entirely.

## What surprised me
I was really surprised that using branches was faster than branchless code. I thought avoiding branches was always the right move, but forcing the CPU to evaluate `std::min` every single time created a data dependency chain that ruined the performance. Letting the branch predictor guess it right 99% of the time was way faster!
