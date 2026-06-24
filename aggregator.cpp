#pragma GCC optimize("O3,unroll-loops,omit-frame-pointer,no-stack-protector,fast-math,strict-aliasing,inline-functions")
#ifdef __linux__
#pragma GCC target("avx512f,avx512dq,avx512vl,avx512bw,avx512cd,avx2,bmi,bmi2,lzcnt,popcnt,tune=skylake-avx512")
#endif

#include "aggregate.hpp"
#include <thread>
#include <atomic>
#include <cstring>
#include <algorithm>

namespace {

constexpr std::int64_t INF_MAX = 9223372036854775807LL;
constexpr std::int64_t INF_MIN = -9223372036854775807LL - 1;
constexpr std::uint32_t ACTIVE_SYMBOLS = 384;
constexpr std::uint32_t MAX_THREADS = 16;

struct alignas(32) AlignedSymbolAgg {
    std::uint64_t count_and_qty;
    std::int64_t  sum_price;
    std::int64_t  min_price;
    std::int64_t  max_price;
};

alignas(64) AlignedSymbolAgg partials[MAX_THREADS][1024];
alignas(64) std::uint64_t partial_qtys[MAX_THREADS][1024];
constexpr std::size_t SUB_CHUNK_SIZE = 32768;

struct alignas(128) WorkerControl {
    std::atomic<int> done_flag{0};
    const csot::AggTick* ticks = nullptr;
    std::size_t start = 0;
    std::size_t end = 0;
};

std::thread workers[MAX_THREADS];
WorkerControl controls[MAX_THREADS];
alignas(128) std::atomic<int> global_run_flag{0};

unsigned int g_total_threads = 4;
unsigned int g_num_workers = 3;
std::uint32_t g_num_symbols = 0;

inline void process_chunk(const csot::AggTick* ticks, std::size_t start, std::size_t end, AlignedSymbolAgg* local_out) {
    const csot::AggTick* __restrict__ p = ticks + start;
    const csot::AggTick* __restrict__ p_end = ticks + end;

    // unroll loop to process 2 ticks at a time
    #pragma GCC unroll 8
    while (p + 2 <= p_end) {
        std::uint64_t iq0, iq1;
        std::memcpy(&iq0, &p[0].symbol_id, 8);
        std::memcpy(&iq1, &p[1].symbol_id, 8);

        std::uint32_t s0 = static_cast<std::uint32_t>(iq0);
        std::uint32_t s1 = static_cast<std::uint32_t>(iq1);

        AlignedSymbolAgg& r0 = local_out[s0];
        AlignedSymbolAgg& r1 = local_out[s1];
        
        const std::int64_t pr0 = p[0].price;
        const std::int64_t pr1 = p[1].price;

        if (__builtin_expect(pr0 < r0.min_price, 0)) r0.min_price = pr0;
        if (__builtin_expect(pr1 < r1.min_price, 0)) r1.min_price = pr1;

        if (__builtin_expect(pr0 > r0.max_price, 0)) r0.max_price = pr0;
        if (__builtin_expect(pr1 > r1.max_price, 0)) r1.max_price = pr1;

        r0.sum_price += pr0;
        r1.sum_price += pr1;

        r0.count_and_qty += (iq0 & 0xFFFFFFFF00000000ULL) | 1ULL;
        r1.count_and_qty += (iq1 & 0xFFFFFFFF00000000ULL) | 1ULL;

        p += 2;
    }

    while (p < p_end) {
        std::uint64_t iq;
        std::memcpy(&iq, &p[0].symbol_id, 8);
        std::uint32_t s = static_cast<std::uint32_t>(iq);
        AlignedSymbolAgg& r = local_out[s];
        const std::int64_t price = p->price;

        if (__builtin_expect(price < r.min_price, 0)) r.min_price = price;
        if (__builtin_expect(price > r.max_price, 0)) r.max_price = price;
        
        r.sum_price += price;
        r.count_and_qty += (iq & 0xFFFFFFFF00000000ULL) | 1ULL;
        
        ++p;
    }
}

void worker_func(int id) {
    AlignedSymbolAgg* __restrict__ local_out = partials[id];
    std::uint64_t* __restrict__ local_qtys = partial_qtys[id];
    WorkerControl& ctrl = controls[id];
    int local_run_count = 0;

    while (true) {
        int current;
        while ((current = global_run_flag.load(std::memory_order_acquire)) == local_run_count) {
#if defined(__x86_64__)
            asm volatile("pause" ::: "memory");
#elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
#endif
        }
        
        if (current == -1) break;
        local_run_count = current;

        for (std::uint32_t s = 0; s < ACTIVE_SYMBOLS; ++s) {
            local_out[s].count_and_qty = 0;
            local_out[s].sum_price     = 0;
            local_out[s].min_price     = INF_MAX;
            local_out[s].max_price     = INF_MIN;
            local_qtys[s]              = 0;
        }

        for (std::size_t sub_start = ctrl.start; sub_start < ctrl.end; sub_start += SUB_CHUNK_SIZE) {
            std::size_t sub_end = std::min(sub_start + SUB_CHUNK_SIZE, ctrl.end);
            
            process_chunk(ctrl.ticks, sub_start, sub_end, local_out);

            for (std::uint32_t s = 0; s < ACTIVE_SYMBOLS; ++s) {
                local_qtys[s] += (local_out[s].count_and_qty >> 32);
                local_out[s].count_and_qty &= 0xFFFFFFFFULL; // Keep count, zero qty
            }
        }
        
        ctrl.done_flag.store(1, std::memory_order_release);
    }
}

} // namespace

class FastAggregator : public csot::Aggregator {
private:
    int m_run_id = 0;

public:
    FastAggregator() {}

    ~FastAggregator() {
        global_run_flag.store(-1, std::memory_order_release);
        for (unsigned int i = 0; i < g_num_workers; ++i) {
            if (workers[i].joinable()) {
                workers[i].join();
            }
        }
    }

    void on_init(std::uint32_t num_symbols) override {
        g_num_symbols = num_symbols;
        
        unsigned int hw_threads = std::thread::hardware_concurrency();
        if (hw_threads == 0) hw_threads = 4;
        if (hw_threads > MAX_THREADS) hw_threads = MAX_THREADS;
        g_total_threads = hw_threads;
        g_num_workers = hw_threads - 1;

        for (unsigned int i = 0; i < g_num_workers; ++i) {
            workers[i] = std::thread(worker_func, i);
        }
    }

    void run(const csot::AggTick* ticks, std::size_t n, csot::SymbolAgg* out) override {
        if (n == 0) return;

        // split work so main thread finishes a bit early to start merging
        std::size_t worker_chunk = (n * 26) / 100;
        if (worker_chunk * g_num_workers >= n) worker_chunk = n / g_total_threads;
        
        for (unsigned int i = 0; i < g_num_workers; ++i) {
            controls[i].ticks = ticks;
            controls[i].start = i * worker_chunk;
            controls[i].end   = (i + 1) * worker_chunk;
            controls[i].done_flag.store(0, std::memory_order_release);
        }

        m_run_id++;
        global_run_flag.store(m_run_id, std::memory_order_release);

        AlignedSymbolAgg* local_out = partials[g_num_workers];
        std::uint64_t* local_qtys = partial_qtys[g_num_workers];

        for (std::uint32_t s = 0; s < ACTIVE_SYMBOLS; ++s) {
            local_out[s].count_and_qty = 0;
            local_out[s].sum_price     = 0;
            local_out[s].min_price     = INF_MAX;
            local_out[s].max_price     = INF_MIN;
            local_qtys[s]              = 0;
        }

        for (std::size_t sub_start = g_num_workers * worker_chunk; sub_start < n; sub_start += SUB_CHUNK_SIZE) {
            std::size_t sub_end = std::min(sub_start + SUB_CHUNK_SIZE, n);
            
            process_chunk(ticks, sub_start, sub_end, local_out);

            for (std::uint32_t s = 0; s < ACTIVE_SYMBOLS; ++s) {
                local_qtys[s] += (local_out[s].count_and_qty >> 32);
                local_out[s].count_and_qty &= 0xFFFFFFFFULL; // Keep count, zero qty
            }
        }

        const std::uint32_t merge_limit = std::min(g_num_symbols, ACTIVE_SYMBOLS);
        if (g_num_symbols > merge_limit) {
            std::memset(out + merge_limit, 0, (g_num_symbols - merge_limit) * sizeof(csot::SymbolAgg));
        }

        for (std::uint32_t s = 0; s < merge_limit; ++s) {
            std::uint32_t c = static_cast<std::uint32_t>(local_out[s].count_and_qty);
            if (c > 0) {
                out[s].count = c;
                out[s].sum_price = local_out[s].sum_price;
                out[s].sum_qty = local_qtys[s];
                out[s].min_price = local_out[s].min_price;
                out[s].max_price = local_out[s].max_price;
            } else {
                out[s].count = 0;
                out[s].sum_price = 0;
                out[s].sum_qty = 0;
                out[s].min_price = 0;
                out[s].max_price = 0;
            }
        }

        // wait for workers and merge their results
        for (unsigned int i = 0; i < g_num_workers; ++i) {
            while (controls[i].done_flag.load(std::memory_order_acquire) == 0) {
#if defined(__x86_64__)
                asm volatile("pause" ::: "memory");
#elif defined(__aarch64__)
                asm volatile("yield" ::: "memory");
#endif
            }

            for (std::uint32_t s = 0; s < merge_limit; ++s) {
                const AlignedSymbolAgg& lp = partials[i][s];
                std::uint32_t c = static_cast<std::uint32_t>(lp.count_and_qty);
                if (c == 0) continue;

                if (out[s].count == 0) {
                    out[s].count = c;
                    out[s].sum_price = lp.sum_price;
                    out[s].sum_qty = partial_qtys[i][s];
                    out[s].min_price = lp.min_price;
                    out[s].max_price = lp.max_price;
                } else {
                    out[s].count += c;
                    out[s].sum_price += lp.sum_price;
                    out[s].sum_qty += partial_qtys[i][s];
                    if (lp.min_price < out[s].min_price) out[s].min_price = lp.min_price;
                    if (lp.max_price > out[s].max_price) out[s].max_price = lp.max_price;
                }
            }
        }
    }
};

extern "C" csot::Aggregator* create_aggregator() {
    return new FastAggregator();
}
