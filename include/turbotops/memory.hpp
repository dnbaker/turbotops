// SPDX-License-Identifier: MIT
// memory.hpp — physical memory query and peak resident set size.
#ifndef TURBOTOPS_MEMORY_HPP
#define TURBOTOPS_MEMORY_HPP

#include <sys/resource.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include "core.hpp"

namespace tops {

inline u64 physical_memory_bytes() {
#if defined(__APPLE__)
    u64 mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) return mem;
    return 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES), page = sysconf(_SC_PAGE_SIZE);
    return pages > 0 && page > 0 ? u64(pages) * u64(page) : 0;
#endif
}

// Default working-memory budget: half of physical memory (8 GiB if unknown).
inline u64 default_memory_budget() {
    const u64 m = physical_memory_bytes();
    return m ? m / 2 : (u64(8) << 30);
}

inline double peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
    return double(ru.ru_maxrss) / (1024.0 * 1024.0);   // bytes
#else
    return double(ru.ru_maxrss) / 1024.0;              // kilobytes
#endif
}

}  // namespace tops

#endif  // TURBOTOPS_MEMORY_HPP
