// SPDX-License-Identifier: MIT
// turbotops — command-line interface.
//
//   turbotops run <input> [options]       persistence diagram to stdout
//   turbotops gen <kind> <n> <dim> <seed> <output>
//   turbotops test [--ripser PATH]        self-test (+ parity vs the ripser CLI)
//
// See `turbotops --help` for options.
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "turbotops/turbotops.hpp"

using namespace tops;

namespace {

const char* kUsage =
    "usage:\n"
    "  turbotops run <input> [options]   compute a persistence diagram\n"
    "  turbotops gen <uniform|gauss|torus|sphere> <n> <dim> <seed> <output>\n"
    "  turbotops test [--ripser PATH]    self-test; parity vs the ripser CLI\n"
    "  turbotops --version               print the version and exit\n"
    "\n"
    "input (run):\n"
    "  text   one point per line, coordinates separated by commas or spaces\n"
    "         ('#' starts a comment)\n"
    "  .bin   raw little-endian float32, row-major; requires --dim\n"
    "  --edges  input is a weighted edge list 'i j w' (0-based), one per line\n"
    "\n"
    "options:\n"
    "  --maxdim D          highest homology dimension (default 1)\n"
    "  --threshold T       filtration cutoff, inclusive (default inf)\n"
    "  --metric M          l1 | l2 | linf | lp:<p>, p >= 1 (default l2)\n"
    "  --graph full|knn    complex: all pairs, or the kNN graph (default full)\n"
    "  -k N                neighbors for --graph knn (default 15)\n"
    "  --approximate       kNN via HNSW instead of exact search\n"
    "  --hnsw-m N, --ef-construction N, --ef-search N   HNSW parameters\n"
    "  --verify-recall S   with --approximate: measure kNN recall on S points\n"
    "                      against exact search (S = 0: all points)\n"
    "  --dim D             coordinates per point for .bin input\n"
    "  --mode M            boundary | coboundary | auto (default auto)\n"
    "  --coboundary        force the coboundary engine (cohomology reducer;\n"
    "                      general path: any maxdim <= 30, 64/128-bit indices)\n"
    "  --no-coboundary,\n"
    "  --boundary          force the boundary engine (RedZeD; single-threaded,\n"
    "                      64-bit simplex indices, n < 2^31 — errors if the\n"
    "                      problem does not fit)\n"
    "  --engine E          legacy alias for --mode: reducer = coboundary,\n"
    "                      redzed = boundary, auto = auto\n"
    "  --threads N         worker threads (default: all cores)\n"
    "  --memory-budget B   bytes, with optional k/m/g suffix (default: half of RAM)\n"
    "  --output FILE       write the diagram to FILE instead of stdout\n"
    "  --quiet             no summary on stderr\n"
    "\n"
    "mode auto (default): coboundary is on by default; boundary (RedZeD) is\n"
    "chosen only where measurements show it faster — H1 on dense full-Rips\n"
    "input or graphs of mean degree >= 16 — and only when the complex fits\n"
    "64-bit simplex indices. Both engines give byte-identical diagrams; the\n"
    "chosen mode and why are printed on stderr.\n"
    "\n"
    "output: one bar per line, 'dim birth death' (death 'inf' for essential\n"
    "classes); bars of zero length are omitted.\n";

[[noreturn]] void die(const std::string& m) {
    std::fprintf(stderr, "turbotops: %s\n", m.c_str());
    std::exit(2);
}

bool ends_with(const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

u64 parse_bytes(const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    double mul = 1;
    if (end && *end) {
        switch (*end | 0x20) {
            case 'k': mul = 1024.0; break;
            case 'm': mul = 1024.0 * 1024; break;
            case 'g': mul = 1024.0 * 1024 * 1024; break;
            default: die("bad byte count '" + s + "'");
        }
    }
    if (!(v > 0)) die("bad byte count '" + s + "'");
    return u64(v * mul);
}

u64 parse_u64(const std::string& s, const char* what) {
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno || !end || *end || s.empty()) die(std::string("bad ") + what + " '" + s + "'");
    return v;
}

// Checked narrowing: a raw u32() cast would silently wrap values past 2^32-1
// (and strtoull accepts "-1" as 2^64 - 1 without setting errno).
u32 parse_u32(const std::string& s, const char* what) {
    const u64 v = parse_u64(s, what);
    if (v > u64(~0u)) die(std::string("bad ") + what + " '" + s + "' (out of range)");
    return u32(v);
}

f32 parse_f32(const std::string& s, const char* what) {
    char* end = nullptr;
    const f32 v = std::strtof(s.c_str(), &end);
    if (!end || *end || s.empty()) die(std::string("bad ") + what + " '" + s + "'");
    return v;
}

// ---- input ----------------------------------------------------------------
std::vector<f32> read_text_points(const std::string& path, u32& n, u32& dim) {
    std::ifstream in(path);
    if (!in) die("cannot open " + path);
    {   // strip a UTF-8 BOM if present (common in exported CSVs)
        char bom[3];
        in.read(bom, 3);
        if (!(in.gcount() == 3 && (u8)bom[0] == 0xEF && (u8)bom[1] == 0xBB && (u8)bom[2] == 0xBF)) {
            in.clear();
            in.seekg(0);
        }
    }
    std::vector<f32> x;
    std::string line;
    n = 0;
    dim = 0;
    while (std::getline(in, line)) {
        if (auto h = line.find('#'); h != std::string::npos) line.resize(h);
        for (char& ch : line)
            if (ch == ',' || ch == '\t' || ch == ';') ch = ' ';
        std::istringstream ss(line);
        std::string tok;
        u32 c = 0;
        while (ss >> tok) {
            x.push_back(parse_f32(tok, "coordinate"));
            ++c;
        }
        if (!c) continue;
        if (!dim) dim = c;
        else if (c != dim) die(path + ": inconsistent row length at point " + std::to_string(n));
        ++n;
    }
    return x;
}

std::vector<f32> read_bin_points(const std::string& path, u32 dim, u32& n) {
    if (!dim) die("--dim is required for .bin input");
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) die("cannot open " + path);
    const u64 bytes = u64(in.tellg());
    if (bytes % (u64(dim) * 4)) die(path + ": size is not a multiple of 4 * dim");
    n = u32(bytes / (u64(dim) * 4));
    std::vector<f32> x(u64(n) * dim);
    in.seekg(0);
    in.read(reinterpret_cast<char*>(x.data()), std::streamsize(bytes));
    return x;
}

CsrGraph read_edges(const std::string& path) {
    std::ifstream in(path);
    if (!in) die("cannot open " + path);
    std::vector<std::vector<std::pair<u32, f32>>> rows;
    std::string line;
    u64 lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        if (auto h = line.find('#'); h != std::string::npos) line.resize(h);
        for (char& ch : line)
            if (ch == ',' || ch == '\t') ch = ' ';
        std::istringstream ss(line);
        std::string a, b, w;
        if (!(ss >> a)) continue;
        if (!(ss >> b >> w)) die(path + ":" + std::to_string(lineno) + ": expected 'i j w'");
        const u64 i = parse_u64(a, "vertex"), j = parse_u64(b, "vertex");
        const f32 d = parse_f32(w, "weight");
        if (i == j) continue;
        if (!(d >= 0) || !std::isfinite(d)) die(path + ":" + std::to_string(lineno) + ": bad weight");
        const u64 mx = std::max(i, j);
        if (mx >= u64(~0u)) die("vertex id too large");
        if (rows.size() <= mx) rows.resize(mx + 1);
        rows[i].emplace_back(u32(j), d);
        rows[j].emplace_back(u32(i), d);
    }
    // Keep the smallest weight of repeated edges.
    for (auto& r : rows) {
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end(),
                            [](const auto& p, const auto& q) { return p.first == q.first; }),
                r.end());
    }
    return CsrGraph::from_rows(rows);
}

void write_diagram(FILE* f, const Barcode& bc) {
    // Canonical order (dimension, then birth, then death): both engines emit
    // the same bars but discover them in different orders, and mode selection
    // must never change the output, only the speed.
    struct Bar {
        u32 d;
        f32 b, e;
    };
    std::vector<Bar> bars;
    for (u32 d = 0; d < bc.ndims(); ++d)
        for (u64 i = 0; i < bc.birth[d].size(); ++i)
            bars.push_back({d, bc.birth[d][i], bc.death[d][i]});
    std::stable_sort(bars.begin(), bars.end(), [](const Bar& a, const Bar& b) {
        if (a.d != b.d) return a.d < b.d;
        if (a.b != b.b) return a.b < b.b;
        return a.e < b.e;
    });
    for (const Bar& r : bars)
        if (std::isinf(r.e))
            std::fprintf(f, "%u %.9g inf\n", r.d, double(r.b));
        else
            std::fprintf(f, "%u %.9g %.9g\n", r.d, double(r.b), double(r.e));
}

// ---- gen --------------------------------------------------------------------
std::vector<f32> generate(const std::string& kind, u32 n, u32 dim, u64 seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<f32> U(0.0f, 1.0f);
    std::normal_distribution<f32> N(0.0f, 1.0f);
    std::vector<f32> x(u64(n) * dim, 0.0f);
    if (kind == "uniform") {
        for (auto& v : x) v = U(rng);
    } else if (kind == "gauss") {
        for (auto& v : x) v = N(rng);
    } else if (kind == "sphere") {   // unit sphere S^{dim-1}
        for (u32 i = 0; i < n; ++i) {
            f32 s = 0;
            for (u32 k = 0; k < dim; ++k) s += (x[u64(i) * dim + k] = N(rng)) * x[u64(i) * dim + k];
            s = std::sqrt(s);
            for (u32 k = 0; k < dim; ++k) x[u64(i) * dim + k] /= s;
        }
    } else if (kind == "torus") {    // standard torus in R^3 (R = 2, r = 1)
        if (dim != 3) die("torus requires dim 3");
        for (u32 i = 0; i < n; ++i) {
            const f32 a = 6.283185307f * U(rng), b = 6.283185307f * U(rng);
            x[u64(i) * 3 + 0] = (2 + std::cos(a)) * std::cos(b);
            x[u64(i) * 3 + 1] = (2 + std::cos(a)) * std::sin(b);
            x[u64(i) * 3 + 2] = std::sin(a);
        }
    } else {
        die("unknown cloud kind '" + kind + "'");
    }
    return x;
}

void write_points(const std::string& path, const std::vector<f32>& x, u32 dim) {
    if (ends_with(path, ".bin")) {
        std::ofstream o(path, std::ios::binary);
        o.write(reinterpret_cast<const char*>(x.data()), std::streamsize(x.size() * 4));
        if (!o) die("cannot write " + path);
        return;
    }
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) die("cannot write " + path);
    for (u64 i = 0; i < x.size(); i += dim) {
        for (u32 k = 0; k < dim; ++k) std::fprintf(f, k ? ",%.9g" : "%.9g", double(x[i + k]));
        std::fputc('\n', f);
    }
    std::fclose(f);
}

// ---- test -------------------------------------------------------------------
struct Tally {
    int pass = 0, fail = 0, skip = 0;
    void check(bool ok, const std::string& what) {
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
        ok ? ++pass : ++fail;
    }
};

using Bars = std::vector<std::vector<std::pair<double, double>>>;

Bars to_bars(const Barcode& bc) {
    Bars b(bc.ndims());
    for (u32 d = 0; d < bc.ndims(); ++d) {
        for (u64 i = 0; i < bc.birth[d].size(); ++i)
            if (bc.death[d][i] > bc.birth[d][i]) b[d].emplace_back(bc.birth[d][i], bc.death[d][i]);
        std::sort(b[d].begin(), b[d].end());
    }
    return b;
}

bool same_bars(Bars a, Bars b, double rtol, std::string& why) {
    const size_t D = std::max(a.size(), b.size());
    a.resize(D);
    b.resize(D);
    for (size_t d = 0; d < D; ++d) {
        if (a[d].size() != b[d].size()) {
            why = "H" + std::to_string(d) + " has " + std::to_string(a[d].size()) + " vs " +
                  std::to_string(b[d].size()) + " bars";
            return false;
        }
        for (size_t i = 0; i < a[d].size(); ++i) {
            const double v[4] = {a[d][i].first, a[d][i].second, b[d][i].first, b[d][i].second};
            for (int t = 0; t < 2; ++t) {
                const double x = v[t], y = v[t + 2];
                if (std::isinf(x) != std::isinf(y) ||
                    (std::isfinite(x) && std::fabs(x - y) > rtol * std::max(1.0, std::fabs(y)))) {
                    char buf[160];
                    std::snprintf(buf, sizeof buf, "H%zu bar %zu: (%.9g, %.9g) vs (%.9g, %.9g)", d, i,
                                  v[0], v[1], v[2], v[3]);
                    why = buf;
                    return false;
                }
            }
        }
    }
    return true;
}

std::string find_ripser(const std::string& hint) {
    std::vector<std::string> cand;
    if (!hint.empty()) cand.push_back(hint);
    cand.push_back("third_party/ripser/ripser");
    if (const char* p = std::getenv("PATH")) {
        std::string s(p);
        size_t a = 0;
        while (a <= s.size()) {
            size_t b = s.find(':', a);
            if (b == std::string::npos) b = s.size();
            cand.push_back(s.substr(a, b - a) + "/ripser");
            a = b + 1;
        }
    }
    for (const auto& c : cand)
        if (access(c.c_str(), X_OK) == 0) return c;
    return "";
}

// Run the ripser CLI on a file and parse its intervals.
bool run_ripser(const std::string& ripser, const std::string& fmt, const std::string& file,
                u32 maxdim, Bars& out) {
    const std::string cmd = "'" + ripser + "' --format " + fmt + " --dim " +
                            std::to_string(maxdim) + " '" + file + "' 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return false;
    out.assign(maxdim + 1, {});
    char line[512];
    int d = -1;
    while (std::fgets(line, sizeof line, p)) {
        int dd;
        if (std::sscanf(line, "persistence intervals in dim %d:", &dd) == 1) {
            d = dd;
            continue;
        }
        const char* lb = std::strchr(line, '[');
        if (d < 0 || !lb) continue;
        double b = std::strtod(lb + 1, nullptr);
        const char* comma = std::strchr(lb, ',');
        if (!comma) continue;
        const char* q = comma + 1;
        while (*q == ' ') ++q;
        const double e = (*q == ')') ? INFINITY : std::strtod(q, nullptr);
        if (u32(d) <= maxdim && e > b) out[d].emplace_back(b, e);
    }
    const int rc = pclose(p);
    for (auto& v : out) std::sort(v.begin(), v.end());
    return rc == 0;
}

int run_tests(const std::string& ripser_hint) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    Tally T;
    std::printf("unit tests:\n");

    T.check(TURBOTOPS_VERSION_MAJOR == 0 && TURBOTOPS_VERSION_MINOR == 1 &&
                TURBOTOPS_VERSION_PATCH == 0 && std::string(TURBOTOPS_VERSION_STRING) == "0.1.0",
            "version macros are set and consistent with v0.1.0");

    // Metrics: symmetry, identity, triangle inequality, large-p stability.
    {
        std::mt19937_64 rng(7);
        std::normal_distribution<f32> N(0.0f, 3.0f);
        const f32 ps[] = {1.0f, 1.5f, 2.0f, 3.0f, 7.5f, 50.0f, INFINITY};
        for (u32 dim : {3u, 17u}) {
            std::vector<f32> x(3 * dim);
            bool sym = true, ident = true, tri = true;
            for (f32 p : ps) {
                const Metric m = Metric::minkowski(p);
                for (int rep = 0; rep < 200; ++rep) {
                    for (auto& v : x) v = N(rng);
                    const f32 *a = x.data(), *b = a + dim, *c = b + dim;
                    sym &= m(a, b, dim) == m(b, a, dim);
                    ident &= m(a, a, dim) == 0.0f;
                    tri &= m(a, c, dim) <= (m(a, b, dim) + m(b, c, dim)) * (1 + 1e-5f);
                }
            }
            T.check(sym && ident && tri, "Minkowski p in {1,1.5,2,3,7.5,50,inf}, dim " +
                                             std::to_string(dim) + ": symmetry, identity, triangle");
        }
        std::vector<f32> a(9, 0.0f), b(9);
        for (u32 k = 0; k < 9; ++k) b[k] = 1000.0f * (k + 1);
        const f32 d50 = Metric::minkowski(50.0f)(a.data(), b.data(), 9);
        const f32 dinf = Metric::minkowski(INFINITY)(a.data(), b.data(), 9);
        T.check(std::isfinite(d50) && d50 >= dinf && d50 <= dinf * 1.05f,
                "lp:50 on coordinates of size 1e4 is finite and close to linf");
        T.check(std::fabs(Metric::minkowski(3.0f)(a.data(), b.data(), 1) - 1000.0f) < 1e-3f,
                "lp:3 in one dimension equals |x - y|");
    }

    // Exact neighbor structures against brute force, all metrics.
    {
        const u32 n = 300, dim = 4;
        auto x = generate("gauss", n, dim, 11);
        for (const char* ms : {"l1", "l2", "linf", "lp:1.5", "lp:4"}) {
            const PointCloud c{x.data(), n, dim, Metric::parse(ms)};
            const auto L = knn_lists_exact(c, 7);
            bool ok = true;
            for (u32 i = 0; i < n && ok; ++i) {
                std::vector<std::pair<f32, u32>> all;
                for (u32 j = 0; j < n; ++j)
                    if (j != i) all.emplace_back(c(i, j), j);
                std::sort(all.begin(), all.end());
                for (u32 t = 0; t < 7; ++t) ok &= L.row(i)[t] == all[t].second;
            }
            const f32 tau = 1.2f;
            const CsrGraph g = radius_graph(c, tau);
            u64 m = 0;
            for (u32 i = 0; i < n; ++i)
                for (u32 j = i + 1; j < n; ++j) m += c(i, j) <= tau;
            T.check(ok && g.valid() && g.m == m,
                    std::string("exact kNN and radius graph vs brute force (") + ms + ")");
        }
    }

    // Internal consistency: 64- vs 128-bit indices, threads, MST vs dense H0,
    // dense vs sparse complete graph, known topology.
    {
        auto x = generate("torus", 160, 3, 3);
        const PointCloud c{x.data(), 160, 3, Metric::parse("l2")};
        const Bars a = to_bars(rips::compute_dense(160, c, INFINITY, 2));
        std::string why;
        T.check(same_bars(a, to_bars(rips::compute_dense(160, c, INFINITY, 2, nullptr, true)), 0, why),
                "128-bit simplex indices give identical diagrams " + why);
        const u32 nt = num_threads();
        set_num_threads(1);
        const Bars one = to_bars(rips::compute_dense(160, c, INFINITY, 2));
        set_num_threads(nt);
        T.check(same_bars(a, one, 0, why), "1 thread and all threads give identical diagrams " + why);
        Bars h0a(1, a[0]), h0b = to_bars(mst::h0(c));
        h0b.resize(1);
        T.check(same_bars(h0a, h0b, 0, why), "MST H0 equals the Rips H0 " + why);
        const CsrGraph full = radius_graph(c, 1e30f);
        rips::DenseComplex dc = rips::make_dense(160, c);
        dc.threshold = dc.enclosing_radius();
        T.check(same_bars(to_bars(rips::compute(dc, 2)),
                          to_bars(rips::compute_graph(full, 2, dc.threshold)), 0, why),
                "dense matrix and complete sparse graph agree " + why);
        auto s = generate("sphere", 120, 3, 5);
        const PointCloud cs{s.data(), 120, 3, Metric::parse("l2")};
        const Barcode bs = rips::compute_dense(120, cs, INFINITY, 2);
        u64 ess = 0, long2 = 0;
        for (u32 d = 1; d <= 2; ++d)
            for (u64 i = 0; i < bs.birth[d].size(); ++i) {
                ess += std::isinf(bs.death[d][i]);
                long2 += d == 2 && bs.death[d][i] - bs.birth[d][i] > 0.5f;
            }
        T.check(ess == 0 && long2 == 1, "S^2 sample: one dominant H2 class, no essential H1/H2");
    }

    // Robustness gates and option validation: degenerate parameters must be
    // loud errors (std::invalid_argument), never silent garbage; the dense
    // lower layout and the HNSW path must agree with their exact
    // counterparts.
    std::printf("validation and coverage:\n");
    {
        auto x = generate("torus", 120, 3, 13);
        const PointCloud c{x.data(), 120, 3, Metric::parse("l2")};
        auto throws_invalid = [&](auto&& fn) {
            try {
                fn();
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        };
        bool ok = true;
        {
            Options o;
            o.threshold = -1.0f;
            ok &= throws_invalid([&] { persistence(x.data(), 120, 3, o); });
            o.threshold = std::nanf("");
            ok &= throws_invalid([&] { persistence(x.data(), 120, 3, o); });
            o.threshold = INFINITY;
            o.graph = Options::Graph::Knn;
            o.k = 0;
            ok &= throws_invalid([&] { persistence(x.data(), 120, 3, o); });
        }
        {
            Options o;
            o.maxdim = 31;
            o.mode = Mode::Boundary;
            ok &= throws_invalid([&] { persistence(x.data(), 120, 3, o); });
            ok &= throws_invalid([&] { (void)parse_mode("diagonal"); });
            ok &= throws_invalid([&] { (void)parse_engine("warp"); });
        }
        {
            CsrGraph bad = knn_graph(c, knn_lists_exact(c, 6));
            std::swap(bad.weights[0], bad.weights[bad.weights.size() - 1]);
            ok &= throws_invalid([&] { persistence(bad, 1); });
            ok &= throws_invalid([&] { persistence(bad, 1, std::nanf("")); });
            ok &= throws_invalid([&] { persistence(bad, 1, -0.5f); });
        }
        {
            std::vector<f32> bad_x(x);
            bad_x[7] = std::nanf("");
            const PointCloud bc{bad_x.data(), 120, 3, Metric::parse("l2")};
            ok &= throws_invalid([&] { (void)mst::h0(bc); });
        }
        T.check(ok, "option validation: bad threshold/k/maxdim/graph/coordinates rejected");

        // MST threshold: deaths past the cutoff become essential classes.
        {
            const Bars all = to_bars(mst::h0(c));
            u64 n_fin = 0;
            for (const auto& pr : all[0])
                if (!std::isinf(pr.second)) ++n_fin;
            const f32 thr = all[0][n_fin / 2].second;   // median finite death
            u64 all_gt = 0;                             // finite deaths past the cutoff
            for (const auto& pr : all[0])
                if (!std::isinf(pr.second) && pr.second > thr) ++all_gt;
            const Bars cut = to_bars(mst::h0(c, thr));
            u64 cut_ess = 0, cut_fin = 0;
            bool mok = true;
            for (const auto& pr : cut[0])
                if (std::isinf(pr.second)) ++cut_ess;
                else { ++cut_fin; mok = mok && pr.second <= thr; }
            mok = mok && cut_ess == 1 + all_gt && cut_fin == n_fin - all_gt;
            T.check(mok, "MST threshold cuts deaths into essential classes");
        }
        // Dense lower layout (forced by a small budget) matches the full one.
        {
            std::string why;
            rips::RipsStats full_st, low_st;
            const Bars bf = to_bars(rips::compute_dense(120, c, INFINITY, 2, &full_st));
            const u64 full_bytes = 120ull * 120 * 4, low_bytes = 120ull * 119 / 2 * 4;
            const u64 budget = (full_bytes + low_bytes) / 2;
            const Bars bl = to_bars(rips::compute_dense(120, c, INFINITY, 2, &low_st, false, budget));
            T.check(!full_st.wide_index && !low_st.wide_index &&
                        same_bars(bf, bl, 0, why),
                    "dense lower layout equals the full layout " + why);
        }
#ifndef TURBOTOPS_NO_HNSW
        // HNSW approximate kNN: valid graph, high measured recall.
        {
            const KnnLists L = knn_lists_hnsw(c, 10, HnswParams{});
            const CsrGraph g = knn_graph(c, L);
            const Recall R = knn_recall(c, L, 64);
            T.check(L.k == 10 && g.valid() && g.m > 0 && R.recall() >= 0.7 && R.checked_points == 64,
                    "HNSW kNN builds a valid graph, recall " + std::to_string(R.recall()) +
                        " on 64 points");
        }
#endif
    }

    // RedZeD engine vs the cohomology reducer: identical complexes, exact equality.
    std::printf("redzed parity (vs cohomology reducer):\n");
    {
        struct Case {
            const char* cloud;
            u32 n, dim, maxdim;
            const char* metric;
        };
        const Case cases[] = {
            {"torus", 150, 3, 2, "l2"},   {"sphere", 70, 4, 3, "l2"},    {"uniform", 40, 3, 4, "l2"},
            {"gauss", 90, 5, 2, "l1"},    {"gauss", 90, 5, 2, "linf"},   {"gauss", 90, 5, 2, "lp:3"},
            {"grid", 64, 3, 2, "l1"},     {"grid", 64, 3, 3, "linf"},    {"torus", 400, 3, 1, "l2"},
            {"uniform", 300, 3, 2, "l2"}, {"grid", 200, 2, 2, "l2"},     {"sphere", 150, 3, 2, "l1"},
        };
        for (const Case& cs : cases) {
            std::vector<f32> x;
            if (std::string(cs.cloud) == "grid") {
                std::mt19937_64 rng(9);
                std::uniform_int_distribution<int> U(0, 3);
                x.resize(u64(cs.n) * cs.dim);
                for (auto& v : x) v = f32(U(rng));
            } else {
                x = generate(cs.cloud, cs.n, cs.dim, 21);
            }
            const PointCloud c{x.data(), cs.n, cs.dim, Metric::parse(cs.metric)};
            const std::string tag = std::string(cs.cloud) + " n=" + std::to_string(cs.n) + " " +
                                    cs.metric + " maxdim=" + std::to_string(cs.maxdim);
            rips::DenseComplex dc = rips::make_dense(cs.n, c);
            dc.threshold = dc.enclosing_radius();
            std::string why;
            T.check(same_bars(to_bars(redzed::compute(dc, cs.maxdim)), to_bars(rips::compute(dc, cs.maxdim)), 0, why),
                    "dense  " + tag + " " + why);
            const f32 tau = 0.6f * dc.threshold;
            for (int mode = 0; mode < 3; ++mode) {
                const CsrGraph g = mode == 0 ? knn_graph(c, knn_lists_exact(c, 8))
                                 : mode == 1 ? radius_graph(c, tau)
                                             : radius_graph(c, 1e30f);
                const f32 thr = mode == 2 ? dc.threshold : INFINITY;
                const Bars rz = to_bars(redzed::compute_graph(g, cs.maxdim, thr));
                T.check(same_bars(rz, to_bars(rips::compute_graph(g, cs.maxdim, thr)), 0, why),
                        std::string(mode == 0 ? "kNN-8  " : mode == 1 ? "radius " : "full-g ") + tag + " " + why);
                if (mode == 1)
                    T.check(same_bars(rz, to_bars(redzed::compute_graph(g, cs.maxdim, thr, nullptr, true)), 0, why),
                            "radius " + tag + " u32 and u64 edge ranks agree " + why);
            }
        }
    }

    // Mode selection (--mode boundary|coboundary|auto): the selector must
    // never change the diagram, only the speed; a forced boundary on a
    // problem RedZeD cannot index must error, not switch silently.
    std::printf("mode coverage (--mode boundary|coboundary|auto):\n");
    {
        struct Case {
            const char* cloud;
            u32 n, dim, maxdim;
        };
        const Case cases[] = {
            {"torus", 150, 3, 1},   // dense H1: auto picks boundary
            {"torus", 150, 3, 2},   // dense H2: auto picks coboundary
            {"sphere", 60, 4, 3},   // dense H3: auto picks coboundary
        };
        for (const Case& cs : cases) {
            const auto x = generate(cs.cloud, cs.n, cs.dim, 21);
            const std::string tag = std::string(cs.cloud) + " n=" + std::to_string(cs.n) +
                                    " maxdim=" + std::to_string(cs.maxdim);
            std::string why;
            Options o;
            o.maxdim = cs.maxdim;
            const Bars auto_bars = to_bars(persistence(x.data(), cs.n, cs.dim, o).barcode);
            o.mode = Mode::Coboundary;
            const Bars cob_bars = to_bars(persistence(x.data(), cs.n, cs.dim, o).barcode);
            o.mode = Mode::Boundary;
            const Bars bnd_bars = to_bars(persistence(x.data(), cs.n, cs.dim, o).barcode);
            T.check(same_bars(auto_bars, cob_bars, 0, why) &&
                        same_bars(auto_bars, bnd_bars, 0, why),
                    "dense  " + tag + " auto/coboundary/boundary identical " + why);
        }
        // Sparse graphs, low vs high mean degree, through the CsrGraph entry.
        {
            const auto x = generate("gauss", 200, 4, 31);
            const PointCloud c{x.data(), 200, 4, Metric::parse("l2")};
            for (u32 k : {8u, 64u}) {
                const CsrGraph g = knn_graph(c, knn_lists_exact(c, k));
                const std::string tag = "kNN-" + std::to_string(k);
                std::string why;
                const Bars auto_bars = to_bars(persistence(g, 1, INFINITY, 0, Mode::Auto).barcode);
                const Bars cob_bars = to_bars(persistence(g, 1, INFINITY, 0, Mode::Coboundary).barcode);
                const Bars bnd_bars = to_bars(persistence(g, 1, INFINITY, 0, Mode::Boundary).barcode);
                T.check(same_bars(auto_bars, cob_bars, 0, why) &&
                            same_bars(auto_bars, bnd_bars, 0, why),
                        "sparse " + tag + " auto/coboundary/boundary identical " + why);
            }
        }
        // Forced boundary past RedZeD's 64-bit index limit must error.
        {
            const auto x = generate("uniform", 2000, 3, 41);
            bool threw = false;
            try {
                Options o;
                o.maxdim = 8;
                o.mode = Mode::Boundary;
                persistence(x.data(), 2000, 3, o);
            } catch (const std::invalid_argument&) {
                threw = true;
            }
            T.check(threw, "forced boundary on C(2000, 10) > 2^64 errors (no silent switch)");
        }
        // Auto-rule pins: the selector rule documented at
        // kBoundaryMinMeanDegree above.
        {
            std::string why;
            T.check(detail::select_mode(Mode::Auto, 1000, 1, INFINITY, why) == Engine::RedZeD &&
                        detail::select_mode(Mode::Auto, 1000, 1, 64.0, why) == Engine::RedZeD &&
                        detail::select_mode(Mode::Auto, 1000, 1, 20.0, why) == Engine::RedZeD,
                    "auto picks boundary for H1 on dense / mean degree >= 16");
            T.check(detail::select_mode(Mode::Auto, 1000, 2, INFINITY, why) == Engine::Reducer &&
                        detail::select_mode(Mode::Auto, 1000, 1, 12.0, why) == Engine::Reducer,
                    "auto picks coboundary for H >= 2 and for mean degree < 16");
            T.check(detail::select_mode(Mode::Auto, 5000000, 1, 64.0, why) == Engine::Reducer,
                    "auto capability gate: past 64-bit indices, boundary is unreachable");
            T.check(detail::select_mode(Mode::Boundary, 100, 0, 8.0, why) == Engine::Reducer,
                    "forced boundary at maxdim 0 degrades to the MST path");
        }
    }

    // Parity with the ripser CLI on identical distances.
    const std::string ripser = find_ripser(ripser_hint);
    std::printf("ripser parity:\n");
    if (ripser.empty()) {
        std::printf("  [SKIP] ripser CLI not found (pass --ripser PATH or run "
                    "scripts/fetch_ripser.sh)\n");
        ++T.skip;
    } else {
        std::printf("  using %s\n", ripser.c_str());
        char tmpl[] = "/tmp/turbotops_test_XXXXXX";
        const char* dir = mkdtemp(tmpl);
        if (!dir) die("mkdtemp failed");
        const std::string file = std::string(dir) + "/input.txt";
        struct Case {
            const char* cloud;
            u32 n, dim, maxdim;
            const char* metric;
        };
        const Case cases[] = {
            {"torus", 150, 3, 2, "l2"},      {"sphere", 70, 4, 3, "l2"},
            {"uniform", 40, 3, 4, "l2"},     {"gauss", 90, 5, 2, "l1"},
            {"gauss", 90, 5, 2, "linf"},     {"gauss", 90, 5, 2, "lp:1.5"},
            {"gauss", 90, 5, 2, "lp:3"},     {"grid", 64, 3, 2, "l1"},
            {"grid", 64, 3, 2, "linf"},      {"torus", 400, 3, 1, "l2"},
        };
        for (const Case& cs : cases) {
            std::vector<f32> x;
            if (std::string(cs.cloud) == "grid") {   // integer lattice: many exact ties
                std::mt19937_64 rng(9);
                std::uniform_int_distribution<int> U(0, 3);
                x.resize(u64(cs.n) * cs.dim);
                for (auto& v : x) v = f32(U(rng));
            } else {
                x = generate(cs.cloud, cs.n, cs.dim, 21);
            }
            const PointCloud c{x.data(), cs.n, cs.dim, Metric::parse(cs.metric)};
            const std::string tag = std::string(cs.cloud) + " n=" + std::to_string(cs.n) +
                                    " " + cs.metric + " maxdim=" + std::to_string(cs.maxdim);
            // dense: full Rips
            {
                FILE* f = std::fopen(file.c_str(), "w");
                for (u32 i = 1; i < cs.n; ++i) {
                    for (u32 j = 0; j < i; ++j) std::fprintf(f, j ? ",%.9g" : "%.9g", double(c(i, j)));
                    std::fputc('\n', f);
                }
                std::fclose(f);
                Bars ref, ours = to_bars(rips::compute_dense(cs.n, c, INFINITY, cs.maxdim));
                std::string why;
                const bool ran = run_ripser(ripser, "lower-distance", file, cs.maxdim, ref);
                T.check(ran && same_bars(ours, ref, 2e-5, why), "dense  " + tag + " " + why);
            }
            // sparse: exact kNN graph, and a radius graph at 0.6 x the
            // enclosing radius (sparse inputs have no enclosing-radius cutoff)
            if (cs.maxdim > 3) continue;
            rips::DenseComplex dc = rips::make_dense(cs.n, c);
            const f32 tau = 0.6f * dc.enclosing_radius();
            for (int mode = 0; mode < 2; ++mode) {
                CsrGraph g = mode == 0 ? knn_graph(c, knn_lists_exact(c, 8)) : radius_graph(c, tau);
                FILE* f = std::fopen(file.c_str(), "w");
                for (u32 v = 0; v < g.n; ++v)
                    for (u64 t = g.begin(v); t < g.end(v); ++t)
                        if (g.col_idx[t] > v) std::fprintf(f, "%u %u %.9g\n", v, g.col_idx[t], double(g.weights[t]));
                std::fclose(f);
                Bars ref, ours = to_bars(rips::compute_graph(g, cs.maxdim));
                // ripser sizes a sparse input by its largest edge endpoint, so
                // isolated vertices after it are absent there: drop one
                // essential H0 class per such vertex.
                u32 trailing = 0;
                for (u32 v = g.n; v-- > 0 && g.degree(v) == 0;) ++trailing;
                for (u32 t = 0; t < trailing && !ours[0].empty(); ++t) ours[0].pop_back();
                std::string why;
                const bool ran = run_ripser(ripser, "sparse", file, cs.maxdim, ref);
                T.check(ran && same_bars(ours, ref, 2e-5, why),
                        std::string(mode == 0 ? "kNN-8  " : "radius ") + tag + " " + why);
            }
        }
        std::remove(file.c_str());
        rmdir(dir);
    }
    std::printf("%d passed, %d failed, %d skipped\n", T.pass, T.fail, T.skip);
    return T.fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty() || args[0] == "-h" || args[0] == "--help" || args[0] == "help") {
        std::fputs(kUsage, args.empty() ? stderr : stdout);
        return args.empty() ? 2 : 0;
    }
    if (args[0] == "--version") {
        std::printf("turbotops %s\n", TURBOTOPS_VERSION_STRING);
        return 0;
    }
    const std::string cmd = args[0];
    try {
        if (cmd == "gen") {
            if (args.size() != 6) die("gen <kind> <n> <dim> <seed> <output>");
            const u32 n = parse_u32(args[2], "n"), dim = parse_u32(args[3], "dim");
            write_points(args[5], generate(args[1], n, dim, parse_u64(args[4], "seed")), dim);
            return 0;
        }
        if (cmd == "test") {
            std::string hint;
            for (size_t i = 1; i < args.size(); ++i) {
                if (args[i] == "--ripser") {
                    if (i + 1 >= args.size()) die("missing value for --ripser");
                    hint = args[++i];
                } else die("unknown test option '" + args[i] + "'");
            }
            return run_tests(hint);
        }
        if (cmd != "run") die("unknown command '" + cmd + "' (try --help)");

        Options o;
        std::string input, output;
        bool edges = false, quiet = false, verify = false;
        bool mode_locked = false;   // set once by a mode flag; conflicts die
        auto set_mode = [&](Mode m, const std::string& flag) {
            if (mode_locked && o.mode != m)
                die(flag + " conflicts with an earlier mode/engine flag");
            mode_locked = true;
            o.mode = m;
        };
        u32 dim = 0;
        for (size_t i = 1; i < args.size(); ++i) {
            const std::string& a = args[i];
            auto val = [&]() -> const std::string& {
                if (i + 1 >= args.size()) die("missing value for " + a);
                return args[++i];
            };
            if (a == "--maxdim") o.maxdim = parse_u32(val(), "maxdim");
            else if (a == "--threshold") o.threshold = parse_f32(val(), "threshold");
            else if (a == "--metric") o.metric = val();
            else if (a == "--mode") set_mode(parse_mode(val()), a);
            else if (a == "--coboundary") set_mode(Mode::Coboundary, a);
            else if (a == "--no-coboundary" || a == "--boundary") set_mode(Mode::Boundary, a);
            else if (a == "--engine") {
                const Engine e = parse_engine(val());
                set_mode(e == Engine::Reducer ? Mode::Coboundary
                         : e == Engine::RedZeD ? Mode::Boundary : Mode::Auto,
                         a);
            }
            else if (a == "--graph") {
                const std::string& g = val();
                if (g == "full") o.graph = Options::Graph::Full;
                else if (g == "knn") o.graph = Options::Graph::Knn;
                else die("--graph must be full or knn");
            } else if (a == "-k") o.k = parse_u32(val(), "k");
            else if (a == "--approximate") o.approximate = true;
#ifndef TURBOTOPS_NO_HNSW
            else if (a == "--hnsw-m") o.hnsw.M = parse_u32(val(), "M");
            else if (a == "--ef-construction") o.hnsw.ef_construction = parse_u32(val(), "ef");
            else if (a == "--ef-search") o.hnsw.ef_search = parse_u32(val(), "ef");
#endif
            else if (a == "--verify-recall") {
                verify = true;
                o.recall_samples = parse_u64(val(), "samples");
            } else if (a == "--dim") dim = parse_u32(val(), "dim");
            else if (a == "--threads") o.threads = parse_u32(val(), "threads");
            else if (a == "--memory-budget") o.memory_budget = parse_bytes(val());
            else if (a == "--output") output = val();
            else if (a == "--edges") edges = true;
            else if (a == "--quiet") quiet = true;
            else if (!a.empty() && a[0] == '-') die("unknown option '" + a + "'");
            else if (input.empty()) input = a;
            else die("unexpected argument '" + a + "'");
        }
        if (input.empty()) die("run: missing input file");
        if (verify && !o.approximate) die("--verify-recall requires --approximate");
        if (verify && o.graph != Options::Graph::Knn) die("--verify-recall requires --graph knn");
        if (o.approximate && o.graph != Options::Graph::Knn)
            die("--approximate requires --graph knn");

        // Open the output before computing: a bad path must not waste the run.
        FILE* f = stdout;
        if (!output.empty() && !(f = std::fopen(output.c_str(), "w"))) die("cannot write " + output);

        Result r;
        if (edges) {
            const CsrGraph g = read_edges(input);
            if (!g.n) die(input + ": no vertices (empty edge list?)");
            r = persistence(g, o.maxdim, o.threshold, o.threads, o.mode);
        } else {
            u32 n = 0;
            std::vector<f32> x = ends_with(input, ".bin") ? read_bin_points(input, dim, n)
                                                          : read_text_points(input, n, dim);
            if (!n) die(input + ": no points (empty file, directory, or wrong format?)");
            if (verify && o.recall_samples == 0) o.recall_samples = n;
            r = persistence(x.data(), n, dim, o);
        }

        write_diagram(f, r.barcode);
        if (f != stdout) std::fclose(f);

        if (!quiet) {
            const Stats& s = r.stats;
            std::fprintf(stderr, "turbotops: mode: %s\n", s.mode_reason.c_str());
            std::fprintf(stderr, "turbotops: method=%s threshold=%g edges=%llu index_bits=%d "
                                 "graph=%.3fs total=%.3fs peak_rss=%.1fMB\n",
                         s.method.c_str(), double(s.threshold), (unsigned long long)s.edges,
                         s.wide_index ? 128 : 64, s.seconds_graph, s.seconds_total, s.peak_rss_mb);
            if (s.engine == "coboundary" && !s.columns.empty()) {
                // RipsStats layout: apparent[0] counts edges skipped in the H0
                // pass; apparent[d] (d >= 1) counts columns of dimension d + 1
                // skipped as apparent during assembly; emergent[d - 1] counts
                // dim-d columns closed by the emergent shortcut.
                std::fprintf(stderr, "turbotops: coboundary telemetry:");
                for (size_t d = 0; d < s.columns.size(); ++d) {
                    const u64 ap = d + 1 < s.apparent.size() ? s.apparent[d + 1] : 0;
                    const u64 em = d < s.emergent.size() ? s.emergent[d] : 0;
                    std::fprintf(stderr, " H%zu columns=%llu skipped=%llu", d + 1,
                                 (unsigned long long)s.columns[d],
                                 (unsigned long long)(ap + em));
                }
                std::fputc('\n', stderr);
            }
            for (u32 d = 0; d < r.barcode.ndims(); ++d) {
                u64 fin = 0, ess = 0;
                for (f32 de : r.barcode.death[d]) std::isinf(de) ? ++ess : ++fin;
                std::fprintf(stderr, "turbotops: H%u %llu finite, %llu essential\n", d,
                             (unsigned long long)fin, (unsigned long long)ess);
            }
            if (s.recall.checked_points)
                std::fprintf(stderr,
                             "turbotops: kNN recall %.6f on %llu points; exact lists on %llu "
                             "(95%% lower bound on exact-list rate %.6f)\n",
                             s.recall.recall(), (unsigned long long)s.recall.checked_points,
                             (unsigned long long)s.recall.points_exact,
                             s.recall.exact_rate_lower95());
        }
    } catch (const std::exception& e) {
        die(e.what());
    }
    return 0;
}
