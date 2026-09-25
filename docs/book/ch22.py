"""Chapter 22 — Inside the CPU Backend."""
from gen import *

PART = "IV"


def tile_svg():
    w, h = 470, 150
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    # A (4 rows highlighted), B (2 vectors wide), C tile
    p.append(svg_text(60, 14, "A [m, k]", 8.4, "#56606a"))
    p.append('<rect x="20" y="20" width="80" height="110" fill="#f4f6f7" stroke="#b7c3c8"/>')
    p.append('<rect x="20" y="40" width="80" height="24" fill="#fff4e2" stroke="#a15c00"/>')
    p.append(svg_text(60, 56, "4 rows", 7.6, "#a15c00"))
    p.append(svg_text(200, 14, "B [k, n]", 8.4, "#56606a"))
    p.append('<rect x="140" y="20" width="120" height="110" fill="#f4f6f7" stroke="#b7c3c8"/>')
    p.append('<rect x="170" y="20" width="36" height="110" fill="#e6f2ef" stroke="#0f6b5c"/>')
    p.append(svg_text(188, 140, "2 × 8 cols", 7.6, "#0f6b5c"))
    p.append(svg_text(360, 14, "C [m, n]", 8.4, "#56606a"))
    p.append('<rect x="300" y="20" width="120" height="110" fill="#f4f6f7" stroke="#b7c3c8"/>')
    p.append('<rect x="330" y="40" width="36" height="24" fill="#0f6b5c" stroke="#0f6b5c"/>')
    p.append(svg_text(348, 56, "4×16", 7.6, "#ffffff"))
    p.append(svg_text(360, 90, "8 SIMD registers", 7.6, "#56606a"))
    p.append(svg_text(360, 102, "accumulate over all k", 7.6, "#56606a"))
    p.append(svg_text(120, 80, "×", 12, "#56606a"))
    p.append(svg_text(280, 80, "=", 12, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "cpu",
            "You never call the CPU backend directly, but knowing how it works explains its performance and makes it "
            "easy to extend. It is plain C#: arrays of float, <code>System.Numerics.Vector&lt;float&gt;</code> for SIMD, "
            "<code>Parallel.For</code> for cores, and a register-blocked matrix product. No native code, no BLAS.",
            "Every tensor's data is a pooled <code>float[]</code>; storage is reference-counted and returned to a per-size pool.",
            "Element-wise kernels are structs processing a range with a SIMD loop and a scalar tail; large tensors are split across threads.",
            "Matrix products keep a 4 × (2 · SIMD width) tile of the result in registers across the whole inner loop.",
            "Parallel work starts at 65,536 elements (element-wise) or 131,072 multiply-adds (matrix products); <code>ComputeResources.MaxCpuThreads</code> caps it.",
            "Measured here: up to 129 GFLOP/s for matrix products and 2.5 billion tanh per second on 4 cores.",
        ),
        h2("22.1 Structure"),
        reftable(["Part", "File", "Role"], [
            ["<code>Backend</code>", "Backends/Backend.cs", "The abstract set of primitives every device implements (≈ 60 methods)"],
            ["<code>CpuBackend</code>", "Backends/Cpu/CpuBackend*.cs", "Memory pool, element-wise kernels, reductions, softmax, normalization, im2col, pooling, optimizer steps, decoding kernels"],
            ["<code>CpuMatMul</code>", "Backends/Cpu/CpuMatMul.cs", "The general matrix product (GEMM) with transposes and accumulation"],
            ["<code>MemoryAccountant</code>", "Backends/Backend.cs", "InUse/Cached bookkeeping and limits (" + ch("devices") + ")"],
        ], caption="Table 22.1 — Where the CPU code lives"),
        h2("22.2 Element-wise kernels"),
        para("Each operation is a small <code>readonly struct</code> implementing <code>IRangeKernel.Execute(start, end)</code>. "
             "Being a struct passed as a generic argument, it is specialized and inlined by the JIT: there is no "
             "virtual call or delegate per element. The body views the range as vectors of 8 (or 16) floats and "
             "finishes the remainder one float at a time."),
        snippet("""
            private readonly struct ReluKernel(float[] x, float[] y) : IRangeKernel
            {
                public void Execute(int start, int end)
                {
                    var xs = x.AsSpan(start, end - start);
                    var ys = y.AsSpan(start, end - start);
                    var xv = MemoryMarshal.Cast<float, Vector<float>>(xs);     // whole vectors
                    var yv = MemoryMarshal.Cast<float, Vector<float>>(ys);
                    for (int i = 0; i < xv.Length; i++)
                        yv[i] = Vector.Max(xv[i], Vector<float>.Zero);

                    for (int i = xv.Length * Vector<float>.Count; i < xs.Length; i++)   // scalar tail
                        ys[i] = MathF.Max(xs[i], 0f);
                }
            }

            // Run: inline for small tensors, parallel chunks (multiples of the vector width) for large ones
            internal static void Run<TKernel>(TKernel kernel, int n) where TKernel : struct, IRangeKernel
            {
                if (n < ParallelThreshold || !ComputeResources.AllowParallel) { kernel.Execute(0, n); return; }
                int chunks = ChunkCount(n), size = ChunkSize(n, chunks);
                Parallel.For(0, chunks, ComputeResources.ParallelOptions,
                             c => kernel.Execute(c * size, Math.Min(c * size + size, n)));
            }
            """, caption="From the library source (CpuBackend.cs, abridged)"),
        mex("hardware and element-wise throughput", None,
            """
            Console.WriteLine($"Vector<float>.Count = {Vector<float>.Count}, AVX2 {Avx2.IsSupported}, " +
                              $"AVX-512F {Avx512F.IsSupported}, FMA {Fma.IsSupported}");
            // tanh throughput in billions of elements per second, 1 thread and 4 threads
            """,
            out="""
            Vector<float>.Count = 8, AVX2 True, AVX-512F True, FMA True
            tanh, billions of elements/s:
                 10000   0.16   0.17
                100000   0.21   0.38
               1000000   0.55   2.17
              10000000   0.79   2.45
            """,
            after="Below 65,536 elements a kernel runs on the calling thread (threads would cost more than they "
                  "save), so 4 threads only help from about 100,000 elements. .NET sizes <code>Vector&lt;float&gt;</code> "
                  "at 256 bits (8 floats) by default, even on this AVX-512 CPU."),
        h2("22.3 The matrix product"),
        diagram("Figure 22.1 — Register blocking", tile_svg(),
                "For each block of 4 rows of A and 16 columns of B, eight vector registers hold the 4×16 result tile for the whole k loop."),
        para("Every step of the inner loop loads two vectors from one row of B and four scalars from A, and performs "
             "eight fused multiply-adds: each value loaded from memory is used four or eight times. Row blocks run in "
             "parallel when the product is large enough. A transposed B is copied once into row order so the inner "
             "loop always streams contiguous memory."),
        mex("matrix-product throughput by size", None,
            """
            // Gflops(n, threads): n×n times n×n, repeated for about 2 GFLOP of work after a warm-up
            Console.WriteLine("   n   1 thread   4 threads");
            foreach (int n in new[] { 64, 128, 256, 512, 1024, 2048 })
                Console.WriteLine($"{n,5} {Gflops(n, 1),9:F1} {Gflops(n, 4),11:F1}");
            """,
            out="""
               n   1 thread   4 threads
               64      10.2        29.6
              128      50.7        74.2
              256      58.8        85.3
              512      39.1       128.9
             1024      28.3        94.6
             2048      18.2        72.2
            """),
        cpugpu("one call, two implementations",
               """
               using var a = Tensor.Uniform([512, 512], -1, 1, device: Device.Cpu);
               using var c = a.MatMul(a);      // CpuMatMul: 4x16 register tiles, Parallel.For over row blocks
               """,
               """
               using var a = Tensor.Uniform([512, 512], -1, 1, device: Device.Cuda());
               using var c = a.MatMul(a);      // matmul_f32 PTX kernel: 16x16 shared-memory tiles (Chapter 23)
               """.replace("Chapter 23", ch("cuda")),
               "The tensor's device selects the backend; results agree to float32 rounding."),
        honestbox("Where the CPU GEMM stops scaling",
                  "<p>Performance peaks around n = 256–512 and then falls, because the kernel blocks only for registers, "
                  "not for the CPU caches: for large matrices, rows of B are re-read from main memory. Adding cache "
                  "blocking (packing panels of A and B that fit in L2) is the known next step and would lift large-matrix "
                  "throughput towards the small-matrix peak. The layer sizes in this book mostly fall in the fast range.</p>"),
        h2("22.4 Memory"),
        para("Allocation first looks for a free array of exactly the requested length in a per-length stack; freed "
             "arrays go back there, so a training loop's allocations are all reuses after the first step. Freeing may "
             "happen on the finalizer thread, so the pool is protected by a lock. <code>ReleaseCachedMemory</code> "
             "drops the cached arrays and lets the garbage collector reclaim them."),
        h2("22.5 Adding an operation"),
        deriv("The path of a new element-wise function, e.g. softplus", [
            "Add a member to the <code>UnaryOp</code> enum (Backend.cs).",
            "CPU: write a <code>SoftplusKernel</code> and <code>SoftplusBackwardKernel</code> struct and dispatch them in "
            "<code>Unary</code>/<code>UnaryBackward</code>.",
            "CUDA: add the PTX for the forward and backward kernels (" + ch("cuda") + ") and their names to the kernel list.",
            "Tensor: add <code>public Tensor Softplus() =&gt; Unary(UnaryOp.Softplus);</code> and its name to <code>UnaryNames</code>.",
            "Test: add a gradient check and a CPU/GPU comparison to the test project (" + ch("testing") + ").",
        ]),
        honestbox("Without touching the library",
                  "<p>Most new functions can be composed from existing differentiable operations in your own code, e.g. "
                  "softplus(x) = log(1 + eˣ) as <code>(x.Exp() + 1f).Log()</code>, and wrapped in a <code>Lambda</code> "
                  "layer (" + ch("modules") + "). A dedicated kernel is only worth it for speed or numerical stability.</p>"),
        practice([
            (1, "Why is a kernel a struct passed as a generic type parameter rather than a delegate?",
             "The JIT generates specialized code for each struct type and can inline its <code>Execute</code>, so there is no "
             "per-call indirection and the SIMD loop is optimized as one piece."),
            (1, "How many fused multiply-adds does one inner-loop iteration of the 4×16 tile perform?",
             "Eight vector FMAs of 8 floats each: 64 multiply-adds for 2 vector loads and 4 scalar loads."),
            (2, "On your machine, find the matrix size with the highest single-thread GFLOP/s.",
             "Run the Section 22.3 measurement; expect the peak where A, B and C together still fit in the L2 cache "
             "(n ≈ 128–256 on typical CPUs)."),
            (2, "Implement softplus with existing operations and check its gradient numerically.",
             "<code>static Tensor Softplus(Tensor x) =&gt; (x.Exp() + 1f).Log();</code> then apply the check of " + ch("autograd") +
             " Section 5.6; the gradient should equal sigmoid(x)."),
            (3, "Sketch how to add cache blocking to the GEMM.",
             "Loop over k in panels of, say, 256 and over n in panels that fit in L2; copy (pack) each B panel into a "
             "contiguous buffer once, then run the existing 4×16 register kernel over the panel for every row block, "
             "accumulating into C with beta = 1 after the first panel."),
        ], PART),
        footer("SIMD", "Vector&lt;float&gt;", "Fused multiply-add", "GEMM", "Register blocking", "Cache blocking",
               "Parallel.For", "Memory pool"),
    )
