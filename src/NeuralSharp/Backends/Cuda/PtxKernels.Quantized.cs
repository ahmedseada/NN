using System.Text;

namespace NeuralSharp.Backends.Cuda;

// PTX for int8 weight-only quantization. Weights q[k, n] are signed bytes packed four per 32-bit word along n (each
// row padded to n4 = ceil(n / 4) words), with one float scale per column: w[k, j] = q[k, j] · scale[j].
internal static partial class PtxKernels
{
    public static readonly string[] QuantizedNames = ["int8_matmul_f32", "int8_dequant_f32"];

    private static void BuildQuantized(StringBuilder sb)
    {
        // y[r, 4c..4c+3] = scale · Σ_k x[r, k] · q[k, 4c..4c+3]: one thread per (row, word). Adjacent threads read adjacent
        // words of a weight row, so a warp reads 128 contiguous bytes per step (the weights are read once per row).
        Elementwise(sb, "int8_matmul_f32", ["x", "q", "s", "y"], [("u32", "k"), ("u32", "n4"), ("u32", "cols")], $"""
            div.u32 %r5, %i, %s_n4;
            rem.u32 %r6, %i, %s_n4;
            mul.lo.u32 %r7, %r5, %s_k;
            mul.wide.u32 %rd1, %r7, 4;
            add.u64 %rd2, %b_x, %rd1;
            mul.wide.u32 %rd3, %r6, 4;
            add.u64 %rd4, %b_q, %rd3;
            mul.wide.u32 %rd5, %s_n4, 4;
            mov.f32 %f1, {Zero};
            mov.f32 %f2, {Zero};
            mov.f32 %f3, {Zero};
            mov.f32 %f4, {Zero};
            mov.u32 %r8, 0;
            QM_LOOP:
            setp.ge.u32 %p1, %r8, %s_k;
            @%p1 bra QM_END;
            ld.global.f32 %f5, [%rd2];
            ld.global.u32 %r9, [%rd4];
            bfe.s32 %r10, %r9, 0, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f1, %f5, %f6, %f1;
            bfe.s32 %r10, %r9, 8, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f2, %f5, %f6, %f2;
            bfe.s32 %r10, %r9, 16, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f3, %f5, %f6, %f3;
            bfe.s32 %r10, %r9, 24, 8;
            cvt.rn.f32.s32 %f6, %r10;
            fma.rn.f32 %f4, %f5, %f6, %f4;
            add.u64 %rd2, %rd2, 4;
            add.u64 %rd4, %rd4, %rd5;
            add.u32 %r8, %r8, 1;
            bra QM_LOOP;
            QM_END:
            shl.b32 %r11, %r6, 2;
            mul.wide.u32 %rd6, %r11, 4;
            add.u64 %rd7, %b_s, %rd6;
            mul.lo.u32 %r12, %r5, %s_cols;
            add.u32 %r12, %r12, %r11;
            mul.wide.u32 %rd8, %r12, 4;
            add.u64 %rd9, %b_y, %rd8;
            """ + "\n" + StoreScaledColumns("QM", "%f1", "%f2", "%f3", "%f4"));

        // w[kk, 4c..4c+3] = q[kk, 4c..4c+3] · scale: one thread per packed word.
        Elementwise(sb, "int8_dequant_f32", ["q", "s", "w"], [("u32", "n4"), ("u32", "cols")], """
            ld.global.u32 %r9, [%a_q];
            div.u32 %r5, %i, %s_n4;
            rem.u32 %r6, %i, %s_n4;
            bfe.s32 %r10, %r9, 0, 8;
            cvt.rn.f32.s32 %f1, %r10;
            bfe.s32 %r10, %r9, 8, 8;
            cvt.rn.f32.s32 %f2, %r10;
            bfe.s32 %r10, %r9, 16, 8;
            cvt.rn.f32.s32 %f3, %r10;
            bfe.s32 %r10, %r9, 24, 8;
            cvt.rn.f32.s32 %f4, %r10;
            shl.b32 %r11, %r6, 2;
            mul.wide.u32 %rd6, %r11, 4;
            add.u64 %rd7, %b_s, %rd6;
            mul.lo.u32 %r12, %r5, %s_cols;
            add.u32 %r12, %r12, %r11;
            mul.wide.u32 %rd8, %r12, 4;
            add.u64 %rd9, %b_w, %rd8;
            """ + "\n" + StoreScaledColumns("QD", "%f1", "%f2", "%f3", "%f4"));
    }

    // Stores value[t] · scale[column + t] for the (up to) four columns starting at %r11 that are below n.
    // Expects %rd7 = &scale[column] and %rd9 = &out[row, column].
    private static string StoreScaledColumns(string prefix, params string[] values)
    {
        var sb = new StringBuilder();
        for (int t = 0; t < values.Length; t++)
        {
            sb.AppendLine($"add.u32 %r13, %r11, {t};");
            sb.AppendLine("setp.ge.u32 %p2, %r13, %s_cols;");
            sb.AppendLine($"@%p2 bra {prefix}_STORED;");
            sb.AppendLine($"ld.global.f32 %f7, [%rd7+{4 * t}];");
            sb.AppendLine($"mul.f32 %f8, {values[t]}, %f7;");
            sb.AppendLine($"st.global.f32 [%rd9+{4 * t}], %f8;");
        }

        sb.Append($"{prefix}_STORED:");
        return sb.ToString();
    }
}
