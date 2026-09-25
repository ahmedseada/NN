"""Chapter 3 — Tensors."""
from gen import *

PART = "I"


def layout_svg():
    w, h = 470, 120
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    vals = [1, 2, 3, 4, 5, 6]
    # the grid [2, 3]
    for r in range(2):
        for c in range(3):
            x, y = 20 + c * 36, 22 + r * 34
            p.append(f'<rect x="{x}" y="{y}" width="34" height="32" rx="3" fill="{"#e6f2ef" if r == 0 else "#fff4e2"}" stroke="#0f6b5c" stroke-width="1.1"/>')
            p.append(svg_text(x + 17, y + 20, str(vals[r * 3 + c]), 10))
    p.append(svg_text(74, 14, "shape [2, 3]", 8.5, "#56606a"))
    p.append(svg_text(74, 108, "rows × columns", 8, "#56606a"))
    p.append('<line x1="140" y1="55" x2="190" y2="55" stroke="#56606a" stroke-width="1.1"/>')
    p.append('<polygon points="190,55 183,51 183,59" fill="#56606a"/>')
    for i, v in enumerate(vals):
        x = 200 + i * 42
        p.append(f'<rect x="{x}" y="39" width="40" height="32" rx="3" fill="{"#e6f2ef" if i < 3 else "#fff4e2"}" stroke="#0f6b5c" stroke-width="1.1"/>')
        p.append(svg_text(x + 20, 59, str(v), 10))
        p.append(svg_text(x + 20, 86, f"[{i}]", 7.6, "#56606a"))
    p.append(svg_text(325, 30, "memory: one contiguous row-major buffer", 8.5, "#56606a"))
    p.append(svg_text(325, 104, "element (r, c) sits at index r·3 + c", 8, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "tensors",
            "The tensor is the only data type the math of the library understands. Inputs, targets, weights, "
            "gradients, predictions and losses are all tensors. This chapter covers how to create them, read them "
            "back, combine them, reshape them and move them between devices, with every operation shown on real output.",
            "A tensor is a row-major block of float32 values with a <b>shape</b> and a <b>device</b>.",
            "Create from .NET arrays (<code>Tensor.From</code>) or by value (<code>Zeros</code>, <code>Ones</code>, <code>Full</code>, <code>Uniform</code>, <code>Normal</code>); any numeric type converts to float32.",
            "Operators <code>+ - *</code> are element-wise; <code>MatMul</code> is the matrix product.",
            "<code>+</code> broadcasts a smaller tensor over the trailing dimensions (a bias over rows); other binary operators need equal shapes.",
            "<code>Reshape</code> and <code>Flatten</code> share data; <code>Permute</code>, <code>Transpose</code>, <code>Narrow</code>, <code>Concat</code> and <code>Stack</code> copy.",
        ),
        h2("3.1 Shape, rank and size"),
        para("A tensor's <b>shape</b> lists the length of each dimension, outermost first. A batch of 32 images of "
             "28×28 grey pixels has shape <code>[32, 1, 28, 28]</code>. The <b>rank</b> is the number of dimensions and "
             "the <b>size</b> the total number of elements. A rank-0 tensor (shape <code>[]</code>) is a scalar, such as a loss."),
        diagram("Figure 3.1 — Row-major layout", layout_svg(),
                "The values of a [2, 3] tensor are stored row after row in one buffer; the last dimension changes fastest."),
        mex("inspecting a tensor", None,
            """
            using NeuralSharp;

            var m = Tensor.From([1f, 2, 3, 4, 5, 6], [2, 3]);     // values, shape
            Console.WriteLine($"shape [{string.Join(", ", m.Shape.ToArray())}]  rank {m.Rank}  size {m.Size}");
            Console.WriteLine(m);
            """,
            out="""
            shape [2, 3]  rank 2  size 6
            Tensor(shape=[2, 3], device=cpu)
            [[1.0000, 2.0000, 3.0000]
             [4.0000, 5.0000, 6.0000]]
            """),
        reftable(["Data", "Typical shape", "Dimensions mean"], [
            ["Table of features", "<code>[N, F]</code>", "rows (samples), features"],
            ["Targets for regression", "<code>[N, 1]</code> or <code>[N, K]</code>", "rows, outputs"],
            ["Class labels for SparseCrossEntropy", "<code>[N]</code>", "class index per row, stored as float"],
            ["Images", "<code>[N, C, H, W]</code>", "batch, channels, height, width (NCHW)"],
            ["Sequences of vectors", "<code>[N, T, F]</code>", "batch, time steps, features"],
            ["Token ids", "<code>[N, T]</code>", "batch, positions"],
            ["Weights of <code>Linear(in, out)</code>", "<code>[in, out]</code>", "so that <code>x.MatMul(W)</code> works"],
        ], caption="Table 3.1 — Shape conventions used throughout the library"),
        h2("3.2 Creating tensors"),
        para("Every factory takes an optional <code>device</code> (default: <code>Device.Default</code>) and "
             "<code>requiresGrad</code> (default: false; see " + ch("autograd") + "). Values are always stored as "
             "float32, the type every kernel computes in; the generic <code>From&lt;T&gt;</code> overloads convert "
             "<code>int</code>, <code>double</code>, <code>byte</code>, <code>Half</code>, <code>decimal</code> and any "
             "other .NET number type."),
        reftable(["Factory", "Result"], [
            ["<code>Tensor.From(float[] v)</code>", "1-D tensor <code>[v.Length]</code>"],
            ["<code>Tensor.From(float[,] v)</code>", "2-D tensor <code>[rows, cols]</code>"],
            ["<code>Tensor.From(ReadOnlySpan&lt;float&gt; v, ReadOnlySpan&lt;int&gt; shape)</code>", "Any shape; <code>v.Length</code> must equal the element count"],
            ["<code>Tensor.From&lt;T&gt;(…)</code>", "Same three forms for any numeric <code>T</code>, converted to float32"],
            ["<code>Tensor.Scalar(x)</code>", "Rank-0 tensor"],
            ["<code>Tensor.Zeros(shape)</code>, <code>Ones(shape)</code>, <code>Full(shape, x)</code>", "Constant tensors"],
            ["<code>Tensor.Uniform(shape, low, high, random?)</code>", "Values uniform in [low, high)"],
            ["<code>Tensor.Normal(shape, mean, std, random?)</code>", "Normally distributed values (glossary <b>Normal distribution</b>)"],
            ["<code>Tensor.OneHot(indices, classes)</code>", "<code>[..., classes]</code> rows with a single 1 at each index"],
        ], caption="Table 3.2 — Creating tensors"),
        mex("the factories in action",
            "Collection expressions (<code>[1f, 2, 3]</code>) work anywhere a span or an array is expected. Pass a seeded "
            "<code>Random</code> to make random tensors reproducible.",
            """
            var ints  = Tensor.From<int>([1, 2, 3], [3]);                 // int -> float32
            var dbl   = Tensor.From(new double[,] { { 0.5, 1.5 } });      // double[,] -> [1, 2]
            var sevens = Tensor.Full([2, 2], 7f);
            var noise = Tensor.Normal([1000], mean: 0f, std: 1f, random: new Random(7));

            Console.WriteLine(ints);
            Console.WriteLine(dbl);
            Console.WriteLine(sevens);
            Console.WriteLine($"noise mean {noise.Mean().Item():F3}");
            """,
            out="""
            Tensor(shape=[3], device=cpu)
            [1.0000, 2.0000, 3.0000]
            Tensor(shape=[1, 2], device=cpu)
            [[0.5000, 1.5000]]
            Tensor(shape=[2, 2], device=cpu)
            [[7.0000, 7.0000]
             [7.0000, 7.0000]]
            noise mean -0.003
            """),
        cpugpu("creating data on a chosen device",
               """
               using var x = Tensor.From(features, device: Device.Cpu);
               using var w = Tensor.Normal([784, 128], 0f, 0.05f, device: Device.Cpu);
               """,
               """
               var gpu = Device.Cuda();
               using var x = Tensor.From(features, device: gpu);    // uploaded once
               using var w = Tensor.Normal([784, 128], 0f, 0.05f, device: gpu);
               """,
               "Random values are generated on the CPU with <code>System.Random</code> and uploaded, so the same seed "
               "gives the same numbers on both devices."),
        h2("3.3 Reading values back"),
        reftable(["Method", "Returns"], [
            ["<code>ToArray()</code>", "<code>float[]</code>, all elements in row-major order"],
            ["<code>ToArray2D()</code>", "<code>float[,]</code> for a rank-2 tensor"],
            ["<code>ToArray&lt;T&gt;()</code>", "<code>T[]</code>, converted (saturating; integers truncate toward zero)"],
            ["<code>Item()</code>", "The single value of a one-element tensor (a loss, a mean)"],
            ["<code>CopyTo(Span&lt;float&gt; dest, int offset)</code>", "Copies <code>dest.Length</code> elements starting at <code>offset</code>, without allocating"],
            ["<code>ToString()</code>", "Shape, device and up to 64 values (a grid for small matrices)"],
        ], caption="Table 3.3 — Reading data"),
        mex("reading a [2, 3] tensor five ways", None,
            """
            Console.WriteLine(string.Join(", ", m.ToArray()));
            float[,] grid = m.ToArray2D();
            Console.WriteLine(grid[1, 2]);
            Console.WriteLine(m.Sum().Item());
            Console.WriteLine(string.Join(", ", m.ToArray<int>()));

            Span<float> window = stackalloc float[2];
            m.CopyTo(window, offset: 3);                              // elements 3 and 4
            Console.WriteLine($"{window[0]} {window[1]}");
            """,
            out="""
            1, 2, 3, 4, 5, 6
            6
            21
            1, 2, 3, 4, 5, 6
            4 5
            """),
        trap("reading values inside a hot GPU loop",
             "<p>On the GPU every read copies data to the CPU and waits for all queued work. Calling "
             "<code>loss.Item()</code> on every step of a fast loop can halve its speed. Read once every few "
             "dozen steps, or let the <code>Trainer</code> do it (it accumulates losses on the device).</p>"),
        h2("3.4 Arithmetic and broadcasting"),
        para("The operators <code>+</code>, <code>-</code> and <code>*</code> between two tensors work element by "
             "element and need equal shapes, with one exception: <code>a + b</code> <b>broadcasts</b> <code>b</code> "
             "when its shape equals the trailing dimensions of <code>a</code>, repeating it over the leading ones. That "
             "is exactly what adding a bias vector <code>[F]</code> to a batch <code>[N, F]</code> needs, or a mask "
             "<code>[T, T]</code> to attention scores <code>[B, T, T]</code>. A number combined with a tensor "
             "(<code>2f * x</code>, <code>x - 1f</code>, <code>x / 4f</code>) applies to every element."),
        mex("operators", None,
            """
            var bias = Tensor.From([10f, 20f, 30f]);
            Console.WriteLine(m + bias);          // [2,3] + [3]: broadcast over rows
            Console.WriteLine(m * m);             // element-wise product
            Console.WriteLine(2f * m - 1f);       // scalars apply everywhere
            Console.WriteLine(m / 2f);
            """,
            out="""
            Tensor(shape=[2, 3], device=cpu)
            [[11.0000, 22.0000, 33.0000]
             [14.0000, 25.0000, 36.0000]]
            Tensor(shape=[2, 3], device=cpu)
            [[1.0000, 4.0000, 9.0000]
             [16.0000, 25.0000, 36.0000]]
            Tensor(shape=[2, 3], device=cpu)
            [[1.0000, 3.0000, 5.0000]
             [7.0000, 9.0000, 11.0000]]
            Tensor(shape=[2, 3], device=cpu)
            [[0.5000, 1.0000, 1.5000]
             [2.0000, 2.5000, 3.0000]]
            """),
        reftable(["Expression", "Shapes allowed", "Notes"], [
            ["<code>a + b</code>", "equal, or <code>b</code> = trailing dims of <code>a</code>", "broadcasting only for <code>+</code>, and only on the right"],
            ["<code>a - b</code>, <code>a * b</code>", "equal", "<code>*</code> is element-wise, not the matrix product"],
            ["<code>a ± x</code>, <code>x ± a</code>, <code>a * x</code>, <code>x * a</code>, <code>a / x</code>, <code>-a</code>", "any", "<code>x</code> is a <code>float</code>"],
        ], caption="Table 3.4 — Operators"),
        trap("expecting NumPy-style broadcasting everywhere",
             "<p><code>a - b</code> with <code>b</code> of shape <code>[F]</code> throws <i>Shapes … do not match</i>. "
             "Write <code>a + (-1f * b)</code>, or keep the value as a float when it is a single number "
             "(<code>a - mean.Item()</code>). There is no tensor-by-tensor division; multiply by a reciprocal computed "
             "on the host when you need one.</p>"),
        h2("3.5 Element-wise functions"),
        reftable(["Method", "Computes", "Range / note"], [
            ["<code>Sigmoid()</code>", "1 / (1 + e<sup>−x</sup>)", "(0, 1); probabilities of yes/no"],
            ["<code>Tanh()</code>", "hyperbolic tangent", "(−1, 1); zero-centred"],
            ["<code>Relu()</code>", "max(x, 0)", "[0, ∞); the usual hidden activation"],
            ["<code>Gelu()</code>", "smooth ReLU (tanh approximation)", "transformer activation"],
            ["<code>Exp()</code>, <code>Log()</code>", "e<sup>x</sup>, natural log", "Log needs x &gt; 0"],
            ["<code>Square()</code>, <code>Abs()</code>", "x², |x|", "used by MSE and MAE"],
            ["<code>Dropout(p, seed)</code>", "zeroes a fraction p, scales the rest by 1/(1−p)", "the mask comes from the seed, costs no memory"],
        ], caption="Table 3.5 — Element-wise functions (all differentiable)"),
        honestbox("The math behind the activations",
                  "<p>Their formulas, derivatives, ranges and effect on gradients are the subject of the Activation "
                  "Functions volume; exponentials and logarithms are covered in the Pre-Calc volume <i>Numbers, "
                  "Shapes, and Functions</i>. Here they are simply tools; " + ch("dense") + " shows how to choose one.</p>"),
        h2("3.6 Reductions, softmax and classes"),
        para("<code>Sum()</code> and <code>Mean()</code> without arguments reduce everything to a scalar. With a "
             "dimension they reduce only along it; negative dimensions count from the end, and "
             "<code>keepDim: true</code> leaves a dimension of size 1 so the result lines up with the input. "
             "<code>Softmax()</code> turns each row of scores into probabilities that sum to 1 (glossary <b>Softmax</b>), "
             "and <code>ArgMax()</code> returns the index of the largest value in each row."),
        mex("reducing along a dimension and picking classes", None,
            """
            Console.WriteLine(m.Sum(0));                          // down the columns -> [3]
            Console.WriteLine(m.Mean(1, keepDim: true));          // across each row  -> [2, 1]

            var logits = Tensor.From(new float[,] { { 2, 1, 0.1f }, { 0.5f, 2.5f, 0.2f } });
            Console.WriteLine(logits.Softmax());
            Console.WriteLine(logits.ArgMax());
            Console.WriteLine(Tensor.OneHot(Tensor.From([2f, 0f]), 3));
            """,
            out="""
            Tensor(shape=[3], device=cpu)
            [5.0000, 7.0000, 9.0000]
            Tensor(shape=[2, 1], device=cpu)
            [[2.0000]
             [5.0000]]
            Tensor(shape=[2, 3], device=cpu)
            [[0.6590, 0.2424, 0.0986]
             [0.1095, 0.8093, 0.0811]]
            Tensor(shape=[2], device=cpu)
            [0.0000, 1.0000]
            Tensor(shape=[2, 3], device=cpu)
            [[0.0000, 0.0000, 1.0000]
             [1.0000, 0.0000, 0.0000]]
            """),
        reftable(["Method", "Shape in → out", "Differentiable"], [
            ["<code>Sum()</code>, <code>Mean()</code>", "any → <code>[]</code>", "yes"],
            ["<code>Sum(dim, keepDim)</code>, <code>Mean(dim, keepDim)</code>", "drops (or keeps as 1) <code>dim</code>", "yes"],
            ["<code>Softmax()</code>, <code>LogSoftmax()</code>", "same shape; over the last dimension", "yes"],
            ["<code>ArgMax()</code>", "drops the last dimension; indices as floats", "no"],
            ["<code>Tensor.OneHot(indices, k)</code>", "<code>[...]</code> → <code>[..., k]</code>", "no (targets only)"],
        ], caption="Table 3.6 — Reductions and classification helpers"),
        h2("3.7 Matrix products"),
        para("<code>a.MatMul(b)</code> is the matrix product, the operation that does almost all of the arithmetic "
             "in a network. It accepts three shape patterns; the transpose flags use the last two dimensions of an "
             "operand transposed without copying it."),
        reftable(["a", "b", "result", "Used for"], [
            ["<code>[m, k]</code>", "<code>[k, n]</code>", "<code>[m, n]</code>", "a dense layer on a batch"],
            ["<code>[B, m, k]</code>", "<code>[B, k, n]</code>", "<code>[B, m, n]</code>", "attention: one product per batch item"],
            ["<code>[..., k]</code>", "<code>[k, n]</code>", "<code>[..., n]</code>", "a shared weight over sequences <code>[N, T, k]</code>"],
        ], caption="Table 3.7 — MatMul shapes (with transposeA / transposeB applied first)"),
        mex("products and transposes", None,
            """
            var w = Tensor.From(new float[,] { { 1, 0 }, { 0, 1 }, { 1, 1 } });   // [3, 2]
            Console.WriteLine(m.MatMul(w));                       // [2,3] x [3,2] -> [2,2]
            Console.WriteLine(m.MatMul(m, transposeB: true));     // m · mᵀ -> [2,2]
            var batch = Tensor.Ones([4, 2, 3]);
            Console.WriteLine(string.Join(",", batch.MatMul(w).Shape.ToArray()));
            """,
            out="""
            Tensor(shape=[2, 2], device=cpu)
            [[4.0000, 5.0000]
             [10.0000, 11.0000]]
            Tensor(shape=[2, 2], device=cpu)
            [[14.0000, 32.0000]
             [32.0000, 77.0000]]
            4,2,2
            """),
        h2("3.8 Changing shape"),
        reftable(["Method", "Does", "Copies?"], [
            ["<code>Reshape(params int[] shape)</code>", "Same data, new shape; one dimension may be −1 (inferred)", "no (shares data)"],
            ["<code>Flatten(startDim = 1)</code>", "Merges dimensions from <code>startDim</code> on: <code>[N, C, H, W]</code> → <code>[N, C·H·W]</code>", "no"],
            ["<code>Permute(params int[] dims)</code>", "Reorders dimensions", "yes"],
            ["<code>Transpose(d0 = −2, d1 = −1)</code>", "Swaps two dimensions", "yes"],
            ["<code>Narrow(dim, start, length)</code>", "A slice along one dimension", "yes"],
            ["<code>Tensor.Concat(list, dim)</code>", "Joins along an existing dimension", "yes"],
            ["<code>Tensor.Stack(list, dim)</code>", "Joins equal shapes along a new dimension", "yes"],
        ], caption="Table 3.8 — Shape operations (all differentiable)"),
        mex("reshaping, slicing and joining", None,
            """
            Console.WriteLine(m.Reshape(3, -1));                 // [3, 2]
            Console.WriteLine(m.Transpose());                    // [3, 2], columns become rows
            Console.WriteLine(m.Narrow(1, 1, 2));                // columns 1..2
            Console.WriteLine(Tensor.Concat([m, m], dim: 0));    // [4, 3]
            Console.WriteLine(string.Join(",", Tensor.Stack([m, m, m, m], dim: 1).Shape.ToArray()));
            Console.WriteLine(string.Join(",", Tensor.Zeros([8, 3, 28, 28]).Flatten().Shape.ToArray()));
            """,
            out="""
            Tensor(shape=[3, 2], device=cpu)
            [[1.0000, 2.0000]
             [3.0000, 4.0000]
             [5.0000, 6.0000]]
            Tensor(shape=[3, 2], device=cpu)
            [[1.0000, 4.0000]
             [2.0000, 5.0000]
             [3.0000, 6.0000]]
            Tensor(shape=[2, 2], device=cpu)
            [[2.0000, 3.0000]
             [5.0000, 6.0000]]
            Tensor(shape=[4, 3], device=cpu)
            [[1.0000, 2.0000, 3.0000]
             [4.0000, 5.0000, 6.0000]
             [1.0000, 2.0000, 3.0000]
             [4.0000, 5.0000, 6.0000]]
            2,4,3
            8,2352
            """),
        trap("Reshape is not Transpose",
             "<p><code>m.Reshape(3, 2)</code> keeps the row-major order of the values (1 2 | 3 4 | 5 6); "
             "<code>m.Transpose()</code> reorders them (1 4 | 2 5 | 3 6). Use Reshape to regroup, Transpose or "
             "Permute to swap the meaning of dimensions (e.g. <code>[N, T, F]</code> → <code>[T, N, F]</code>).</p>"),
        h2("3.9 Moving, copying and detaching"),
        reftable(["Method", "Result"], [
            ["<code>To(device)</code>", "The same tensor if it is already there, otherwise a copy on <code>device</code>"],
            ["<code>Clone()</code>", "A copy on the same device, not connected to the gradient graph"],
            ["<code>Detach()</code>", "Shares the data but is cut off from the gradient graph (" + ch("autograd") + ")"],
        ], caption="Table 3.9 — Copies and views"),
        h2("3.10 Putting it together: scoring a batch by hand"),
        para("A classifier's last step multiplies features by a weight matrix, adds a bias, turns the scores into "
             "probabilities and picks the most likely class. With tensors alone, that is one line each. The GPU "
             "version differs only in where the tensors are created."),
        cpugpu("scoring three samples against three classes",
               """
               Device.Default = Device.Cpu;
               using var scope = new TensorScope();
               var x = Tensor.From(new float[,] { { 1.0f, 0.2f }, { 0.1f, 0.9f }, { 0.4f, 0.5f } });
               var weights = Tensor.From(new float[,] { { 2, -1, 0 }, { -1, 2, 0.5f } });
               var bias = Tensor.From([0f, 0f, 0.3f]);

               var probabilities = (x.MatMul(weights) + bias).Softmax();
               Console.WriteLine(probabilities);
               Console.WriteLine(probabilities.ArgMax());
               """,
               """
               Device.Default = Device.Cuda();       // the only change
               // ... identical lines ...
               """,
               "Output (identical on both devices, except that the header reads <code>device=cuda:0</code>):"),
        output("""
            Tensor(shape=[3, 3], device=cpu)
            [[0.7478, 0.0678, 0.1844]
             [0.0614, 0.6768, 0.2618]
             [0.2752, 0.3715, 0.3533]]
            Tensor(shape=[3], device=cpu)
            [0.0000, 1.0000, 1.0000]
            """),
        para("A <code>Linear</code> layer (" + ch("dense") + ") is exactly <code>x.MatMul(W) + b</code> with "
             "<code>W</code> and <code>b</code> created as trainable parameters, and "
             "<code>SparseCrossEntropy</code> (" + ch("losses") + ") uses <code>LogSoftmax</code> on the same scores."),
        practice([
            (1, "Create a tensor of shape <code>[4, 5]</code> holding the numbers 0 to 19 and print its second row.",
             "<code>Tensor.From([.. Enumerable.Range(0, 20).Select(i =&gt; (float)i)], [4, 5]).Narrow(0, 1, 1)</code> "
             "prints 5, 6, 7, 8, 9."),
            (1, "Given scores of shape <code>[N, K]</code>, compute the probability of the predicted class for each row.",
             "Take <code>p = scores.Softmax()</code>; the answer per row is the largest value of p. Read it on the host "
             "(<code>p.ToArray2D()</code>) and take each row's maximum, or use <code>(p * Tensor.OneHot(p.ArgMax(), K)).Sum(1)</code>."),
            (2, "Center the columns of a <code>[N, F]</code> matrix <code>x</code> (subtract each column's mean) using only tensor operations.",
             "<code>x + (-1f * x.Mean(0))</code>: the mean has shape <code>[F]</code>, and <code>+</code> broadcasts it over the rows."),
            (2, "Turn a batch of images <code>[N, 28, 28]</code> into the <code>[N, 1, 28, 28]</code> shape a convolution expects, and then into <code>[N, 784]</code> for a dense layer.",
             "<code>images.Reshape(-1, 1, 28, 28)</code>, then <code>.Flatten()</code> (or <code>Reshape(-1, 784)</code>). "
             "Neither copies data."),
            (3, "Compute the pairwise dot products of 1,000 vectors of length 64 on the CPU and on the GPU, and compare the time.",
             "Put the vectors in <code>v</code> of shape <code>[1000, 64]</code> and compute <code>v.MatMul(v, transposeB: true)</code> "
             "(shape <code>[1000, 1000]</code>). Time it as in " + ch("devices") + ", with a warm-up and "
             "<code>Synchronize()</code>."),
        ], PART),
        footer("Tensor", "Shape", "Rank", "Row-major", "Broadcasting", "Scalar", "Softmax", "ArgMax",
               "One-hot", "Matrix product", "Normal distribution", "Reshape", "Float32"),
    )
