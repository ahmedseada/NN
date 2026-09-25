"""Chapter 10 — Convolution and Pooling."""
from gen import *

PART = "II"


def conv_svg():
    w, h = 470, 150
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    # input 6x6 with a 3x3 window
    for r in range(6):
        for c in range(6):
            on = r < 3 and 1 <= c < 4
            p.append(f'<rect x="{20 + c * 18}" y="{24 + r * 18}" width="17" height="17" fill="{"#0f6b5c" if on else "#e6f2ef"}" stroke="#b7d8d0" stroke-width="0.6"/>')
    p.append(svg_text(74, 16, "input 6×6", 8.6, "#56606a"))
    # kernel
    for r in range(3):
        for c in range(3):
            p.append(f'<rect x="{170 + c * 18}" y="{51 + r * 18}" width="17" height="17" fill="#fff4e2" stroke="#a15c00" stroke-width="0.6"/>')
    p.append(svg_text(197, 43, "3×3 filter", 8.6, "#a15c00"))
    p.append(svg_text(155, 81, "×", 12, "#56606a"))
    p.append(svg_text(245, 81, "=", 12, "#56606a"))
    # output 4x4
    for r in range(4):
        for c in range(4):
            on = r == 0 and c == 1
            p.append(f'<rect x="{265 + c * 18}" y="{42 + r * 18}" width="17" height="17" fill="{"#0f6b5c" if on else "#e6f2ef"}" stroke="#b7d8d0" stroke-width="0.6"/>')
    p.append(svg_text(301, 34, "output 4×4", 8.6, "#56606a"))
    p.append(svg_text(400, 60, "one output value =", 7.6, "#56606a"))
    p.append(svg_text(400, 72, "sum of window × filter", 7.6, "#56606a"))
    p.append(svg_text(400, 84, "+ bias; the filter", 7.6, "#56606a"))
    p.append(svg_text(400, 96, "slides over every", 7.6, "#56606a"))
    p.append(svg_text(400, 108, "position", 7.6, "#56606a"))
    p.append(svg_text(235, 144, "one filter shown; Conv2d(C, F) learns F filters, each spanning all C input channels", 7.4, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "conv",
            "Images have structure a dense layer ignores: nearby pixels belong together, and a pattern means the same "
            "thing wherever it appears. A convolution slides small learned filters over the image, so it needs far "
            "fewer parameters than a dense layer and recognizes a pattern in any position. Pooling then shrinks the "
            "image so later layers see larger regions. This chapter covers <code>Conv2d</code>, the pooling layers "
            "and <code>Flatten</code>, how to compute every shape, and how a complete CNN is assembled.",
            "Images are <code>[N, C, H, W]</code>: batch, channels (1 grey, 3 RGB), height, width.",
            "<code>Conv2d(inC, outC, kernelSize, stride, padding)</code>: output size = (H + 2·padding − kernel) / stride + 1.",
            "<code>padding: kernelSize / 2</code> with stride 1 keeps the size; stride 2 or <code>MaxPool2d(2)</code> halves it.",
            "End with <code>Flatten</code> + <code>Linear</code>, or <code>GlobalAveragePool2d</code> + <code>Linear</code>.",
            "Convolutions are the clearest case for the GPU: large regular matrix products.",
        ),
        h2("10.1 What a convolution computes"),
        diagram("Figure 10.1 — A 3×3 filter over a 6×6 image (stride 1, no padding)", conv_svg(),
                "Every output value comes from the same filter applied to a different window."),
        para("A <code>Conv2d(inChannels, outChannels, k)</code> learns <code>outChannels</code> filters, each of "
             "size <code>inChannels × k × k</code>, plus one bias per filter. Each filter produces one output channel "
             "(a feature map). Early layers learn edges and colour blobs; deeper layers combine them into parts and "
             "objects (glossary <b>Feature map</b>)."),
        reftable(["Argument", "Default", "Meaning"], [
            ["<code>inChannels</code>", "—", "Channels of the input (1, 3, or the previous layer's filters)"],
            ["<code>outChannels</code>", "—", "Number of filters = output channels"],
            ["<code>kernelSize</code>", "—", "Filter height and width (3 is the usual choice)"],
            ["<code>stride</code>", "1", "Step between filter positions; 2 halves the size"],
            ["<code>padding</code>", "0", "Zeros added on each border; <code>k / 2</code> keeps the size at stride 1"],
            ["<code>bias</code>, <code>device</code>, <code>random</code>", "true, Default, Shared", "As for <code>Linear</code>; weights start He-uniform"],
        ], caption="Table 10.1 — new Conv2d(inChannels, outChannels, kernelSize, stride, padding, bias, device, random)"),
        defbox("Output size",
               "<p>For input height H, kernel k, stride s and padding p, the output height is " + brk("⌊") + "(H + 2p − k) / s" + brk("⌋") + " + 1 "
               "(the same for the width). A kernel that does not fit throws <i>A k×k kernel does not fit …</i>.</p>"),
        mex("the formula on a 32×32 RGB image", None,
            """
            using var scope = new TensorScope();
            var image = Tensor.Zeros([1, 3, 32, 32]);
            foreach (var (k, s, p) in new[] { (3, 1, 0), (3, 1, 1), (5, 1, 2), (3, 2, 1), (7, 2, 3) })
            {
                using var conv = new Conv2d(3, 8, k, stride: s, padding: p);
                Console.WriteLine($"k={k} s={s} p={p}: [{string.Join(", ", conv.Forward(image).Shape.ToArray())}]");
            }
            """,
            out="""
            k=3 s=1 p=0: [1, 8, 30, 30]
            k=3 s=1 p=1: [1, 8, 32, 32]
            k=5 s=1 p=2: [1, 8, 32, 32]
            k=3 s=2 p=1: [1, 8, 16, 16]
            k=7 s=2 p=3: [1, 8, 16, 16]
            """),
        h2("10.2 Pooling and flattening"),
        reftable(["Layer", "Shape", "Use"], [
            ["<code>MaxPool2d(k, stride = k, padding = 0)</code>", "<code>[N, C, H, W]</code> → <code>[N, C, H/k, W/k]</code>", "Keeps the strongest response in each window; halves the size with k = 2"],
            ["<code>GlobalAveragePool2d()</code>", "<code>[N, C, H, W]</code> → <code>[N, C]</code>", "Averages each channel over the whole image; a light head for deep CNNs"],
            ["<code>Flatten()</code>", "<code>[N, ...]</code> → <code>[N, features]</code>", "Turns feature maps into a vector for <code>Linear</code>"],
        ], caption="Table 10.2 — Pooling and flattening"),
        para("<code>MaxPool2d(3, stride: 2, padding: 1)</code> on a <code>[1, 8, 32, 32]</code> input gives "
             "<code>[1, 8, 16, 16]</code> (overlapping windows), and <code>GlobalAveragePool2d</code> turns "
             "<code>[4, 64, 7, 7]</code> into <code>[4, 64]</code>. Padded positions never win a max-pool window."),
        h2("10.3 A complete CNN, shape by shape"),
        mex("a digit classifier for 28×28 grey images",
            "Two convolution blocks (convolution, BatchNorm, ReLU, pooling) followed by a small dense head. Tracing a "
            "batch through the layers one by one is the fastest way to get every size right.",
            """
            var init = new Random(1);
            using var cnn = new Sequential
            {
                new Conv2d(1, 16, kernelSize: 3, padding: 1, random: init), new BatchNorm(16), new ReLU(),
                new MaxPool2d(2),
                new Conv2d(16, 32, kernelSize: 3, padding: 1, random: init), new BatchNorm(32), new ReLU(),
                new MaxPool2d(2),
                new Flatten(),
                new Linear(32 * 7 * 7, 64, random: init), new ReLU(), new Dropout(0.2f, init),
                new Linear(64, 10, random: init),
            };

            using var scope = new TensorScope();
            var x = Tensor.Zeros([8, 1, 28, 28]);
            foreach (var layer in cnn)
            {
                x = layer.Forward(x);
                Console.WriteLine($"{layer,-45} -> [{string.Join(", ", x.Shape.ToArray())}]");
            }
            """,
            out="""
            Conv2d(1 -> 16, 3x3, stride 1, padding 1)     -> [8, 16, 28, 28]
            BatchNorm(16)                                 -> [8, 16, 28, 28]
            ReLU                                          -> [8, 16, 28, 28]
            MaxPool2d(2x2, stride 2)                      -> [8, 16, 14, 14]
            Conv2d(16 -> 32, 3x3, stride 1, padding 1)    -> [8, 32, 14, 14]
            BatchNorm(32)                                 -> [8, 32, 14, 14]
            ReLU                                          -> [8, 32, 14, 14]
            MaxPool2d(2x2, stride 2)                      -> [8, 32, 7, 7]
            Flatten                                       -> [8, 1568]
            Linear(1568 -> 64)                            -> [8, 64]
            ReLU                                          -> [8, 64]
            Dropout(p=0.2)                                -> [8, 64]
            Linear(64 -> 10)                              -> [8, 10]
            """),
        para("<code>cnn.Summary()</code> reports 105,962 parameters. The convolutions hold only 4,800 of them "
             "(1·9·16 + 16 and 16·9·32 + 32); the first dense layer holds 100,416, which is typical. Replacing "
             "<code>Flatten</code> + <code>Linear(1568, 64)</code> with <code>GlobalAveragePool2d</code> + "
             "<code>Linear(32, 64)</code> cuts the model to 7,658 parameters, at some cost in accuracy on small images."),
        cpugpu("training a CNN (the loop is the same as for any model)",
               """
               Device.Default = Device.Cpu;
               ComputeResources.MaxCpuThreads = Environment.ProcessorCount;   // convolutions use every core
               using var cnn = BuildCnn();
               // images: [N, 1, 28, 28], labels: [N] class ids; loss: Losses.SparseCrossEntropy
               """,
               """
               Device.Default = Device.Cuda();
               using var cnn = BuildCnn();
               // same data and loss. A batch of 64-128 images keeps the GPU busy;
               // convolutions are where the GPU gains most over the CPU
               """,
               "The full training program, with data loading and evaluation, is " + ch("cnn") + "; OCR of characters is " + ch("ocr") + "."),
        h2("10.4 How it runs: im2col and one big matrix product"),
        para("Rather than looping over windows, <code>Conv2d</code> first copies every k×k×C window into one row of a "
             "large matrix (<i>im2col</i>, glossary <b>im2col</b>), then multiplies that matrix by the filters with the "
             "same optimized matrix product that <code>Linear</code> uses. This is why convolutions run efficiently on "
             "both backends, and why their memory use grows with k²: the unfolded matrix has N·OH·OW rows of "
             "C·k·k values. " + ch("cpu") + " and " + ch("cuda") + " show the kernels."),
        reftable(["Stage", "Shape (example: [8, 16, 14, 14], 3×3, 32 filters)"], [
            ["Input", "<code>[8, 16, 14, 14]</code>"],
            ["Unfolded windows (im2col)", "<code>[8·14·14, 16·3·3]</code> = <code>[1568, 144]</code>"],
            ["× filters transposed", "<code>[144, 32]</code>"],
            ["Product", "<code>[1568, 32]</code>"],
            ["Rearranged to NCHW", "<code>[8, 32, 14, 14]</code>"],
        ], caption="Table 10.3 — Inside one Conv2d forward pass"),
        trap("feeding NHWC images",
             "<p>Image libraries often give pixels as height × width × channels. <code>Conv2d</code> expects channels "
             "before height and width. Rearrange once when loading (e.g. <code>Permute(0, 3, 1, 2)</code> on "
             "<code>[N, H, W, C]</code>), or write the pixels into a <code>float[]</code> in NCHW order directly.</p>"),
        trap("forgetting to scale pixels",
             "<p>Raw byte values 0–255 make training slow and unstable. Divide by 255 (to [0, 1]) or standardize per "
             "channel before creating the tensor.</p>"),
        practice([
            (1, "What is the output shape of <code>Conv2d(3, 64, 5, stride: 1, padding: 0)</code> on <code>[16, 3, 64, 64]</code>?",
             "(64 + 0 − 5)/1 + 1 = 60, so <code>[16, 64, 60, 60]</code>."),
            (1, "How many parameters does <code>Conv2d(64, 128, 3)</code> have?",
             "128 filters × (64·3·3) weights + 128 biases = 73,728 + 128 = 73,856."),
            (2, "Design the layers that take 64×64 RGB images down to a 4×4 map with 64 channels before the head.",
             "Four halvings are needed (64 → 32 → 16 → 8 → 4): e.g. four blocks of <code>Conv2d(…, 3, padding: 1)</code>, "
             "<code>BatchNorm</code>, <code>ReLU</code>, <code>MaxPool2d(2)</code> with channels 3 → 16 → 32 → 64 → 64."),
            (2, "Convert a grey image stored as <code>byte[] pixels</code> (row-major, 28×28) into the input tensor for the CNN of Section 10.3.",
             "<code>Tensor.From(pixels.Select(b =&gt; b / 255f).ToArray(), [1, 1, 28, 28])</code>. For a batch, concatenate "
             "the float arrays and use shape <code>[N, 1, 28, 28]</code>."),
            (3, "Replace the dense head of the Section 10.3 CNN with global average pooling and compare parameter counts and accuracy on your data.",
             "Swap <code>Flatten</code>, <code>Linear(1568, 64)</code> for <code>GlobalAveragePool2d</code>, <code>Linear(32, 64)</code>. "
             "Parameters drop from 105,962 to 7,658; accuracy usually drops slightly on 28×28 inputs, which you can "
             "recover by adding a third convolution block with more channels."),
        ], PART),
        footer("Convolution", "Filter", "Feature map", "Channel", "Stride", "Padding", "Max pooling",
               "Global average pooling", "Flatten", "NCHW", "im2col", "CNN"),
    )
