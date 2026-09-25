"""Chapter 33 — OCR: Reading Characters."""
from gen import *

PART = "VI"


def pipeline_svg():
    w, h = 470, 90
    steps = [("page image", "PGM / pixels"), ("segment", "lines → characters"), ("normalize", "each to 20×20"),
             ("CNN", "one batch"), ("text", "ArgMax → letters")]
    p = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">']
    for i, (a, b) in enumerate(steps):
        x = 6 + i * 93
        p.append(f'<rect x="{x}" y="18" width="80" height="44" rx="5" fill="{"#0f6b5c" if a == "CNN" else "#e6f2ef"}" stroke="#0f6b5c"/>')
        color = "#ffffff" if a == "CNN" else "#0f6b5c"
        p.append(svg_text(x + 40, 37, a, 8.4, color))
        p.append(svg_text(x + 40, 51, b, 7.2, color))
        if i < 4:
            p.append(f'<line x1="{x + 80}" y1="40" x2="{x + 93}" y2="40" stroke="#56606a"/>')
    p.append(svg_text(235, 82, "only the CNN is learned; segmentation is classic image processing", 7.4, "#56606a"))
    p.append("</svg>")
    return "".join(p)


def build():
    return page(
        chapter_open(
            "ocr",
            "Optical character recognition turns an image of printed text into a string. This project, the "
            "repository's <code>NeuralSharp.Samples.Ocr</code>, trains a CNN on randomly distorted renderings of the "
            "36 characters 0–9 and A–Z, then reads whole lines: it cuts the image into characters, classifies all of "
            "them in one batch, and reassembles the text with spaces and line breaks. It also reads your own images "
            "in PGM format.",
            "Two parts: <b>segmentation</b> (plain C#: find lines, characters and word gaps) and <b>recognition</b> (a CNN classifier with 36 classes).",
            "Training data is generated: every character rendered with random size, slant, position, stroke weight, brightness and noise.",
            "Result: 100% on unseen character renderings; 99.85% character accuracy on 40 random text lines (39 exactly right).",
            "Inference batches all characters of a page into one <code>Predict</code> call.",
            "The recipe transfers to any \"cut into pieces, classify each piece\" problem.",
        ),
        diagram("Figure 33.1 — Reading a line", pipeline_svg(), "Segmentation finds the characters; the CNN names them."),
        h2("33.1 The recognizer"),
        snippet("""
            const int Cell = Renderer.Cell;                  // 20: characters are normalized to 20x20
            string alphabet = Font.Characters;               // "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            var init = new Random(1);
            using var model = new Sequential
            {
                new Conv2d(1, 32, kernelSize: 3, padding: 1, device: device, random: init), new BatchNorm(32, device: device), new ReLU(),
                new MaxPool2d(2),                                                      // 20x20 -> 10x10
                new Conv2d(32, 64, kernelSize: 3, padding: 1, device: device, random: init), new BatchNorm(64, device: device), new ReLU(),
                new MaxPool2d(2),                                                      // 10x10 -> 5x5
                new Flatten(),
                new Linear(64 * 5 * 5, 128, device: device, random: init), new ReLU(), new Dropout(0.3f, init),
                new Linear(128, alphabet.Length, device: device, random: init),
            };
            """, caption="CNN over 20×20 character images (228,580 parameters)"),
        snippet("""
            Dataset Characters(int perClass, int seed)
            {
                var rng = new Random(seed);
                int count = perClass * alphabet.Length;
                var pixels = new float[count, Cell * Cell];
                var labels = new int[count];
                var image = new float[Cell * Cell];
                for (int s = 0; s < count; s++)
                {
                    labels[s] = s % alphabet.Length;
                    Array.Clear(image);
                    Renderer.RenderCharacter(labels[s], image, rng);         // random distortions
                    for (int i = 0; i < image.Length; i++) pixels[s, i] = image[i];
                }
                return Dataset.FromClassLabels(pixels, labels, alphabet.Length, [.. alphabet.Select(c => c.ToString())])
                              .WithFeatureShape(1, Cell, Cell);
            }

            var train = Characters(perClass: 300, seed: 2);                   // 10,800 images
            var test = Characters(perClass: 60, seed: 3);                     // 2,160 unseen renderings
            """, caption="Generated training data"),
        h2("33.2 Training"),
        snippet("""
            int epochs = 10;
            using var optimizer = new AdamW(model.Parameters(), learningRate: 0.003f, weightDecay: 1e-4f);
            var trainer = new Trainer(model, optimizer,
                (logits, targets) => Losses.CrossEntropy(logits, targets, labelSmoothing: 0.05f))
            {
                Metrics = { Metric.Accuracy },
                Scheduler = new CosineAnnealing(optimizer, epochs, warmupEpochs: 1),
            };
            trainer.Fit(new DataLoader(train, 64, shuffle: true, device: device, seed: 4), epochs,
                        validation: new DataLoader(test, 500, device: device));
            """),
        output("""
            Rendered 10,800 training and 2,160 test characters in 1796 ms

            Training on cpu (CPU (4 threads, 8-wide SIMD)) | 10,800 samples, 2,160 validation | batch 64, 169 steps/epoch | AdamW lr=0.0015 | 228,580 parameters | 4 CPU threads
            Epoch  1/10  loss 1.688845  accuracy 0.6114  val_loss 0.570783  val_accuracy 0.9880  7915.9 ms  1,364 samples/s  *
            Epoch  2/10  loss 0.721386  accuracy 0.9500  val_loss 0.486671  val_accuracy 0.9977  6256.4 ms  1,726 samples/s  *
            Epoch  3/10  loss 0.621004  accuracy 0.9807  val_loss 0.464879  val_accuracy 1.0000  6015.5 ms  1,795 samples/s  *
            Epoch 10/10  loss 0.522551  accuracy 0.9961  val_loss 0.416508  val_accuracy 1.0000  6070.1 ms  1,779 samples/s  *
            Finished 10 epochs in 62.83 s | best epoch 10 loss 0.416508

            Character accuracy on unseen renderings: 100.00 %
            Read 40 random text lines: 99.85 % character accuracy, 39/40 lines exactly right
            """, caption="Training output on the CPU (selected epochs)"),
        para("Training accuracy stays below validation accuracy because dropout and the random distortions make the "
             "training images harder than the clean test renderings; the loss stays above 0 because of label smoothing. "
             "At 63 s on 4 CPU cores this is the first project in the book where the GPU makes a real difference: "
             "convolutions over 10,800 images per epoch."),
        h2("33.3 Reading a line"),
        snippet("""
            string Read(float[] pixels, int width, int height)
            {
                var glyphs = Segmenter.Segment(pixels, width, height);           // characters, with space/newline flags
                if (glyphs.Count == 0) return "";

                var batch = new float[glyphs.Count * Cell * Cell];
                for (int i = 0; i < glyphs.Count; i++)
                    glyphs[i].Pixels.CopyTo(batch, i * Cell * Cell);

                using var input = Tensor.From(batch, [glyphs.Count, 1, Cell, Cell], device);   // the whole page at once
                using var logits = model.Predict(input);
                using var best = logits.ArgMax();
                var classes = best.ToArray();

                var text = new System.Text.StringBuilder();
                for (int i = 0; i < glyphs.Count; i++)
                {
                    text.Append(glyphs[i].NewLineBefore ? "\\n" : glyphs[i].SpaceBefore ? " " : "");
                    text.Append(alphabet[(int)classes[i]]);
                }
                return text.ToString();
            }
            """, caption="Segmentation + batched recognition"),
        output("""
            Rendered line "HELLO WORLD 2026":

                  ##      +##   ###########   ##+           ##+             #######
                  ##+      ##   ###+##+###+   ##+           ##+           ###########
                  ##      +##   ##+           ##            ##+           ##      +##
                  ###########   #########     ##+           ##+           ##+      ##
                  ###########   #########     ##            ##+           ##      +##
                  ##+      ##   ##            ##+           ##            ##      +##
                  ##      +##   #####++#++#   ##+           ##+    + +    ##+  ++ ###
                  ##+     +##   ###########   ###########   ###########     #######
            Recognized: HELLO WORLD 2026
            """, caption="--predict --input \"HELLO WORLD 2026\" (preview cropped to the first five letters)"),
        cpugpu("commands",
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Ocr -- --cpu
               dotnet run -c Release --project samples/NeuralSharp.Samples.Ocr -- --cpu --predict --input "HELLO WORLD 2026"
               dotnet run -c Release --project samples/NeuralSharp.Samples.Ocr -- --cpu --predict --image page.pgm
               """,
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Ocr -- --cuda
               dotnet run -c Release --project samples/NeuralSharp.Samples.Ocr -- --cuda --predict --save-image line.pgm
               """,
               "Convert a PNG or JPEG to PGM with any image tool, e.g. <code>magick scan.png scan.pgm</code> (ImageMagick)."),
        h2("33.4 From the sample to real documents"),
        reftable(["Real-world issue", "Approach"], [
            ["Other fonts, lower case, punctuation", "Render training data from those fonts and characters (or collect labelled crops); enlarge the alphabet"],
            ["Touching or broken characters", "Improve segmentation, or recognize whole words with a sequence model reading the line left to right (" + ch("recurrent") + ")"],
            ["Skewed or noisy scans", "Deskew and threshold before segmenting; add rotation and noise to the training renderings"],
            ["Handwriting", "Collect labelled samples; larger CNN; heavy augmentation"],
            ["Confidence per character", "<code>Softmax()</code> of the logits; flag characters below e.g. 0.9 for review"],
        ], caption="Table 33.1 — Going further"),
        trap("recognizing characters one Predict call at a time",
             "<p>A page may contain thousands of characters. One call per character pays the call overhead (and on the GPU "
             "a synchronization) thousands of times; batch all glyphs into one tensor, as <code>Read</code> does.</p>"),
        practice([
            (1, "Why does validation accuracy exceed training accuracy in the log?",
             "Dropout is active and distortions are random during training; evaluation runs without dropout on clean test renderings."),
            (1, "How many classes does the recognizer have, and what would change to add lower-case letters?",
             "36. Adding a–z makes 62: extend the alphabet (and the renderer's font), and the last layer becomes <code>Linear(128, 62)</code>."),
            (2, "Report the three least confident characters of a recognized line.",
             "Apply <code>Softmax()</code> to the logits, take each row's maximum as confidence, and list the three glyphs with the lowest values."),
            (2, "Read a multi-line PGM page and print the lines separately.",
             "<code>Segmenter</code> already marks <code>NewLineBefore</code>; <code>Read</code> inserts line breaks, so split the result on newlines."),
            (3, "Recognize license plates from photos.",
             "Locate the plate (a separate detector or a fixed camera region), threshold and segment it like a text line, train the CNN on "
             "renderings of the plate font with perspective and blur augmentations, and batch-recognize the characters; validate on "
             "real labelled photos."),
        ], PART),
        footer("OCR", "Segmentation", "Glyph", "Data augmentation", "Synthetic data", "PGM"),
    )
