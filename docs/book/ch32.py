"""Chapter 32 — Image Classification with CNNs."""
from gen import *

PART = "VI"


def build():
    return page(
        chapter_open(
            "cnn",
            "This project teaches a convolutional network to recognize four shapes (circle, square, triangle, cross) "
            "in small noisy grey images, drawn at random positions and sizes. It is the template for any image "
            "classifier: product photos, defects on a production line, medical scans, handwritten digits. The code is "
            "the repository's <code>NeuralSharp.Samples.Images</code>; this chapter explains each part and how to "
            "feed it real image files.",
            "Task type: <b>image classification</b>. Input <code>[N, C, H, W]</code>; model Conv2d/BatchNorm/ReLU/MaxPool blocks + dense head.",
            "Data: <code>Dataset.FromClassLabels(pixels, labels, K).WithFeatureShape(1, 16, 16)</code>.",
            "Result: 100% test accuracy after 12 epochs (8 s on the book's 4-core CPU).",
            "Convolutions are where the GPU shines; use <code>--cuda</code> and larger batches for real image sizes.",
            "Scale pixels to [0, 1]; keep images the same size; use NCHW order.",
        ),
        h2("32.1 The model"),
        snippet("""
            const int Size = 16;
            string[] shapes = ["circle", "square", "triangle", "cross"];
            var init = new Random(3);
            using var model = new Sequential
            {
                new Conv2d(1, 16, kernelSize: 3, padding: 1, device: device, random: init), new BatchNorm(16, device: device), new ReLU(),
                new MaxPool2d(2),                                                      // 16x16 -> 8x8
                new Conv2d(16, 32, kernelSize: 3, padding: 1, device: device, random: init), new BatchNorm(32, device: device), new ReLU(),
                new MaxPool2d(2),                                                      // 8x8 -> 4x4
                new Flatten(),
                new Linear(32 * 4 * 4, 64, device: device, random: init), new ReLU(), new Dropout(0.2f, init),
                new Linear(64, shapes.Length, device: device, random: init),
            };
            """, caption="A two-block CNN (37,988 parameters)"),
        para("Two convolution blocks detect strokes and then corners and curves; each pooling halves the image. "
             "The dense head combines the 32 feature maps of 4×4 into a decision. " + ch("conv") + " explains every "
             "layer and the shape arithmetic."),
        h2("32.2 Data"),
        snippet("""
            // pixels: float[count, 16 * 16] in [0, 1], labels: int[count] in 0..3
            var data = Dataset.FromClassLabels(pixels, labels, shapes.Length, shapes)
                              .WithFeatureShape(1, Size, Size);          // batches come out as [N, 1, 16, 16]
            """, caption="Turning pixel rows into an image dataset"),
        para("The sample draws its images procedurally (outline shapes with random size, position, brightness and noise), "
             "4,000 for training and 800 for testing, so it needs no downloads. The same two lines take any set of "
             "equally sized images."),
        h2("32.3 Training and results"),
        snippet("""
            int epochs = 12;
            using var optimizer = new AdamW(model.Parameters(), learningRate: 0.003f);
            var trainer = new Trainer(model, optimizer, (logits, targets) => Losses.CrossEntropy(logits, targets))
            {
                Metrics = { Metric.Accuracy },
                Scheduler = new CosineAnnealing(optimizer, epochs),
            };
            trainer.Fit(new DataLoader(train, 64, shuffle: true, device: device, seed: 4), epochs,
                        validation: new DataLoader(test, 400, device: device));
            """),
        output("""
            4000 training and 800 test images of 16x16 pixels, 4 classes

            Training on cpu (CPU (4 threads, 8-wide SIMD)) | 4,000 samples, 800 validation | batch 64, 63 steps/epoch | AdamW lr=0.003 | 37,988 parameters | 4 CPU threads
            Epoch  1/12  loss 0.438070  accuracy 0.8375  val_loss 0.053383  val_accuracy 0.9900  1498.1 ms  2,670 samples/s  *
            Epoch  2/12  loss 0.051849  accuracy 0.9835  val_loss 0.016776  val_accuracy 0.9962  676.3 ms  5,914 samples/s  *
            Epoch  4/12  loss 0.009389  accuracy 0.9990  val_loss 0.005124  val_accuracy 1.0000  586.3 ms  6,822 samples/s  *
            Epoch  8/12  loss 0.003615  accuracy 0.9992  val_loss 0.002073  val_accuracy 0.9988  618.7 ms  6,465 samples/s  *
            Epoch 12/12  loss 0.001750  accuracy 1.0000  val_loss 0.000860  val_accuracy 1.0000  644.6 ms  6,205 samples/s  *
            Finished 12 epochs in 8.25 s | best epoch 12 loss 0.000860

            Test accuracy: 100.0 %
            """, caption="Training output (selected epochs)"),
        output("""
            Image 1: predicted circle (100 %), actually circle
                    ++###++
                  +#+     +##
                 +#         ##
                 ##          #
                 +#+        ##
                  +#++   ++#+
                     ++++++
            """, caption="The sample prints a few test images as ASCII art with the model's verdict"),
        cpugpu("training and predicting",
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Images -- --cpu
               dotnet run -c Release --project samples/NeuralSharp.Samples.Images -- --cpu --predict --input "circle,cross"
               """,
               """
               dotnet run -c Release --project samples/NeuralSharp.Samples.Images -- --cuda --batch-size 256
               // convolutions are large regular matrix products: the GPU's best case
               """),
        h2("32.4 Using your own image files"),
        para("NeuralSharp has no image decoder, so reading PNG or JPEG files needs a small helper. Two dependency-free "
             "routes: convert images to PGM (a trivial format) with any image tool and read them as the OCR sample does "
             "(" + ch("ocr") + "), or use an imaging library you already have. On Windows, "
             "<code>System.Drawing</code> works; cross-platform libraries such as ImageSharp or SkiaSharp do too. "
             "Whatever reads the file, the steps into NeuralSharp are the same:"),
        deriv("From files to a dataset", [
            "Resize every image to the same size (e.g. 64×64) and convert to grey (C = 1) or keep RGB (C = 3).",
            "Write pixels in <b>NCHW</b> order: all red values row by row, then all green, then all blue, each divided by 255.",
            "Collect the rows into <code>float[count, C·H·W]</code> and the labels (e.g. from folder names) into <code>int[]</code>.",
            "<code>Dataset.FromClassLabels(pixels, labels, K, classNames).WithFeatureShape(C, H, W)</code>; split; train as above.",
            "Save the class names with the weights so predictions can be turned back into labels.",
        ]),
        snippet("""
            // pixel buffer in HWC order (as most decoders return it) -> NCHW row for NeuralSharp
            static void ToChw(ReadOnlySpan<byte> hwc, int height, int width, Span<float> chw)
            {
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                        for (int c = 0; c < 3; c++)
                            chw[c * height * width + y * width + x] = hwc[(y * width + x) * 3 + c] / 255f;
            }
            """, caption="The one conversion every image pipeline needs"),
        reftable(["Image size / data", "Suggested model"], [
            ["16–32 px, a few classes", "2 blocks (16, 32 channels), dense head: this chapter"],
            ["28×28 digits or characters", "2 blocks (32, 64), dense head: " + ch("ocr")],
            ["64×64 photos, 10+ classes", "4 blocks (32, 64, 128, 128), <code>GlobalAveragePool2d</code>, <code>Linear(128, K)</code>"],
            ["Few images per class (under ~100)", "Augment (flips, small shifts, brightness), heavy dropout; consider fine-tuning (" + ch("finetune") + ")"],
        ], caption="Table 32.1 — Sizing a CNN"),
        trap("images of different sizes in one batch",
             "<p>A batch is one tensor, so every image must have the same C, H and W. Resize (or crop and pad) when loading.</p>"),
        practice([
            (1, "What input shape does the model expect for a batch of 32 of the sample's images?",
             "<code>[32, 1, 16, 16]</code>."),
            (1, "Add a fifth shape class (for example a diamond). What changes in the model?",
             "Only the last layer: <code>Linear(64, 5)</code> (and the class name list)."),
            (2, "Double the image size to 32×32. Which layer's size must change and to what?",
             "After two poolings the maps are 8×8, so the first dense layer becomes <code>Linear(32 * 8 * 8, 64)</code>; or add a "
             "third block to return to 4×4."),
            (2, "Add simple data augmentation: horizontally flip half of the training images each epoch.",
             "Build a flipped copy of each training image (reverse each pixel row) and add both versions to the dataset, or "
             "generate a new randomly flipped dataset per epoch and call <code>Fit</code> one epoch at a time."),
            (3, "Build a folder-per-class image classifier for 64×64 RGB photos.",
             "Enumerate subfolders as classes; load, resize to 64×64 and convert each file to CHW floats with a helper like "
             "<code>ToChw</code>; <code>FromClassLabels(...).WithFeatureShape(3, 64, 64)</code>; a 4-block CNN with global average "
             "pooling; train on the GPU with batch 128; save weights plus class names."),
        ], PART),
        footer("CNN", "Convolution", "Feature map", "NCHW", "Data augmentation", "Pooling", "Image classification"),
    )
