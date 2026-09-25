"""Chapter 40 — Fine-Tuning, Freezing and Multi-Output Models."""
from gen import *

PART = "VII"

TRANSFER = """
    // Body: the part of the network that learns features; heads: task-specific last layers.
    static Sequential Body(Random r) => new()
    {
        new Linear(8, 64, random: r), new Tanh(),
        new Linear(64, 64, random: r), new Tanh(),
    };

    // 1. pretrain body + head A on task A (5,000 samples), then keep the body
    using var body = Body(new Random(1));
    using var headA = new Linear(64, 1, random: new Random(2));
    using (var modelA = new Sequential { body, headA })
    {
        using var opt = new Adam(modelA.Parameters(), 3e-3f);
        new Trainer(modelA, opt, Losses.MeanSquaredError).Fit(new DataLoader(taskA, 64, shuffle: true, seed: 1), 40);
        body.Save("body.weights");                       // a Sequential saves on its own
    }

    // 2a. frozen body + new head on the small task (40 samples)
    using var frozenBody = Body(new Random(5));
    frozenBody.Load("body.weights");
    foreach (var p in frozenBody.Parameters()) p.RequiresGrad = false;     // no gradients for the body
    using var headB = new Linear(64, 1, random: new Random(6));
    using var modelB = new Sequential { frozenBody, headB };
    using var optB = new Adam(headB.Parameters(), 3e-3f);                    // only the head is optimized
    new Trainer(modelB, optB, Losses.MeanSquaredError).Fit(new DataLoader(smallB, 32, shuffle: true, seed: 1), 200);

    // 2b. fine-tune everything: load the body, keep RequiresGrad, optimize all at a 10x smaller rate
    //     using var opt = new Adam(model.Parameters(), 3e-4f);
"""

MULTI = """
    // Targets per row: [value, value > 0 ? 1 : 0]; the model has one shared body and a 2-output head.
    using var multi = new Sequential { Body(new Random(7)), new Linear(64, 2, random: new Random(8)) };

    static Tensor TwoTaskLoss(Tensor output, Tensor target) =>
        Losses.MeanSquaredError(output.Narrow(1, 0, 1), target.Narrow(1, 0, 1))                    // regression head
        + 0.5f * Losses.BinaryCrossEntropyWithLogits(output.Narrow(1, 1, 1), target.Narrow(1, 1, 1));   // yes/no head

    var valueMae = new Metric("value_mae", (p, t) => Losses.MeanAbsoluteError(p.Narrow(1, 0, 1), t.Narrow(1, 0, 1)));
    var signAcc = new Metric("sign_acc", (p, t) => Metric.BinaryAccuracy(0f).BatchMean(p.Narrow(1, 1, 1), t.Narrow(1, 1, 1)));

    using var opt = new Adam(multi.Parameters(), 3e-3f);
    var trainer = new Trainer(multi, opt, TwoTaskLoss) { Metrics = { valueMae, signAcc } };
    trainer.Fit(new DataLoader(train, 64, shuffle: true, seed: 1), 30);
    var result = trainer.Evaluate(new DataLoader(test, 1000));
    Console.WriteLine($"two heads: value MAE {result.Metrics["value_mae"]:F4}, sign accuracy {result.Metrics["sign_acc"]:P1}");
"""


def build():
    return page(
        chapter_open(
            "finetune",
            "Models rarely start from nothing in practice. A network trained on a large task learns features that "
            "help related tasks with little data (transfer learning); a model can be adjusted to new data without "
            "retraining it from scratch (fine-tuning); and one network can predict several things at once "
            "(multi-output). All three come down to three tools you already have: saving and loading parts of a "
            "model, choosing which parameters the optimizer updates, and building a loss from several pieces.",
            "Save and load any sub-module: a <code>Sequential</code> body is a module with its own weights file.",
            "Freeze: <code>p.RequiresGrad = false</code> on the body's parameters and give only the head to the optimizer.",
            "Fine-tune: train everything with a learning rate about 10× smaller than the original.",
            "Measured: with 40 samples of a related task, a frozen pretrained body cut the test error 4× (0.060 → 0.016).",
            "Multi-output: one head with several outputs, split with <code>Narrow</code>, and a weighted sum of losses.",
        ),
        h2("40.1 Transfer learning, measured"),
        para("The experiment uses 8 inputs that drive 4 hidden features. Task A (5,000 samples) is one combination "
             "of those features. Task B, with only 40 samples, uses the same features with different weights; task C "
             "needs a product of two features that task A never required. Each small task is trained three ways."),
        snippet(TRANSFER, caption="Pretraining, freezing and fine-tuning (data generation omitted)"),
        output("""
            task B (shares features)      test MSE: scratch 0.0601   frozen body 0.0156   fine-tuned 0.0389
            task C (needs new features)   test MSE: scratch 0.2073   frozen body 0.1681   fine-tuned 0.1697
            """, caption="Test error on 2,000 unseen samples (targets have a variance of about 0.9)"),
        para("When the new task uses the features the body already learned (B), freezing the body and fitting only a "
             "65-parameter head is four times better than training all 4,801 parameters from 40 samples. When it needs new "
             "features (C), the pretrained body helps less; with more data for task C, training from scratch or "
             "fine-tuning would overtake it. In an earlier run with 100 samples of a task like C, training from scratch "
             "won outright (0.049 against 0.151 frozen): transfer pays when data is scarce and the tasks are related."),
        reftable(["Situation", "Strategy"], [
            ["Very little data, closely related task", "Freeze the body, train a new head"],
            ["Moderate data, related task", "Train the head first (frozen body), then unfreeze and fine-tune all with lr/10"],
            ["Plenty of data or an unrelated task", "Train from scratch (or fine-tune everything)"],
            ["Same task, new data arriving (drift)", "Fine-tune the existing model on recent data at a low rate; validate on recent data"],
        ], caption="Table 40.1 — Choosing a strategy"),
        cpugpu("pretrain on the GPU, adapt on the CPU",
               """
               // on a laptop: adapt the pretrained body to a small local dataset
               using var body = Body(new Random(5));                  // Device.Default = Cpu
               body.Load("body.weights");
               """,
               """
               // on a GPU server: pretrain on the large dataset
               Device.Default = Device.Cuda();
               // ... train body + head A ...
               body.Save("body.weights");                             // device-independent file
               """),
        trap("forgetting that frozen BatchNorm still updates",
             "<p>Setting <code>RequiresGrad = false</code> stops gradient updates, but a <code>BatchNorm</code> layer in "
             "training mode still updates its running statistics from the new data. To freeze it completely, put the "
             "body in evaluation mode (<code>body.Eval()</code>) while training the head.</p>"),
        h2("40.2 One model, several outputs"),
        para("Related predictions can share a body: a price and whether the house sells within a month, a class and a "
             "bounding box, tomorrow's demand for five products. The last layer outputs all values side by side; the loss "
             "splits them with <code>Narrow</code> and adds the parts, weighted so that none dominates."),
        snippet(MULTI, caption="A regression head and a yes/no head sharing one body"),
        output("""
            two heads: value MAE 0.0308, sign accuracy 98.8 %
            """),
        reftable(["Outputs", "Head", "Loss"], [
            ["Several numbers", "<code>Linear(h, K)</code>", "<code>MeanSquaredError</code> on all K at once (scale the targets alike)"],
            ["Number + yes/no", "<code>Linear(h, 2)</code>", "MSE on column 0 + w · BCE-with-logits on column 1"],
            ["Class + number", "<code>Linear(h, K + 1)</code>", "<code>CrossEntropy</code> on columns 0…K−1 + w · MSE on column K"],
            ["Several yes/no labels", "<code>Linear(h, K)</code>", "<code>BinaryCrossEntropyWithLogits</code> on all K"],
        ], caption="Table 40.2 — Multi-output recipes"),
        honestbox("Weighting the parts",
                  "<p>The loss weights (0.5 above) balance how much each task pulls on the shared body. Start by making "
                  "each part contribute similar amounts at the beginning of training (compare their initial values), "
                  "then adjust the weight of the task you care about most. Report each task's own metric, as the "
                  "custom metrics above do.</p>"),
        practice([
            (1, "How do you freeze all but the last layer of a <code>Sequential</code> named <code>model</code>?",
             "<code>foreach (var p in model.Parameters().Except(model[model.Count - 1].Parameters())) p.RequiresGrad = false;</code> "
             "and create the optimizer with <code>model[model.Count - 1].Parameters()</code>."),
            (1, "Why fine-tune with a smaller learning rate than the original training?",
             "The weights are already good; large steps would destroy the learned features before the new data can refine them."),
            (2, "Save only the head of a model and load it onto a different body of the same shape.",
             "Call <code>head.Save(path)</code> on the <code>Linear</code> (any module can be saved) and <code>newHead.Load(path)</code>; "
             "combine with <code>new Sequential { body, newHead }</code>."),
            (2, "Build a model that predicts a class (3 classes) and a confidence score in [0, 1] from the same features.",
             "Head <code>Linear(h, 4)</code>; loss = <code>CrossEntropy(output.Narrow(1, 0, 3), oneHot)</code> + w · "
             "<code>BinaryCrossEntropyWithLogits(output.Narrow(1, 3, 1), score)</code>, with targets packed as 4 columns."),
            (3, "Fine-tune the house-price model on 200 new sales from a different city.",
             "Load the model and scalers; keep the scalers (the input distribution may shift, so check it); train all layers "
             "with Adam at 2e-4 for a few epochs with early stopping on a held-out part of the new sales; compare MAE on "
             "the new city before and after, and on the old test set to check nothing important was forgotten."),
        ], PART),
        footer("Transfer learning", "Fine-tuning", "Freezing", "Body and head", "Multi-output model", "Multi-task loss"),
    )
