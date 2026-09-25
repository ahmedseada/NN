"""Chapter 30 — Recommenders and Categorical Embeddings."""
from gen import *

PART = "V"

MF = """
    /// rating = mean + userBias + itemBias + userVector · itemVector;  input [N, 2] = (user id, item id)
    sealed class MatrixFactorization : Module
    {
        private readonly Embedding _users, _items, _userBias, _itemBias;
        private readonly float _mean;

        public MatrixFactorization(int users, int items, int dim, float mean, Random? random = null, Device? device = null)
        {
            _users = new Embedding(users, dim, device, random);
            _items = new Embedding(items, dim, device, random);
            _userBias = new Embedding(users, 1, device, random);
            _itemBias = new Embedding(items, 1, device, random);
            _mean = mean;
        }

        protected override Tensor ForwardCore(Tensor x)
        {
            var user = x.Narrow(1, 0, 1).Flatten(0);                       // [N] user ids
            var item = x.Narrow(1, 1, 1).Flatten(0);                       // [N] item ids
            var dot = (_users.Forward(user) * _items.Forward(item)).Sum(1, keepDim: true) * 0.1f;   // [N, 1]
            return dot + _userBias.Forward(user) * 0.1f + _itemBias.Forward(item) * 0.1f + _mean;
        }

        public override IEnumerable<Module> Children() => [_users, _items, _userBias, _itemBias];
    }
"""

TRAIN = """
    // ratings: (user, item) -> stars, e.g. loaded from a CSV with columns user,item,rating
    var (train, test) = Dataset.FromArrays(ids, ratings, ["user", "item"], ["rating"]).Split(0.9, seed: 1);
    float mean = train.Targets.ToArray().Average();

    using var model = new MatrixFactorization(Users, Items, dim: 16, mean, new Random(1));
    using var optimizer = new AdamW(model.Parameters(), 5e-3f, weightDecay: 1e-2f);
    var trainer = new Trainer(model, optimizer, Losses.MeanSquaredError)
    {
        Metrics = { Metric.RootMeanSquaredError },
        EarlyStoppingPatience = 5,
    };
    var history = trainer.Fit(new DataLoader(train, 256, shuffle: true, seed: 1), epochs: 100,
                              validation: new DataLoader(test, 2048));
    var best = history.Epochs[history.BestEpoch - 1];
    double baseline = Math.Sqrt(test.Targets.ToArray().Average(r => (r - mean) * (r - mean)));
    Console.WriteLine($"{history.Epochs.Count} epochs; test RMSE {best.ValidationMetrics!["rmse"]:F3} stars " +
                      $"(predicting the average: {baseline:F3})");

    // top 5 recommendations for user 7 among the movies they have not rated
    int user = 7;
    var unseen = Enumerable.Range(0, Items).Where(i => !rated.Contains((user, i))).ToArray();
    var query = new float[unseen.Length, 2];
    for (int k = 0; k < unseen.Length; k++) { query[k, 0] = user; query[k, 1] = unseen[k]; }
    float[,] scores = model.Predict(query);
    Console.WriteLine($"user {user}: top 5 of {unseen.Length} unseen movies");
    foreach (int k in Enumerable.Range(0, unseen.Length).OrderByDescending(k => scores[k, 0]).Take(5))
        Console.WriteLine($"  movie {unseen[k],3}: predicted {scores[k, 0]:F2}, " +
                          $"true taste {TrueRating(user, unseen[k]):F2}");   // TrueRating: from the synthetic generator
"""

TABULAR = """
    /// Numeric columns [promo, temperature] + embedded weekday (dim 3) and store (dim 8), then an MLP.
    /// Input rows: [promo, temperature, weekday id, store id].
    sealed class SalesModel : Module
    {
        private readonly Embedding _weekday, _store;
        private readonly Sequential _mlp;

        public SalesModel(int stores, Random? random = null, Device? device = null)
        {
            _weekday = new Embedding(7, 3, device, random);
            _store = new Embedding(stores, 8, device, random);
            _mlp = new Sequential
            {
                new Linear(2 + 3 + 8, 64, device: device, random: random), new ReLU(),
                new Linear(64, 64, device: device, random: random), new ReLU(),
                new Linear(64, 1, device: device, random: random),
            };
        }

        protected override Tensor ForwardCore(Tensor x)
        {
            var numeric = x.Narrow(1, 0, 2);                                   // [N, 2]
            var weekday = _weekday.Forward(x.Narrow(1, 2, 1).Flatten(0));     // [N, 3]
            var store = _store.Forward(x.Narrow(1, 3, 1).Flatten(0));         // [N, 8]
            return _mlp.Forward(Tensor.Concat([numeric, weekday, store], dim: 1));
        }

        public override IEnumerable<Module> Children() => [_weekday, _store, _mlp];
    }
"""


def build():
    return page(
        chapter_open(
            "recommender",
            "Ids are everywhere in business data: users, products, stores, countries, weekdays. Treating an id as a "
            "number (store 17 is \"more\" than store 3) is meaningless, and one-hot columns get huge. Embeddings "
            "(" + ch("embedding") + ") give each id a learned vector instead. This chapter builds two projects on that "
            "idea: a movie recommender that learns user and item vectors from ratings, and a sales model that mixes "
            "numeric columns with embedded categories. Both are custom composite modules.",
            "Recommender (matrix factorization): rating ≈ mean + user bias + item bias + user vector · item vector.",
            "Input rows hold ids as floats; <code>Narrow</code> + <code>Flatten(0)</code> extracts an id column for an <code>Embedding</code>.",
            "Result: test RMSE 0.431 stars against 0.946 for always predicting the average.",
            "Tabular data: embedding the store id cut the error from 27.3 to 1.5 units compared with feeding the id as a number.",
            "Recommending = scoring every unseen item for a user and taking the best.",
        ),
        h2("30.1 A recommender from ratings"),
        para("The data are (user, movie, stars) triples; most pairs are unrated. Matrix factorization "
             "(glossary <b>Matrix factorization</b>) gives every user and every movie a vector of 16 numbers, trained "
             "so that their dot product, plus a per-user and per-movie bias, reproduces the known ratings. Users with "
             "similar tastes end up with similar vectors, and a user's predicted rating for an unseen movie follows. "
             "The factor 0.1 scales down the embeddings' N(0, 1) starting values so the initial predictions are close "
             "to the mean rating."),
        snippet(MF, caption="The model: four embeddings and a dot product"),
        snippet(TRAIN, caption="Training and recommending (the synthetic ratings generator is omitted: 500 users, 300 movies, 30,000 ratings)"),
        output("""
            49 epochs; test RMSE 0.431 stars (predicting the average: 0.946)
            user 7: top 5 of 246 unseen movies
              movie  76: predicted 4.65, true taste 4.27
              movie 258: predicted 4.48, true taste 4.11
              movie  44: predicted 4.44, true taste 4.44
              movie 286: predicted 4.35, true taste 3.93
              movie  61: predicted 4.34, true taste 3.70
            """, caption="Output (\"true taste\" is available only because the data are synthetic)"),
        cpugpu("scale",
               """
               // 500 users x 300 movies trains in seconds on the CPU
               Device.Default = Device.Cpu;
               """,
               """
               // millions of ratings, 100k+ users: the GPU and large batches (4,096+)
               Device.Default = Device.Cuda();
               var loader = new DataLoader(train, 4096, shuffle: true);
               // scoring all items for a user is one Predict call on [items, 2]
               """),
        reftable(["Extension", "How"], [
            ["Implicit feedback (clicks, purchases, no stars)", "Targets 1 for interactions and 0 for sampled non-interactions; <code>BinaryCrossEntropyWithLogits</code>"],
            ["New users with no history (cold start)", "Add user features (age group, country) as extra embeddings/columns; recommend popular items until history exists"],
            ["Item side information (genre, price)", "Add item-feature embeddings to the item vector"],
            ["\"Customers who bought X also bought\"", "Nearest item vectors by cosine similarity (" + ch("embedding") + ")"],
            ["Serving", "Precompute all item vectors; a user's scores are one matrix product"],
        ], caption="Table 30.1 — Recommender extensions"),
        h2("30.2 Categorical columns in tabular models"),
        para("The second project predicts daily sales of 50 stores from a promotion flag, the temperature, the "
             "weekday and the store. Weekday and store are categories. The same data trains two models: an MLP that "
             "reads the four columns as numbers, and a model that embeds weekday and store."),
        snippet(TABULAR, caption="Mixing numeric columns and embeddings"),
        output("""
            ids as numbers  test MAE 27.27 units  (4,545 parameters)
            embeddings      test MAE 1.46 units  (5,542 parameters)
            """, caption="Both models trained with Adam, early stopping, 16,000 training rows"),
        para("With ids as numbers, the MLP would have to carve 50 arbitrary store levels out of one numeric input; "
             "with an embedding, each store simply gets its own learned vector. The same pattern handles any number of "
             "categorical columns: one <code>Embedding</code> per column, sized by Table 9.2 (" + ch("embedding") + "), "
             "concatenated with the scaled numeric columns."),
        deriv("Preparing categorical columns", [
            "Collect the distinct values of each categorical column in the training data and number them 0…K−1 "
            "(reserve one id for values not seen in training).",
            "Replace each value by its id when building the feature rows; keep the numeric columns scaled.",
            "Save the value-to-id maps (e.g. as JSON) together with the weights and the scaler: they are part of the model.",
        ]),
        trap("scaling the id columns",
             "<p>A <code>StandardScaler</code> fitted on all columns would turn ids into fractional numbers that no longer "
             "select embedding rows. Scale only the numeric columns (fit the scaler on them alone), and leave id columns "
             "as integers.</p>"),
        practice([
            (1, "How many parameters do the four embeddings of the recommender have for 500 users, 300 movies and dim 16?",
             "(500 + 300)·16 + (500 + 300)·1 = 12,800 + 800 = 13,600."),
            (1, "Why does the recommender add the mean rating as a constant?",
             "So the model starts near the right level and the embeddings only have to learn deviations from it."),
            (2, "Find the 3 movies most similar to movie 44 from the learned vectors.",
             "Read the item embedding table (<code>Weight.ToArray2D()</code>; expose the embedding from the module), compute the "
             "cosine similarity of row 44 with every other row, and take the three largest."),
            (2, "Add a country column (40 countries) to the sales model.",
             "Add <code>new Embedding(40, 4)</code>, a fifth input column with the country id, extract it with "
             "<code>x.Narrow(1, 4, 1).Flatten(0)</code>, and widen the first Linear to 2 + 3 + 8 + 4 inputs."),
            (3, "Turn the recommender into an implicit-feedback model for purchase data.",
             "Use purchased (user, item) pairs as positives with target 1, sample random unpurchased pairs as negatives "
             "with target 0 (e.g. 4 per positive), drop the mean term, train with <code>BinaryCrossEntropyWithLogits</code>, "
             "and rank unseen items by score; evaluate with the share of held-out purchases that appear in each user's top 10."),
        ], PART),
        footer("Recommender system", "Matrix factorization", "Embedding", "Categorical feature", "Cold start",
               "Implicit feedback", "Dot product"),
    )
