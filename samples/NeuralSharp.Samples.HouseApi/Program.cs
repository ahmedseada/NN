// House-price Web API in a few lines: the inference engine from NeuralSharp.AspNetCore serves the package saved by the
// HousePrices sample (network, weights and both scalers in one .nsm file).
//
//   dotnet run -c Release --project samples/NeuralSharp.Samples.HousePrices                 # train; writes models/house-price.nsm
//   dotnet run -c Release --project samples/NeuralSharp.Samples.HouseApi -- --Package <path to house-price.nsm>
//   curl localhost:5090/predict/house-price -H 'Content-Type: application/json' \
//        -d '{"area":2100,"bedrooms":4,"bathrooms":2,"age":15,"distanceKm":9.5,"quality":7,"garageSpaces":2,"hasPool":0,"lotSqft":6500}'
//   curl localhost:5090/predict/house-price/batch -H 'Content-Type: application/json' -d '[{...}, {...}]'
//   curl localhost:5090/status

using NeuralSharp.AspNetCore;
using NeuralSharp.Samples.HouseApi;

var builder = WebApplication.CreateBuilder(args);
string package = builder.Configuration["Package"]
    ?? Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "NeuralSharp.Samples.HousePrices", "bin", "Release", "net10.0", "models", "house-price.nsm");

builder.Services.AddNeuralSharp()
    .AddPredictor<House, PriceEstimate>("house-price", package, p => p
        .Input<House>(h => [h.Area, h.Bedrooms, h.Bathrooms, h.Age, h.DistanceKm, h.Quality, h.GarageSpaces, h.HasPool, h.LotSqft])
        .Output(v => new PriceEstimate(MathF.Round(v[0], 0)))
        .Batching(maxBatch: 256, maxWait: TimeSpan.FromMilliseconds(2)));

var app = builder.Build();
app.MapPredictor<House, PriceEstimate>("/predict/house-price", "house-price");
app.MapNeuralSharpStatus("/status");
app.Run(builder.Configuration["Urls"] is null ? "http://localhost:5090" : null);

namespace NeuralSharp.Samples.HouseApi
{
    /// <summary>A house to price: the nine features of the HousePrices dataset, in its column order.</summary>
    public sealed record House(float Area, float Bedrooms, float Bathrooms, float Age, float DistanceKm, float Quality,
        float GarageSpaces, float HasPool, float LotSqft);

    /// <summary>The model's answer.</summary>
    public sealed record PriceEstimate(float Price);
}
