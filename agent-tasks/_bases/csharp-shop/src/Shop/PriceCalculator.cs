namespace Shop;

public static class PriceCalculator
{
    /// <summary>The price after a discount of <paramref name="percent"/> (0–100), rounded to cents.</summary>
    public static Money ApplyDiscount(Money price, decimal percent)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(percent);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(percent, 100m);
        return price.Multiply(1m - (percent / 100m)).Round();
    }

    /// <summary>The price with VAT at <paramref name="rate"/> (for example 0.21), rounded to cents.</summary>
    public static Money WithTax(Money price, decimal rate) => price.Multiply(1m + rate).Round();
}
