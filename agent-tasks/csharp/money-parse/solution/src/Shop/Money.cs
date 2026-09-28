using System.Globalization;

namespace Shop;

/// <summary>An amount in one currency (ISO code such as EUR or USD).</summary>
public readonly record struct Money(decimal Amount, string Currency)
{
    public static Money Zero(string currency) => new(0m, currency);

    public static Money Parse(string text) =>
        TryParse(text, out var money) ? money : throw new FormatException($"'{text}' is not an amount such as \"EUR 12.50\".");

    public static bool TryParse(string? text, out Money money)
    {
        money = default;
        var parts = text?.Split(' ');
        if (parts is not [var currency, var amount] || currency.Length != 3 || !currency.All(char.IsAsciiLetterUpper)
            || !decimal.TryParse(amount, NumberStyles.AllowLeadingSign | NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out var value))
        {
            return false;
        }

        money = new Money(value, currency);
        return true;
    }

    public Money Add(Money other)
    {
        EnsureSameCurrency(other);
        return this with { Amount = Amount + other.Amount };
    }

    public Money Subtract(Money other)
    {
        EnsureSameCurrency(other);
        return this with { Amount = Amount - other.Amount };
    }

    public Money Multiply(decimal factor) => this with { Amount = Amount * factor };

    public Money Round() => this with { Amount = Math.Round(Amount, 2, MidpointRounding.AwayFromZero) };

    public override string ToString() => $"{Currency} {Amount.ToString("0.00", CultureInfo.InvariantCulture)}";

    private void EnsureSameCurrency(Money other)
    {
        if (other.Currency != Currency)
        {
            throw new InvalidOperationException($"Cannot combine {Currency} and {other.Currency}.");
        }
    }
}
