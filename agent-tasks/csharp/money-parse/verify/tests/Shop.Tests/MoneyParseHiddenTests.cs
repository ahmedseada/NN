using System.Globalization;

namespace Shop.Tests;

public class MoneyParseHiddenTests
{
    [Theory]
    [InlineData("EUR 12.50", 12.50, "EUR")]
    [InlineData("USD -3.00", -3.00, "USD")]
    public void ParsesWhatToStringWrites(string text, decimal amount, string currency) =>
        Assert.Equal(new Money(amount, currency), Money.Parse(text));

    [Theory]
    [InlineData("")]
    [InlineData("EUR")]
    [InlineData("eur 1.00")]
    [InlineData("EURO 1.00")]
    [InlineData("EUR 1,00")]
    [InlineData("EUR  1.00")]
    [InlineData("EUR 1.00 extra")]
    [InlineData("EUR abc")]
    public void RejectsOtherText(string text)
    {
        Assert.False(Money.TryParse(text, out _));
        Assert.Throws<FormatException>(() => Money.Parse(text));
    }

    [Fact]
    public void TryParseAcceptsNull() => Assert.False(Money.TryParse(null, out _));

    [Fact]
    public void RoundTripsUnderAnyCulture()
    {
        var previous = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");
            var money = new Money(1234.5m, "EUR");
            Assert.Equal(money, Money.Parse(money.ToString()));
        }
        finally
        {
            CultureInfo.CurrentCulture = previous;
        }
    }
}
