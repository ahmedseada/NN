namespace Shop.Tests;

public class MoneyTests
{
    [Fact]
    public void AddsAmountsInTheSameCurrency() =>
        Assert.Equal(new Money(12.75m, "EUR"), new Money(10m, "EUR").Add(new Money(2.75m, "EUR")));

    [Fact]
    public void RefusesToMixCurrencies() =>
        Assert.Throws<InvalidOperationException>(() => new Money(1m, "EUR").Add(new Money(1m, "USD")));

    [Theory]
    [InlineData(2.345, 2.35)]
    [InlineData(2.344, 2.34)]
    [InlineData(-2.345, -2.35)]
    public void RoundsToCentsAwayFromZero(decimal amount, decimal expected) =>
        Assert.Equal(expected, new Money(amount, "EUR").Round().Amount);

    [Fact]
    public void FormatsWithTheCurrencyCode() => Assert.Equal("EUR 5.50", new Money(5.5m, "EUR").ToString());
}
