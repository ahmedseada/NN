namespace Shop.Tests;

public class PriceCalculatorHiddenTests
{
    [Theory]
    [InlineData(20.00, 15, 17.00)]
    [InlineData(19.99, 10, 17.99)]
    [InlineData(10.00, 0, 10.00)]
    [InlineData(10.00, 100, 0.00)]
    [InlineData(0.99, 33, 0.66)]
    public void AppliesPercentDiscounts(decimal price, decimal percent, decimal expected) =>
        Assert.Equal(new Money(expected, "EUR"), PriceCalculator.ApplyDiscount(new Money(price, "EUR"), percent));

    [Fact]
    public void RejectsDiscountsOutsideZeroToHundred()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => PriceCalculator.ApplyDiscount(new Money(1m, "EUR"), -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => PriceCalculator.ApplyDiscount(new Money(1m, "EUR"), 101));
    }
}
