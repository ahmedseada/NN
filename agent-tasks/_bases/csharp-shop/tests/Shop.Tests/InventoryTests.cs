namespace Shop.Tests;

public class InventoryTests
{
    [Fact]
    public void AddsAndRemovesStock()
    {
        var inventory = new Inventory();
        inventory.AddStock("MUG-1", 5);
        inventory.Remove("mug-1", 2);
        Assert.Equal(3, inventory.Quantity("MUG-1"));
    }

    [Fact]
    public void RefusesToRemoveMoreThanAvailable()
    {
        var inventory = new Inventory();
        inventory.AddStock("MUG-1", 1);
        Assert.Throws<InvalidOperationException>(() => inventory.Remove("MUG-1", 2));
    }
}
