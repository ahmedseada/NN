namespace Shop.Tests;

public class InventoryReservationHiddenTests
{
    private static Inventory WithMugs(int count)
    {
        var inventory = new Inventory();
        inventory.AddStock("MUG-1", count);
        return inventory;
    }

    [Fact]
    public void ReservingHoldsStockWithoutRemovingIt()
    {
        var inventory = WithMugs(5);
        Assert.True(inventory.TryReserve("mug-1", 3));
        Assert.Equal(5, inventory.Quantity("MUG-1"));
        Assert.Equal(2, inventory.Available("MUG-1"));
    }

    [Fact]
    public void CannotReserveMoreThanAvailable()
    {
        var inventory = WithMugs(5);
        Assert.True(inventory.TryReserve("MUG-1", 4));
        Assert.False(inventory.TryReserve("MUG-1", 2));
        Assert.Equal(1, inventory.Available("MUG-1"));
        Assert.False(inventory.TryReserve("PAN-9", 1));
    }

    [Fact]
    public void ReleasingGivesStockBack()
    {
        var inventory = WithMugs(5);
        inventory.TryReserve("MUG-1", 3);
        inventory.Release("MUG-1", 2);
        Assert.Equal(4, inventory.Available("MUG-1"));
        Assert.Throws<InvalidOperationException>(() => inventory.Release("MUG-1", 2));
    }

    [Fact]
    public void RemoveLeavesHeldStockAlone()
    {
        var inventory = WithMugs(5);
        inventory.TryReserve("MUG-1", 4);
        Assert.Throws<InvalidOperationException>(() => inventory.Remove("MUG-1", 2));
        inventory.Remove("MUG-1", 1);
        Assert.Equal(4, inventory.Quantity("MUG-1"));
        Assert.Equal(0, inventory.Available("MUG-1"));
    }
}
