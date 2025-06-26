from alphabet_pyramid import PyramidDrawer


def test_pyramid_drawer():
    pd = PyramidDrawer()

    ans = pd.draw("a", 1)
    assert ans == "a\n"

    ans = pd.draw("a", 2)
    assert ans == " a\naaa\n"
