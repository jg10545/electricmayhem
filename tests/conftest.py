import os
import pytest



@pytest.fixture()
def test_png_1():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "fixtures", "testimg1.png")


@pytest.fixture()
def test_png_2():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "fixtures", "testimg2.png")