from topaz.torch import set_num_threads

def test_set_num_threads():
    assert set_num_threads(0) == 0
    assert set_num_threads(1) == 1
    assert set_num_threads(-1) > 0