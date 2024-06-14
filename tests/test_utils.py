from src.train.utils import load_data

def test_load_data():
    X, y = load_data()
    assert X.shape[0] == 150
    assert len(y) == 150
