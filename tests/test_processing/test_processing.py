import pytest

from issue_classifier.processing.processing import encoding_model, processing, split


def test_encoding_model_correct(fake_model, train,val,test):
    tr_emb, val_emb, tst_emb = encoding_model(fake_model,train,val,test)
    assert len(tr_emb) == len(train)
    assert len(val_emb) == len(tst_emb)
    assert len(tst_emb) == len(test)

def test_encoding_model_wrong_model(wrong_model,train,val,test):
    with pytest.raises(ValueError):
        encoding_model(wrong_model,train,val,test)

def test_processing_correct(tmp_path,parquet_aux):
    path = tmp_path / "aux.parquet"
    X, y = processing(path=path)
    assert X.columns.tolist() == ['text']
    assert y.tolist() == [1,1,0,2,3]

def test_processing_unknown_label(tmp_path,parquet_unknown_label):
    path = tmp_path / "aux.parquet"
    with pytest.raises(ValueError):
        processing(path=path)

def test_split_equal_split(X_aux,y_aux):
    train,val,test, y_train, y_val, y_test = split(X_aux,y_aux)
    assert 1 in y_train.values
    assert 1 in y_val.values
    assert 1 in y_test.values
    assert len(train)/len(X_aux) == pytest.approx(0.7, 0.05)
    assert len(val) == 6
    assert len(test) == 6

def test_split_alineation(X_aux,y_aux):
    train,val,test, y_train, y_val, y_test = split(X_aux,y_aux)
    assert train.index.equals(y_train.index)
    assert val.index.equals(y_val.index)
    assert test.index.equals(y_test.index)