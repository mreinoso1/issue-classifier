import numpy as np
import pandas as pd
import pytest


class DummySentenceTransformer:
    def encode(self, text, **kwargs):
        if isinstance(text,list):
            if all(isinstance(item,str) for item in text):
                return np.ones(len(text))
        raise TypeError("Text has to be a list")

class WrongDummySentenceTransformer:
    def encode(self, text, **kwargs):
        if isinstance(text,list):
            if all(isinstance(item,str) for item in text):
                return np.ones(shape=2)
        raise TypeError("Text has to be a list")

@pytest.fixture
def fake_model():
    return DummySentenceTransformer()

@pytest.fixture
def wrong_model():
    return WrongDummySentenceTransformer()

@pytest.fixture
def train():
    aux = ["ROW1","ROW2","ROW3","ROW4"]
    train = pd.DataFrame()
    train["text"] = aux
    return train

@pytest.fixture
def val():
    aux = ["ROW1","ROW2","ROW3"]
    train = pd.DataFrame()
    train["text"] = aux
    return train
    
@pytest.fixture
def test():
    aux = ["ROW1","ROW2","ROW3"]
    train = pd.DataFrame()
    train["text"] = aux
    return train

@pytest.fixture
def parquet_aux(tmp_path):
    tmp_dict = {
        "title":
        [
            'Issue1','Issue2','Issue3','Issue4','Issue5'
        ],
        "body":
        [
            'body1','body2','body3','body4','body6'
        ],
        "label":
        [
            'bug','bug','feature','doc','question'
        ]
    }
    aux = pd.DataFrame(tmp_dict)
    path = tmp_path / "aux.parquet"
    aux.to_parquet(path=path)

@pytest.fixture
def parquet_unknown_label(tmp_path):
    tmp_dict = {
        "title":
        [
            'Issue1','Issue2','Issue3','Issue4','Issue5'
        ],
        "body":
        [
            'body1','body2','body3','body4','body6'
        ],
        "label":
        [
            'bug','Fix','Usage','doc','question'
        ]
    }
    aux = pd.DataFrame(tmp_dict)
    path = tmp_path / "aux.parquet"
    aux.to_parquet(path=path)

@pytest.fixture
def X_aux():
    aux = {
        "number": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
                    ,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40],
        "text": ["row1","row2","row3","row4","row5","row6","row7","row8","row9","row10",
                "row11","row12","row13","row14","row15","row16","row17","row18","row19","row20",
                "row21","row22","row23","row24","row25","row26","row27","row28","row29","row30",
                "row31","row32","row33","row34","row35","row36","row37","row38","row39","row40"],
        "repo": ["fastapi","fastapi","fastapi","fastapi","fastapi","scikit","scikit","scikit","scikit","scikit",
                "supabase","supabase","supabase","supabase","supabase","transformers","transformers","transformers",
                "transformers","transformers",
                "fastapi","fastapi","fastapi","fastapi","fastapi","scikit","scikit","scikit","scikit","scikit",
                "supabase","supabase","supabase","supabase","supabase","transformers","transformers","transformers",
                "transformers","transformers"],
        "__index_level_0__": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                            24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]
    }
    return pd.DataFrame(aux)

@pytest.fixture
def y_aux():
    aux = [0,0,3,0,0,1,0,2,1,0,1,1,1,2,2,2,2,3,3,3,0,0,3,0,0,0,0,2,0,0,3,3,1,2,2,2,2,3,3,3]
    return pd.Series(aux, name="label")

