import pandas as pd
import pytest
from d100project.preprocessing._one_hot_encode import ListOneHotEncoder

@pytest.mark.parametrize(
    "data,columns,expected",
    [
        ( 
            pd.DataFrame({
                'features': [['A', 'B'], ['B', 'C'], ['A'], []],
                'values': [1, 2, 3, 4]
            }),
            ['features'],
            pd.DataFrame({
                'values': [1, 2, 3, 4],
                'features__A': [1, 0, 1, 0],
                'features__B': [1, 1, 0, 0],
                'features__C': [0, 1, 0, 0]
            })
        ),
        (
            pd.DataFrame({
                'tags': [['x', 'y'], ['y'], ['z', 'x'], ['x', 'y', 'z']],
                'amount': [10, 20, 30, 40]
            }),
            ['tags'],
            pd.DataFrame({
                'amount': [10, 20, 30, 40],
                'tags__x': [1, 0, 1, 1],
                'tags__y': [1, 1, 0, 1],
                'tags__z': [0, 0, 1, 1]
            })
        ),
        (
            pd.DataFrame({
                'categories': [[], ['D'], ['E', 'F'], ['D', 'E']],
                'score': [5, 15, 25, 35]
            }),
            ['categories'],
            pd.DataFrame({
                'score': [5, 15, 25, 35],
                'categories__D': [0, 1, 0, 1],
                'categories__E': [0, 0, 1, 1],
                'categories__F': [0, 0, 1, 0]
            })
        ),
    ]
)
def test_list_one_hot_encoder(data, columns, expected):
    encoder = ListOneHotEncoder(columns=columns)
    encoder.fit(data)
    transformed_data = encoder.transform(data)
    pd.testing.assert_frame_equal(transformed_data, expected)
