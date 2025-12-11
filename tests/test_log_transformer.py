import pandas as pd
import numpy as np
import pytest
from d100project.preprocessing._log_transformer import LogTransformer

@pytest.mark.parametrize(
        "data,columns,expected",
        [
            (
                pd.DataFrame({
                    'A': [1, 2, 3, 4, 5],
                    'B': [10, 20, 30, 40, 50]
                }),
                ['A', 'B'],
                pd.DataFrame({
                    'A': np.log([1, 2, 3, 4, 5]),
                    'B': np.log([10, 20, 30, 40, 50])
                })
            ),
            (
                pd.DataFrame({
                    'A': [-1, 0, 1, 2],
                    'B': [5, 15, 25, 35]
                }),
                ['A'],
                pd.DataFrame({
                    'A': np.log([-1 + 1.000001, 0 + 1.000001, 1 + 1.000001, 2 + 1.000001]),
                    'B': [5, 15, 25, 35]
                })
            ),
            (
                pd.DataFrame({
                    'A': [0.5, 1.5, 2.5],
                    'B': [-10, -5, 0]
                }),
                ['B'],
                pd.DataFrame({
                    'A': [0.5, 1.5, 2.5],
                    'B': np.log([-10 + 10.000001, -5 + 10.000001, 0 + 10.000001])
                })
            ),
        ]
    )
def test_log_transformer(data, columns, expected):
    transformer = LogTransformer(variables=columns)
    transformer.fit(data)
    transformed_data = transformer.log_transform(data)
    pd.testing.assert_frame_equal(transformed_data, expected)