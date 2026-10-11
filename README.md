# Linear Regression Algorithms

This repository contains implementations and examples for linear regression algorithms.

## Contents

- `linear_regression.py`: Core linear regression implementation
- `examples.py`: Example usage
- `tests/test_linear_regression.py`: Unit tests

## Quick start

```bash
python examples.py
pytest -q
```

## What is included?

- Simple linear regression using ordinary least squares (OLS)
- Prediction function
- Cost calculation
- Training from sample data

## Example

```python
from linear_regression import LinearRegression

X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

model = LinearRegression()
model.fit(X, Y)
print(model.predict(6))
```
