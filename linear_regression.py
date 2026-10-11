import math
from typing import List


class LinearRegression:
    """Simple linear regression implementation using least squares."""

    def __init__(self):
        self.slope = 0.0
        self.intercept = 0.0

    def fit(self, x: List[float], y: List[float]) -> "LinearRegression":
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(x) < 2:
            raise ValueError("At least two data points are required")

        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            raise ValueError("x values must not all be identical")

        self.slope = numerator / denominator
        self.intercept = y_mean - self.slope * x_mean
        return self

    def predict(self, x: float) -> float:
        return self.intercept + self.slope * x

    def cost(self, x: List[float], y: List[float]) -> float:
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")

        total = 0.0
        for xi, yi in zip(x, y):
            prediction = self.predict(xi)
            total += (prediction - yi) ** 2
        return total / len(x)

    def __repr__(self) -> str:
        return f"LinearRegression(slope={self.slope}, intercept={self.intercept})"
