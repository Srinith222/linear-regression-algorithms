import math

from linear_regression import LinearRegression


def test_fit_linear_relationship():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    model = LinearRegression().fit(x, y)

    assert abs(model.slope - 2) < 1e-9
    assert abs(model.intercept - 0) < 1e-9


def test_predict_value():
    x = [1, 2, 3, 4]
    y = [1, 3, 5, 7]

    model = LinearRegression().fit(x, y)

    assert abs(model.predict(5) - 9) < 1e-9


def test_cost_is_non_negative():
    x = [1, 2, 3, 4]
    y = [2, 4, 6, 8]

    model = LinearRegression().fit(x, y)

    assert model.cost(x, y) >= 0


def test_raises_on_identical_x_values():
    x = [5, 5, 5]
    y = [1, 2, 3]

    model = LinearRegression()

    try:
        model.fit(x, y)
        assert False, "Expected ValueError"
    except ValueError:
        pass
