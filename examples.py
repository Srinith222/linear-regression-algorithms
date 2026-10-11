from linear_regression import LinearRegression


def main():
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    model = LinearRegression().fit(x, y)
    print(f"Slope: {model.slope}")
    print(f"Intercept: {model.intercept}")
    print(f"Prediction for x=6: {model.predict(6)}")
    print(f"Cost: {model.cost(x, y)}")


if __name__ == "__main__":
    main()
