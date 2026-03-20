
def run_model():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    import zipfile

    with zipfile.ZipFile("Sample - Superstore.csv.zip", 'r') as zip_ref:
        zip_ref.extractall()

    df = pd.read_csv("Sample - Superstore.csv", encoding='latin1')

    df = df[['Order Date','Sales']]
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df = df.sort_values('Order Date')

    df['Year'] = df['Order Date'].dt.year
    df['Month'] = df['Order Date'].dt.month
    df['Day'] = df['Order Date'].dt.day

    X = df[['Year','Month','Day']]
    y = df['Sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, df, mae, r2
