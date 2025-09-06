
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

np.random.seed(42)

n = 200
tv     = np.random.gamma(2.0, 150, n)
radio  = np.random.gamma(2.0, 60,  n)
social = np.random.gamma(2.0, 40,  n)
price  = np.random.normal(20, 3.0, n)

sales = (30 + 0.05*tv + 0.09*radio + 0.10*social - 2.0*price
         + np.random.normal(0, 8, n))

df = pd.DataFrame({"Sales": sales, "TV": tv, "Radio": radio, "Social": social, "Price": price})

X = df[["TV", "Radio", "Social", "Price"]]
y = df["Sales"]

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression().fit(X_tr, y_tr)

yhat_tr = model.predict(X_tr)
yhat_te = model.predict(X_te)

def metrics(name, yt, yp):
    r2   = r2_score(yt, yp)
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    print(f"{name:>10} | R2={r2:.3f}  MAE={mae:.2f}  RMSE={rmse:.2f}")


print("\n=== Mini Regression Metrics ===")
metrics("Train", y_tr, yhat_tr)
metrics("Test",  y_te, yhat_te)

tv_line = LinearRegression().fit(df[["TV"]], df["Sales"])
tv_grid = np.linspace(df["TV"].min(), df["TV"].max(), 200).reshape(-1,1)
sales_line = tv_line.predict(tv_grid)

plt.figure(figsize=(5.6,4.0))
plt.scatter(df["TV"], df["Sales"], s=12, alpha=0.7)
plt.plot(tv_grid, sales_line, lw=2)  # tek değişkenli fit
plt.xlabel("TV"); plt.ylabel("Sales"); plt.title("Sales vs TV (quick view)")
plt.tight_layout(); plt.savefig("scatter_tv_sales.png", dpi=140); plt.close()

lims = [min(y_te.min(), yhat_te.min()), max(y_te.max(), yhat_te.max())]
plt.figure(figsize=(5.2,5.2))
plt.scatter(y_te, yhat_te, s=14, alpha=0.8)
plt.plot(lims, lims, 'k--', lw=1)
plt.xlabel("Actual Sales"); plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted (Test)")
plt.tight_layout(); plt.savefig("actual_vs_pred.png", dpi=140); plt.close()

coef = pd.Series(model.coef_, index=X.columns).sort_values()
print("\nCoefficients:\n", coef)
print("\nSaved plots: scatter_tv_sales.png, actual_vs_pred.png")
