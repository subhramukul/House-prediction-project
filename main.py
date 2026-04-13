import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

data = pd.read_csv("data.csv")

print("=" * 45)
print("       DATASET PREVIEW")
print("=" * 45)
print(data.to_string(index=False))
print(f"\nTotal rows : {len(data)}")
print(f"Columns    : {list(data.columns)}")

X = data[['Size', 'Rooms', 'Location']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing  samples : {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

print("\n" + "=" * 45)
print("       MODEL COEFFICIENTS")
print("=" * 45)
print(f"Intercept  : {model.intercept_:,.2f}")
print(f"Size       : {model.coef_[0]:,.2f}  (per sq ft)")
print(f"Rooms      : {model.coef_[1]:,.2f}  (per room)")
print(f"Location   : {model.coef_[2]:,.2f}  (per level)")

predictions = model.predict(X_test)

print("\n" + "=" * 45)
print("       ACTUAL vs PREDICTED")
print("=" * 45)
print(f"{'Actual':>12}  {'Predicted':>12}  {'Error':>10}")
print("-" * 40)
for actual, pred in zip(y_test.values, predictions):
    err = actual - pred
    print(f"${actual:>10,.0f}  ${pred:>10,.0f}  ${err:>+9,.0f}")

r2        = r2_score(y_test, predictions)
mae       = mean_absolute_error(y_test, predictions)
residuals = y_test.values - predictions

print("\n" + "=" * 45)
print("       MODEL PERFORMANCE")
print("=" * 45)
print(f"R² Score : {r2:.4f}  ({r2*100:.1f}% accurate)")
print(f"MAE      : ${mae:,.0f}")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35)

BLUE  = '#378ADD'
RED   = '#E24B4A'
GREEN = '#1D9E75'
AMBER = '#EF9F27'
BG    = '#0f1117'
CARD  = '#1a1d27'
TEXT  = '#e0e0e0'
MUTED = '#888780'

def style_ax(ax, title):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_title(title, color=TEXT, fontsize=11,
                 fontweight='bold', pad=10)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2d3a')
    ax.grid(color='#2a2d3a', linestyle='--', linewidth=0.5)

cards = [
    ("R² Accuracy",     f"{r2:.3f}",          BLUE),
    ("Test Samples",    str(len(y_test)),      GREEN),
    ("Avg Error (MAE)", f"${mae/1000:.1f}k",  AMBER),
]
for i, (label, value, color) in enumerate(cards):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor(CARD)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)
    ax.text(0.5, 0.62, value, ha='center', va='center',
            fontsize=30, fontweight='bold', color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha='center', va='center',
            fontsize=10, color=MUTED, transform=ax.transAxes)

ax1 = fig.add_subplot(gs[1, 0])
style_ax(ax1, "Actual vs Predicted Price")
ax1.scatter(y_test / 1000, predictions / 1000,
            color=BLUE, s=120, zorder=5, edgecolors='white',
            linewidths=0.5)

for actual, pred in zip(y_test.values, predictions):
    ax1.annotate(f"${actual/1000:.0f}k",
                 (actual/1000, pred/1000),
                 textcoords="offset points",
                 xytext=(5, 5), fontsize=7, color=MUTED)

mn = min(y_test.min(), predictions.min()) / 1000
mx = max(y_test.max(), predictions.max()) / 1000
ax1.plot([mn, mx], [mn, mx], color=RED,
         linestyle='--', linewidth=1.5, label='Perfect fit')
ax1.set_xlabel("Actual Price ($k)")
ax1.set_ylabel("Predicted Price ($k)")
ax1.legend(fontsize=8, labelcolor=MUTED,
           facecolor=CARD, edgecolor='#2a2d3a')

ax2 = fig.add_subplot(gs[1, 1])
style_ax(ax2, "Feature Coefficients")
features = ['Size\n(per sq ft)', 'Rooms', 'Location']
coefs    = model.coef_
bars = ax2.bar(features, coefs,
               color=[BLUE, GREEN, AMBER],
               alpha=0.85, width=0.5, edgecolor='none')
for bar, val in zip(bars, coefs):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 200,
             f'${val:,.0f}',
             ha='center', va='bottom',
             fontsize=9, color=TEXT)
ax2.set_ylabel("Impact on Price ($)")

ax3 = fig.add_subplot(gs[1, 2])
style_ax(ax3, "Size vs Price (Full Dataset)")
sc = ax3.scatter(data['Size'], data['Price'] / 1000,
                 c=data['Rooms'], cmap='Blues',
                 s=80, edgecolors='white', linewidths=0.4,
                 zorder=5)
cbar = plt.colorbar(sc, ax=ax3)
cbar.set_label('Rooms', color=MUTED, fontsize=9)
cbar.ax.yaxis.set_tick_params(color=MUTED)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED, fontsize=8)
ax3.set_xlabel("Size (sq ft)")
ax3.set_ylabel("Price ($k)")

plt.suptitle(
    "House Price Prediction — Linear Regression Dashboard",
    color=TEXT, fontsize=13, fontweight='bold', y=1.02)

plt.savefig("ml_dashboard.png", dpi=150,
            bbox_inches='tight', facecolor=BG)
print("\nDashboard saved: ml_dashboard.png")
plt.show()

print("\n" + "=" * 45)
print("       PREDICT YOUR HOUSE PRICE")
print("=" * 45)
size_inp     = float(input("Enter house size (sq ft) : "))
rooms_inp    = int(input("Enter number of rooms    : "))
location_inp = int(input("Enter location (1/2/3)   : "))

result = model.predict([[size_inp, rooms_inp, location_inp]])
print(f"\n>>> Predicted House Price: ${result[0]:,.0f} <<<")