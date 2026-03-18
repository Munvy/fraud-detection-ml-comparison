import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, confusion_matrix
from sklearn.model_selection import train_test_split

# 1. Wczytanie i przygotowanie danych
# Plik creditcard.csv musi być w tym samym folderze co skrypt
data = pd.read_csv('creditcard.csv')

# Wybieramy cechy V1-V28 oraz kwotę transakcji
features = [f'V{i}' for i in range(1, 29)] + ['Amount']
X = data[features]
y = data.Class

# Podział na zbiór treningowy i walidacyjny (80/20)
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=1)

# Funkcja do sprawdzania wyników każdego modelu
def ocen_model(model, nazwa, X_wal, y_wal):
    predykcje = model.predict(X_wal)
    
    # Zamieniamy wyniki regresji na klasy 0/1 (próg 0.5)
    klasy_pred = [1 if x > 0.5 else 0 for x in predykcje]
    
    cm = confusion_matrix(y_wal, klasy_pred)
    mae = mean_absolute_error(y_wal, predykcje)

    print(f"\n--- WYNIKI DLA: {nazwa} ---")
    print(f"Błąd MAE: {mae:.6f}")
    print(f"Wykryte oszustwa: {cm[1][1]} / {cm[1][1] + cm[1][0]}")
    print(f"Fałszywe alarmy: {cm[0][1]}")
    return mae

# --- Model 1: Drzewo Decyzyjne ---
print("Trenuję Decision Tree...")
dt_model = DecisionTreeRegressor(random_state=1, max_leaf_nodes=100)
dt_model.fit(train_X, train_y)
ocen_model(dt_model, "Decision Tree", val_X, val_y)

# --- Model 2: Las Losowy (Random Forest) ---
print("Trenuję Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
rf_model.fit(train_X, train_y)
ocen_model(rf_model, "Random Forest", val_X, val_y)

# --- Model 3: XGBoost (z wagami klas) ---
print("Trenuję XGBoost...")
# Ustawiamy scale_pos_weight, żeby model lepiej wyłapywał rzadkie oszustwa
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    random_state=1,
    n_jobs=-1,
    scale_pos_weight=580
)
xgb_model.fit(train_X, train_y)
ocen_model(xgb_model, "XGBoost (Weighted)", val_X, val_y)
