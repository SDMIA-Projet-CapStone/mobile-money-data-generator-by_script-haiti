"""
Simulateur MonCash / NatCash — Haïti  v3.0  (vectorisé)
Génération ~10× plus rapide grâce à NumPy vectorisé.
Auteur : Projet Détection Fraude Haiti
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
START_DATE   = datetime(2024, 1, 1)
TX_LIMIT_HTG = 50_000

DEPARTEMENTS = {
    "Ouest": 0.40, "Artibonite": 0.18, "Centre": 0.09,
    "Nord": 0.09,  "Nord-Est": 0.04,   "Nord-Ouest": 0.04,
    "Sud": 0.07,   "Sud-Est": 0.04,    "Grande-Anse": 0.03, "Nippes": 0.02,
}
DEPT_NAMES   = list(DEPARTEMENTS.keys())
DEPT_WEIGHTS = np.array(list(DEPARTEMENTS.values()))
DEPT_WEIGHTS /= DEPT_WEIGHTS.sum()

VILLES = {
    "Ouest":       ["Port-au-Prince","Pétion-Ville","Delmas","Carrefour"],
    "Artibonite":  ["Gonaïves","Saint-Marc"],
    "Centre":      ["Hinche","Mirebalais"],
    "Nord":        ["Cap-Haïtien","Fort-Liberté"],
    "Nord-Est":    ["Fort-Liberté","Ouanaminthe"],
    "Nord-Ouest":  ["Port-de-Paix"],
    "Sud":         ["Les Cayes"],
    "Sud-Est":     ["Jacmel"],
    "Grande-Anse": ["Jérémie"],
    "Nippes":      ["Miragoâne"],
}

OPERATORS  = ["MonCash", "NatCash"]
PLATFORMS  = ["USSD", "APP_ANDROID", "APP_IOS", "AGENT"]

_raw = [0.01,0.01,0.01,0.01,0.01,0.02,
        0.04,0.07,0.09,0.08,0.07,0.06,
        0.06,0.07,0.07,0.06,0.07,0.08,
        0.08,0.07,0.05,0.03,0.02,0.01]
HOUR_W = np.array(_raw)
HOUR_W /= HOUR_W.sum()

RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────
# CRÉATION VECTORISÉE DES UTILISATEURS
# ─────────────────────────────────────────────

def create_users(n: int = 6000) -> pd.DataFrame:
    """Génère tous les utilisateurs en blocs NumPy — 50× plus rapide."""
    profiles = RNG.choice(["LOW","MEDIUM","HIGH"], size=n, p=[0.65, 0.25, 0.10])

    avg_amount = np.where(profiles == "LOW",
                    RNG.integers(500, 5_001, n),
                    np.where(profiles == "MEDIUM",
                        RNG.integers(5_000, 25_001, n),
                        RNG.integers(25_000, 80_001, n)))

    balance = np.where(profiles == "LOW",
                RNG.integers(1_000, 20_001, n),
                np.where(profiles == "MEDIUM",
                    RNG.integers(10_000, 100_001, n),
                    RNG.integers(50_000, 500_001, n)))

    tx_per_day = np.where(profiles == "LOW",
                    RNG.integers(1, 4, n),
                    np.where(profiles == "MEDIUM",
                        RNG.integers(2, 7, n),
                        RNG.integers(3, 13, n)))

    dept_idx  = RNG.choice(len(DEPT_NAMES), size=n, p=DEPT_WEIGHTS)
    depts     = np.array(DEPT_NAMES)[dept_idx]
    villes    = np.array([random.choice(VILLES[d]) for d in depts])
    operators = RNG.choice(OPERATORS, size=n, p=[0.60, 0.40])
    platforms = RNG.choice(PLATFORMS, size=n, p=[0.40, 0.35, 0.15, 0.10])

    return pd.DataFrame({
        "user_id":          [f"U{i:07d}" for i in range(n)],
        "profile":          profiles,
        "avg_amount":       avg_amount.astype(float),
        "balance":          balance.astype(float),
        "tx_per_day":       tx_per_day,
        "account_age_days": RNG.integers(30, 3651, n),
        "device_id":        [f"DEV{x:05d}" for x in RNG.integers(1, 15_001, n)],
        "departement":      depts,
        "ville":            villes,
        "operator":         operators,
        "platform":         platforms,
        "is_agent":         (RNG.random(n) < 0.05).astype(int),
        "kyc_level":        RNG.choice([1,2,3], size=n, p=[0.50,0.35,0.15]),
    })


def create_recipients(n: int = 12_000) -> pd.DataFrame:
    dept_idx = RNG.choice(len(DEPT_NAMES), size=n, p=DEPT_WEIGHTS)
    return pd.DataFrame({
        "recipient_id":     [f"R{i:08d}" for i in range(n)],
        "account_age_days": RNG.integers(1, 3651, n),
        "is_known_mule":    (RNG.random(n) < 0.002).astype(int),
        "departement":      np.array(DEPT_NAMES)[dept_idx],
    })


# ─────────────────────────────────────────────
# GÉNÉRATION VECTORISÉE DES LÉGITIMES
# ─────────────────────────────────────────────

def gen_legit_batch(users_df: pd.DataFrame,
                    recipients_df: pd.DataFrame,
                    n: int) -> pd.DataFrame:
    """Génère n transactions légitimes en blocs vectorisés."""

    u_idx = RNG.integers(0, len(users_df), n)
    r_idx = RNG.integers(0, len(recipients_df), n)
    users = users_df.iloc[u_idx].reset_index(drop=True)
    recs  = recipients_df.iloc[r_idx].reset_index(drop=True)

    # Montants log-normaux centrés sur avg_amount
    noise   = RNG.normal(1.0, 0.4, n)
    amounts = np.abs(users["avg_amount"].values * noise)
    amounts = np.clip(amounts, 100, TX_LIMIT_HTG)

    # Si montant > 90% solde → ramener à 10-50% solde
    bal     = users["balance"].values.copy()
    mask    = amounts > bal * 0.9
    
    if mask.sum() > 0:
        amounts[mask] = bal[mask] * RNG.uniform(0.1, 0.5, mask.sum())
    
    amounts = np.where(amounts < 100, np.nan, amounts)

    # Timestamps
    days    = RNG.integers(0, 365, n)
    hours   = RNG.choice(24, size=n, p=HOUR_W)
    mins    = RNG.integers(0, 60, n)
    base_ts = np.array([START_DATE + timedelta(days=int(d)) for d in days])
    timestamps = np.array([
        b.replace(hour=int(h), minute=int(m))
        for b, h, m in zip(base_ts, hours, mins)
    ])

    df = pd.DataFrame({
        "tx_id":              [f"TX{x}" for x in RNG.integers(10**9, 10**10, n, dtype=np.int64)],
        "timestamp":          timestamps,
        "user_id":            users["user_id"].values,
        "recipient_id":       recs["recipient_id"].values,
        "device_id":          users["device_id"].values,
        "operator":           users["operator"].values,
        "platform":           users["platform"].values,
        "amount":             np.round(amounts, 2),
        "oldbalance":         np.round(bal, 2),
        "newbalance":         np.round(np.maximum(bal - amounts, 0), 2),
        "departement":        users["departement"].values,
        "ville":              users["ville"].values,
        "account_age_days":   users["account_age_days"].values,
        "profile":            users["profile"].values,
        "kyc_level":          users["kyc_level"].values,
        "is_agent":           users["is_agent"].values,
        "hour":               hours,
        "day_of_week":        np.array([t.weekday() for t in timestamps]),
        "is_weekend":         np.array([1 if t.weekday()>=5 else 0 for t in timestamps]),
        "is_night":           ((hours < 6) | (hours >= 22)).astype(int),
        "amount_ratio":       np.round(amounts / np.maximum(bal, 1), 4),
        "exceeds_tx_limit":   (amounts > TX_LIMIT_HTG).astype(int),
        "recipient_age_days": recs["account_age_days"].values,
        "recipient_is_new":   (recs["account_age_days"].values < 30).astype(int),
        "is_new_device":      0,
        "isFraud":            0,
        "fraud_type":         None,
    })

    return df.dropna(subset=["amount"])


# ─────────────────────────────────────────────
# FRAUDES VECTORISÉES
# ─────────────────────────────────────────────

def gen_fraud_batch(users_df: pd.DataFrame,
                    recipients_df: pd.DataFrame,
                    n_target: int) -> pd.DataFrame:
    """Génère toutes les fraudes en une passe vectorisée."""

    FRAUD_W = {"ATO": 0.35, "SMURFING": 0.25, "SIM_SWAP": 0.20,
               "CASHOUT_AGENT": 0.12, "MULE": 0.08}
    fraud_types = list(FRAUD_W.keys())
    fraud_probs = np.array(list(FRAUD_W.values()))
    fraud_probs /= fraud_probs.sum()

    rows = []
    count = 0

    while count < n_target:
        ftype = RNG.choice(fraud_types, p=fraud_probs)
        u     = users_df.iloc[int(RNG.integers(0, len(users_df)))].to_dict()
        day   = int(RNG.integers(0, 365))

        if ftype == "ATO":
            amount = u["balance"] * RNG.uniform(0.70, 0.99)
            r      = recipients_df.iloc[int(RNG.integers(0, len(recipients_df)))].to_dict()
            h      = int(RNG.choice([0,1,2,3,4,23])) if RNG.random() < 0.6 \
                     else int(RNG.choice(24, p=HOUR_W))
            ts     = (START_DATE + timedelta(days=day)).replace(
                        hour=h, minute=int(RNG.integers(0,60)))
            new_dev = f"DEV{int(RNG.integers(15_001,20_001)):05d}"
            rows.append(_fraud_row(u, r, amount, new_dev, ts, "ATO",
                                   is_new_device=1))
            count += 1

        elif ftype == "SMURFING":
            target  = int(RNG.integers(80_000, 200_001))
            n_split = int(RNG.integers(4, 10))
            chunk   = target / n_split
            for _ in range(n_split):
                amt = chunk * RNG.uniform(0.85, 1.15)
                amt = min(amt, TX_LIMIT_HTG * 0.95)
                if amt > u["balance"]: break
                r  = recipients_df.iloc[int(RNG.integers(0, len(recipients_df)))].to_dict()
                ts = (START_DATE + timedelta(days=day)).replace(
                        hour=int(RNG.choice(24, p=HOUR_W)),
                        minute=int(RNG.integers(0,60)))
                rows.append(_fraud_row(u, r, amt, u["device_id"], ts, "SMURFING"))
                u["balance"] -= amt
                count += 1

        elif ftype == "SIM_SWAP":
            cloned = f"DEV{int(RNG.integers(20_001,25_001)):05d}"
            n_tx   = int(RNG.integers(2, 6))
            base_ts = (START_DATE + timedelta(days=day)).replace(
                        hour=int(RNG.choice(24, p=HOUR_W)),
                        minute=int(RNG.integers(0,60)))
            for k in range(n_tx):
                amt = int(RNG.integers(2_000, 8_001))
                if amt > u["balance"]: break
                r   = recipients_df.iloc[int(RNG.integers(0, len(recipients_df)))].to_dict()
                ts  = base_ts + timedelta(minutes=k * int(RNG.integers(2,9)))
                rows.append(_fraud_row(u, r, amt, cloned, ts, "SIM_SWAP",
                                       is_new_device=1))
                u["balance"] -= amt
                count += 1

        elif ftype == "CASHOUT_AGENT":
            agents = users_df[users_df["is_agent"] == 1]
            if len(agents) == 0: continue
            agent = agents.iloc[int(RNG.integers(0, len(agents)))].to_dict()
            for _ in range(int(RNG.integers(3, 8))):
                amt = int(RNG.integers(5_000, 25_001))
                if amt > u["balance"]: break
                r   = recipients_df.iloc[int(RNG.integers(0, len(recipients_df)))].to_dict()
                ts  = (START_DATE + timedelta(days=day)).replace(
                        hour=int(RNG.choice(24, p=HOUR_W)),
                        minute=int(RNG.integers(0,60)))
                rows.append(_fraud_row(u, r, amt, agent["device_id"], ts,
                                       "CASHOUT_AGENT", is_agent_tx=1))
                u["balance"] -= amt
                count += 1

        else:  # MULE
            mule_idx = int(RNG.integers(0, len(users_df)))
            mule     = users_df.iloc[mule_idx].to_dict()
            victims  = users_df.sample(int(RNG.integers(2,6))).to_dict("records")
            for v in victims:
                amt = int(RNG.integers(8_000, 35_001))
                if v["balance"] < amt: continue
                ts  = (START_DATE + timedelta(days=day)).replace(
                        hour=int(RNG.choice(24, p=HOUR_W)),
                        minute=int(RNG.integers(0,60)))
                r = {"recipient_id": mule["user_id"],
                     "account_age_days": mule["account_age_days"],
                     "is_known_mule": 1, "departement": mule["departement"]}
                rows.append(_fraud_row(v, r, amt, v["device_id"], ts, "MULE"))
                count += 1

    return pd.DataFrame(rows)


def _fraud_row(user, recipient, amount, device, ts, fraud_type, **extra) -> dict:
    old = float(user["balance"])
    new = max(0.0, old - float(amount))
    row = {
        "tx_id":              f"TX{int(RNG.integers(10**9, 10**10, dtype=np.int64))}",
        "timestamp":          ts,
        "user_id":            user["user_id"],
        "recipient_id":       recipient["recipient_id"],
        "device_id":          device,
        "operator":           user["operator"],
        "platform":           user.get("platform","USSD"),
        "amount":             round(float(amount), 2),
        "oldbalance":         round(old, 2),
        "newbalance":         round(new, 2),
        "departement":        user["departement"],
        "ville":              user.get("ville",""),
        "account_age_days":   user["account_age_days"],
        "profile":            user["profile"],
        "kyc_level":          user.get("kyc_level", 1),
        "is_agent":           user.get("is_agent", 0),
        "hour":               ts.hour,
        "day_of_week":        ts.weekday(),
        "is_weekend":         1 if ts.weekday() >= 5 else 0,
        "is_night":           1 if ts.hour < 6 or ts.hour >= 22 else 0,
        "amount_ratio":       round(float(amount) / max(old, 1), 4),
        "exceeds_tx_limit":   1 if float(amount) > TX_LIMIT_HTG else 0,
        "recipient_age_days": recipient["account_age_days"],
        "recipient_is_new":   1 if recipient["account_age_days"] < 30 else 0,
        "is_new_device":      extra.get("is_new_device", 0),
        "is_agent_tx":        extra.get("is_agent_tx", 0),
        "isFraud":            1,
        "fraud_type":         fraud_type,
    }
    return row


# ─────────────────────────────────────────────
# GÉNÉRATEUR PRINCIPAL
# ─────────────────────────────────────────────

def generate(n=100_000, fraud_ratio=0.025,
             output_file="moncash_natcash_v3.csv", seed=42):

    global RNG
    RNG = np.random.default_rng(seed)
    random.seed(seed)

    print("=" * 55)
    print("  MonCash / NatCash — Haïti  v3.0  (vectorisé)")
    print("=" * 55)

    users_df = create_users(6_000)
    recs_df  = create_recipients(12_000)

    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud
    print(f"  Légitimes : {n_legit:,}   |   Fraudes : {n_fraud:,}")

    print("  [1/3] Génération légitimes (vectorisé)...")
    df_legit = gen_legit_batch(users_df, recs_df, n_legit)

    print("  [2/3] Génération fraudes...")
    df_fraud = gen_fraud_batch(users_df, recs_df, n_fraud)

    print("  [3/3] Assemblage et export...")
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in ["amount","oldbalance","newbalance","amount_ratio"]:
        df[col] = df[col].astype("float32")

    df.to_csv(output_file, index=False)
    _report(df, output_file)
    return df


def _report(df, output_file):
    total = len(df)
    fraud = df["isFraud"].sum()
    print()
    print("=" * 55)
    print("  RAPPORT DATASET")
    print("=" * 55)
    print(f"  Fichier      : {output_file}")
    print(f"  Lignes       : {total:,}   Colonnes : {df.shape[1]}")
    print(f"  Fraudes      : {fraud:,}  ({fraud/total*100:.2f}%)")
    print()
    print("  Par type de fraude :")
    for t, c in df[df["isFraud"] == 1]["fraud_type"].value_counts().items():
        print(f"    {t:<20} {c:>6,}  ({c/fraud*100:.1f}%)")
    print()
    print("  Montants HTG :")
    print(f"    Médiane : {df['amount'].median():>10,.0f}")
    print(f"    Max     : {df['amount'].max():>10,.0f}")
    print("=" * 55)


if __name__ == "__main__":
    import time
    t0 = time.time()
    df = generate(n=100_000, fraud_ratio=0.025,
                  output_file="moncash_natcash_v3.csv", seed=42)
    print(f"\n  Temps total : {time.time()-t0:.1f}s")
    print(f"  Dataset     : {df.shape[0]:,} lignes × {df.shape[1]} colonnes")