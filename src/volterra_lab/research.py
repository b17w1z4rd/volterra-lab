from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["rv5", "rv10", "rv21", "rv63", "abs_return", "downside21", "momentum21", "drawdown63"]


def simulate(days=1600, seed=42):
    """Synthetic regime-switching stochastic volatility with occasional jumps."""
    if days < 700:
        raise ValueError("Use at least 700 observations")
    rng = np.random.default_rng(seed)
    price, state, logvol = 100., 0, np.log(.16)
    rows = []
    for day in pd.bdate_range("2018-01-02", periods=days):
        if rng.random() < (.025 if state == 0 else .12):
            state = 1-state
        mean = np.log(.16 if state == 0 else .48)
        logvol = .90*logvol + .10*mean + rng.normal(0, .08)
        shock = rng.normal(0, .035) if rng.random() < .012 else 0.
        r = .05/252 + np.exp(logvol)/np.sqrt(252)*rng.standard_t(7)*np.sqrt(5/7) + shock
        price *= np.exp(r)
        rows.append((day, price))
    return pd.DataFrame(rows, columns=["date", "close"])


def validate_prices(data):
    if not {"date", "close"}.issubset(data.columns):
        raise ValueError("CSV requires date and close columns")
    data = data[["date", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["close"] = pd.to_numeric(data["close"], errors="raise")
    if (data["date"].isna().any() or data["date"].duplicated().any()
            or not data["date"].is_monotonic_increasing):
        raise ValueError("Dates must be unique, valid and increasing")
    if not np.isfinite(data["close"]).all() or (data["close"] <= 0).any():
        raise ValueError("Prices must be finite and positive")
    if len(data) < 700:
        raise ValueError("At least 700 daily observations are required")
    return data.reset_index(drop=True)


def features(data, horizon=5):
    if not 1 <= horizon <= 21:
        raise ValueError("horizon must be from 1 to 21 observations")
    data = validate_prices(data)
    r = np.log(data.close).diff()
    frame = pd.DataFrame({"date": data.date, "close": data.close, "return": r})
    for n in (5, 10, 21, 63):
        frame[f"rv{n}"] = np.sqrt(r.pow(2).rolling(n).mean()*252)
    frame["abs_return"] = r.abs()
    frame["downside21"] = np.sqrt(r.clip(upper=0).pow(2).rolling(21).mean()*252)
    frame["momentum21"] = data.close.pct_change(21)
    frame["drawdown63"] = data.close/data.close.rolling(63).max()-1
    # Row t target uses returns t+1 through t+h; features only use <=t.
    frame["target"] = np.sqrt(sum(r.shift(-j).pow(2) for j in range(1, horizon+1))/horizon*252)
    frame["origin"] = np.arange(len(frame))
    frame["label_end"] = frame.origin+horizon
    return frame.dropna().reset_index(drop=True)


def conformal_radius(residuals, alpha=.1):
    residuals = np.asarray(residuals)
    if not 0 < alpha < 1 or len(residuals) == 0 or not np.isfinite(residuals).all():
        raise ValueError("Invalid calibration residuals or alpha")
    rank = min(len(residuals), math.ceil((len(residuals)+1)*(1-alpha)))
    return float(np.sort(residuals)[rank-1])


def walk_forward(frame, seed=42, minimum_train=300, calibration=100, step=21, alpha=.1):
    if minimum_train < 50 or calibration < 20 or step < 1:
        raise ValueError("Invalid walk-forward sample sizes")
    predictions, audit = [], []
    start = minimum_train+calibration+2*int((frame.label_end-frame.origin).iloc[0])
    for pos in range(start, len(frame), step):
        test = frame.iloc[pos:pos+step]
        mature = frame[frame.label_end < int(test.origin.iloc[0])]
        cal = mature.tail(calibration)
        if len(cal) < calibration:
            continue
        train = frame[frame.label_end < int(cal.origin.iloc[0])]
        if len(train) < minimum_train:
            continue
        x, y = train[FEATURES], np.log(np.maximum(train.target, 1e-6))
        model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=10, learning_rate=.06,
                                             l2_regularization=5, random_state=seed)
        linear = make_pipeline(StandardScaler(), Ridge(alpha=10.))
        model.fit(x, y)
        linear.fit(x, y)
        cal_residual = np.abs(np.log(np.maximum(cal.target, 1e-6))-model.predict(cal[FEATURES]))
        radius = conformal_radius(cal_residual, alpha)
        log_prediction = model.predict(test[FEATURES])
        rows = test.copy()
        rows["forecast"] = np.exp(log_prediction)
        rows["linear"] = np.exp(linear.predict(test[FEATURES]))
        rows["lower"], rows["upper"] = np.exp(log_prediction-radius), np.exp(log_prediction+radius)
        rows["fold"] = len(audit)
        predictions.append(rows)
        audit.append({"fold": len(audit), "train_rows": len(train), "calibration_rows": len(cal),
                      "train_last_label_end": int(train.label_end.max()),
                      "calibration_first_origin": int(cal.origin.min()),
                      "calibration_last_label_end": int(cal.label_end.max()),
                      "test_first_origin": int(test.origin.min()), "test_last_origin": int(test.origin.max())})
    if not predictions:
        raise ValueError("Not enough data for the requested train/calibration split")
    output = pd.concat(predictions, ignore_index=True)
    metrics = {}
    for name in ("forecast", "linear", "rv21"):
        predicted, observed = output[name].to_numpy(), output.target.to_numpy()
        ratio = observed**2/np.maximum(predicted**2, 1e-12)
        metrics[name] = {"mae_vol_points": float(np.mean(np.abs(predicted-observed))*100),
                         "qlike": float(np.mean(ratio-np.log(np.maximum(ratio, 1e-12))-1))}
    metrics["interval"] = {"nominal_coverage": 1-alpha,
                           "observed_coverage": float(((output.target>=output.lower)&(output.target<=output.upper)).mean()),
                           "mean_width_vol_points": float((output.upper-output.lower).mean()*100)}
    metrics["paired_qlike_comparison"] = {}
    observed = output.target.to_numpy()**2
    def loss(column):
        ratio = observed/np.maximum(output[column].to_numpy()**2,1e-12)
        return ratio-np.log(np.maximum(ratio,1e-12))-1
    for reference in ("linear", "rv21"):
        metrics["paired_qlike_comparison"][reference] = block_interval(loss("forecast")-loss(reference), seed=seed)
    return output, audit, metrics


def risk_backtest(predictions, target_vol=.15, cost_bps=5., max_weight=1.5, funding_rate=.05):
    """One extra close-to-close lag between observing a forecast and earning a return."""
    if target_vol <= 0 or cost_bps < 0 or max_weight <= 0 or funding_rate < 0:
        raise ValueError("Invalid strategy assumptions")
    p = predictions.copy()
    if not np.all(np.diff(p.origin.to_numpy()) == 1):
        raise ValueError("Backtest requires consecutive forecast origins")
    desired = (target_vol/p.forecast.clip(lower=.03)).clip(0, max_weight)
    # Forecast at t is implemented at t+1 close, earning return at t+2.
    executed = desired.shift(1).fillna(0.)
    weights = executed.shift(1).fillna(0.)
    turnover = executed.diff().fillna(executed).abs()
    net = weights*np.expm1(p["return"])-turnover*cost_bps/10000-np.maximum(weights-1,0)*funding_rate/252
    nav = (1+net).cumprod()
    buyhold = (1+np.expm1(p["return"])).cumprod()
    drawdown = nav/nav.cummax().clip(lower=1)-1
    p["weight"], p["turnover"], p["net_return"], p["nav"], p["buyhold"], p["drawdown"] = weights, turnover, net, nav, buyhold, drawdown
    metrics = {"target_volatility": target_vol, "cost_bps_per_unit_turnover": cost_bps,
               "max_weight": max_weight, "funding_rate": funding_rate, "annualized_net_volatility": float(net.std(ddof=1)*np.sqrt(252)),
               "annualized_return": float(nav.iloc[-1]**(252/len(nav))-1),
               "max_drawdown": float(drawdown.min()), "total_turnover": float(turnover.sum()),
               "annualized_sharpe_zero_cash": float(net.mean()/net.std(ddof=1)*np.sqrt(252)) if net.std(ddof=1)>0 else None}
    return p, metrics


def block_interval(values, seed=42, block=21, draws=1000):
    """Circular moving-block bootstrap of the mean; descriptive under regime changes."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Expected a finite vector of paired losses")
    rng = np.random.default_rng(seed)
    block = min(block, len(values))
    starts = rng.integers(0, len(values), size=(draws, math.ceil(len(values)/block)))
    indices = ((starts[:,:,None]+np.arange(block)) % len(values)).reshape(draws,-1)[:,:len(values)]
    estimates = values[indices].mean(axis=1)
    return {"mean_delta": float(values.mean()), "lower_95": float(np.quantile(estimates,.025)),
            "upper_95": float(np.quantile(estimates,.975)), "block_length": block, "draws": draws}
