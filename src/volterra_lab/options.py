import numpy as np
from scipy.special import ndtr


def black_scholes(spot, strike, years, rate, volatility, dividend=0.):
    s, k, t, v = np.broadcast_arrays(np.asarray(spot,dtype=float), np.asarray(strike,dtype=float),
                                    np.asarray(years,dtype=float), np.asarray(volatility,dtype=float))
    if any(not np.isfinite(x).all() for x in (s,k,t,v)) or not np.isfinite([rate,dividend]).all():
        raise ValueError("Inputs must be finite")
    if np.any(s<=0) or np.any(k<=0) or np.any(t<=0) or np.any(v<=0):
        raise ValueError("Spot, strike, expiry and volatility must be positive")
    root = np.sqrt(t)
    d1 = (np.log(s/k)+(rate-dividend+.5*v*v)*t)/(v*root)
    d2 = d1-v*root
    discount, divdiscount = np.exp(-rate*t), np.exp(-dividend*t)
    density = np.exp(-.5*d1*d1)/np.sqrt(2*np.pi)
    call = s*divdiscount*ndtr(d1)-k*discount*ndtr(d2)
    put = k*discount*ndtr(-d2)-s*divdiscount*ndtr(-d1)
    return {"call": call, "put": put, "delta": divdiscount*ndtr(d1),
            "gamma": divdiscount*density/(s*v*root),
            "vega_per_vol_point": s*divdiscount*density*root*.01,
            "theta_per_day": (-s*divdiscount*density*v/(2*root)-rate*k*discount*ndtr(d2)
                              +dividend*s*divdiscount*ndtr(d1))/365}


def surface(spot, base_vol, rate=.03):
    strikes = np.linspace(.7*spot, 1.3*spot, 45)
    days = np.linspace(7, 365, 35)
    k, t = np.meshgrid(strikes, days/365)
    # Illustrative skew assumptions; never presented as a market-calibrated IV surface.
    m = np.log(k/spot)
    vol = np.clip(base_vol*(1-.65*m+.9*m*m)+.025*np.sqrt(t), .03, 2.)
    priced = black_scholes(spot, k, t, rate, vol)
    return strikes, days, vol, priced


def stress(spot, volatility, rate=.03):
    moves = np.linspace(-.25, .25, 51)
    vol_shocks = np.linspace(-.10, .25, 43)
    x, y = np.meshgrid(moves, vol_shocks)
    entry = black_scholes(spot, spot, 30/365, rate, volatility)
    future = black_scholes(spot*(1+x), spot, 23/365, rate, np.maximum(.01, volatility+y))
    pnl = future["call"]+future["put"]-entry["call"]-entry["put"]
    return moves, vol_shocks, pnl
