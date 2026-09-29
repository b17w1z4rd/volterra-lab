import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from .research import simulate, features, walk_forward, risk_backtest
from .dashboard import build


def main():
    parser=argparse.ArgumentParser(description="Reproducible ML volatility and options research")
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("demo","run"):
        p=sub.add_parser(command)
        p.add_argument("--out",default="artifacts/"+command)
        p.add_argument("--seed",type=int,default=42)
        p.add_argument("--horizon",type=int,default=5)
        p.add_argument("--cost-bps",type=float,default=5.)
        if command=="demo": p.add_argument("--days",type=int,default=1600)
        else: p.add_argument("--csv",required=True)
    args=parser.parse_args()
    data=simulate(args.days,args.seed) if args.command=="demo" else pd.read_csv(args.csv)
    frame=features(data,args.horizon)
    p,audit,metrics=walk_forward(frame,seed=args.seed)
    p,strategy=risk_backtest(p,cost_bps=args.cost_bps)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    data_bytes=data.to_csv(index=False).encode()
    metadata={"seed":args.seed,"synthetic":args.command=="demo","horizon":args.horizon,
              "rows":len(data),"data_sha256":hashlib.sha256(data_bytes).hexdigest(),
              "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
              "sklearn":sklearn.__version__,"forecast_metrics":metrics,"strategy":strategy}
    (out/"metrics.json").write_text(json.dumps(metadata,indent=2)+"\n")
    (out/"audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    p.to_csv(out/"forecasts.csv",index=False)
    # Persist exact inputs with each experiment for reproducibility.
    (out/"prices.csv").write_bytes(data_bytes)
    build(data,p,metrics,strategy,audit,out/"dashboard.html",synthetic=args.command=="demo",horizon=args.horizon)
    print(json.dumps(metadata,indent=2))
    print("Dashboard: "+str(out/"dashboard.html"))


if __name__=="__main__":
    main()
