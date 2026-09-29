from pathlib import Path
import html
import json
import numpy as np
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from .options import surface, stress, black_scholes

COLORS = {"mint":"#64e4c2", "gold":"#f3bf6a", "blue":"#7f9dff", "red":"#f5829a", "text":"#d8e3f1", "muted":"#8b9ab1"}


def style(fig, title, ylabel="", height=340):
    fig.update_layout(title={"text":title,"font":{"size":15}}, height=height,
                      margin=dict(l=55,r=25,t=55,b=45), paper_bgcolor="#101a2b",plot_bgcolor="#101a2b",
                      font=dict(family="Arial, sans-serif",color=COLORS["text"],size=11),
                      colorway=list(COLORS.values())[:4], hovermode="x unified",
                      legend=dict(orientation="h",y=1.12,x=1,xanchor="right",font=dict(size=10)))
    fig.update_xaxes(gridcolor="#1d2b40",zeroline=False)
    fig.update_yaxes(gridcolor="#1d2b40",zeroline=False,title=ylabel)
    return fig


def build(prices, p, metrics, strategy, audit, destination, synthetic=True, horizon=5):
    figures=[]
    def add(fig, title, note, ylabel="", wide=False, height=340):
        style(fig,title,ylabel,height)
        figures.append((fig,title,note,wide))
    price=go.Figure(go.Scatter(x=prices.date,y=prices.close,name="Price",line=dict(color=COLORS['mint'],width=2)))
    add(price,"01 · The market tape","Synthetic regime-switching prices." if synthetic else "Imported price series; adjustment quality is the user's responsibility.","Price",True,300)
    f=go.Figure()
    f.add_trace(go.Scatter(x=p.date,y=p.upper*100,line=dict(width=0),showlegend=False,hoverinfo="skip"))
    f.add_trace(go.Scatter(x=p.date,y=p.lower*100,line=dict(width=0),fill="tonexty",fillcolor="rgba(100,228,194,.13)",name="90% nominal interval"))
    for column,name,color in (("target","Forward realized",COLORS['gold']),("forecast","Gradient boosting",COLORS['mint']),("rv21","Trailing 21d",COLORS['blue'])):
        f.add_trace(go.Scatter(x=p.date,y=p[column]*100,name=name,line=dict(color=color,width=1.5)))
    add(f,"02 · Forecast with uncertainty",f"{horizon}-observation forward realized volatility. Predictions are out of sample; interval coverage is empirical, not guaranteed.","Annualized vol (%)",True,400)
    covered=((p.target>=p.lower)&(p.target<=p.upper)).astype(float)
    f=go.Figure(go.Scatter(x=p.date,y=covered.rolling(63).mean()*100,name="63d realized coverage",line=dict(color=COLORS['mint'])))
    f.add_hline(y=90,line_dash="dot",line_color=COLORS['gold'])
    add(f,"03 · Does uncertainty hold up?","A rolling calibration diagnostic. Serial dependence and changing regimes can break nominal coverage.","Coverage (%)")
    f=go.Figure(go.Scatter(x=p.target*100,y=p.forecast*100,mode="markers",marker=dict(size=5,color=p.fold,colorscale="Tealgrn",opacity=.6),name="Forecast origins"))
    top=float(max(p.target.max(),p.forecast.max())*100)
    f.add_trace(go.Scatter(x=[0,top],y=[0,top],mode="lines",line=dict(color=COLORS['gold'],dash="dot"),showlegend=False))
    add(f,"04 · Calibration in the cross-section","Each dot is one held-out forecast origin; colors identify successive refits.","Predicted vol (%)")
    f.update_xaxes(title="Realized vol (%)")
    labels=["Gradient boosting","Linear ridge","Trailing 21d"]
    f=go.Figure(go.Bar(x=labels,y=[metrics[k]['qlike'] for k in ('forecast','linear','rv21')],marker_color=[COLORS['mint'],COLORS['blue'],COLORS['gold']]))
    add(f,"05 · Beat a baseline, or say so","QLIKE is a variance forecast loss. Lower is better; fixed hyperparameters were not selected on these test results.","QLIKE loss")
    f=go.Figure()
    errors=(p.forecast-p.target)*100
    f.add_trace(go.Histogram(x=errors,nbinsx=45,marker_color=COLORS['mint'],name="Forecast errors"))
    f.add_vline(x=0,line_dash="dot",line_color=COLORS['gold'])
    add(f,"06 · Where the model misses","Signed errors reveal over- and underestimation. Overlapping forward targets make these errors dependent.","Count")
    f.update_xaxes(title="Predicted − realized vol points")
    f=go.Figure()
    for col,name,color in (("nav","Vol targeting, net",COLORS['mint']),("buyhold","Buy & hold",COLORS['gold'])):
        f.add_trace(go.Scatter(x=p.date,y=p[col],name=name,line=dict(color=color,width=2)))
    add(f,"07 · Risk targeting, after friction",f"15% target; 1.5× cap; {strategy['cost_bps_per_unit_turnover']:g} bps per unit turnover; 5% annual borrowing cost; an extra close-to-close execution lag.","Wealth / initial capital",True,360)
    f=go.Figure(go.Scatter(x=p.date,y=p.drawdown*100,fill="tozeroy",line=dict(color=COLORS['red'],width=1),name="Drawdown"))
    add(f,"08 · The underwater view","Risk-control results from the same held-out period, with zero interest on uninvested cash.","Drawdown (%)")
    f=go.Figure(go.Scatter(x=p.date,y=p.weight,line=dict(color=COLORS['blue'],width=1.5),name="Held exposure"))
    f.add_hline(y=1,line_dash="dot",line_color=COLORS['muted'])
    add(f,"09 · Capital deployed","Exposure uses a forecast from two observations earlier; transaction costs are charged when positions change.","Underlying exposure (×)")
    spot=float(p.close.iloc[-1]); anchor=float(p.forecast.iloc[-1]); assumed_iv=anchor+.04
    strikes,days,vol,priced=surface(spot,assumed_iv)
    f=go.Figure(go.Surface(x=strikes/spot,y=days,z=vol*100,colorscale="Tealgrn",colorbar=dict(title="IV %",len=.7)))
    add(f,"10 · The options landscape","ILLUSTRATIVE implied-volatility assumptions: ML realized-vol forecast + 4 vol points, with a chosen skew and term structure. Not market quotes or an arbitrage-free calibration.",wide=True,height=530)
    f.update_layout(scene=dict(xaxis_title="Strike / spot",yaxis_title="Days to expiry",zaxis_title="Assumed IV (%)",
                              bgcolor="#101a2b",camera=dict(eye=dict(x=1.5,y=-1.6,z=1.0))),hovermode="closest")
    # Surface dropdown makes the same strike/tenor grid an options risk workbench.
    f.update_layout(updatemenus=[dict(type="buttons",direction="right",x=0,y=1.08,
        bgcolor="#1d2b40",font=dict(color="#d8e3f1"),buttons=[
          dict(label=label,method="update",args=[{"z":[z]}, {"scene.zaxis.title.text":zlabel}])
          for label,z,zlabel in [("Implied vol",vol*100,"Assumed IV (%)"),("Call value",priced['call'],"Price per share"),("Gamma",priced['gamma'],"Gamma"),("Vega",priced['vega_per_vol_point'],"Vega / vol point")]])])
    moves,shocks,pnl=stress(spot,assumed_iv)
    f=go.Figure(go.Heatmap(x=moves*100,y=shocks*100,z=pnl,colorscale="RdBu",zmid=0,colorbar=dict(title="P&L")))
    add(f,"11 · Seven-day straddle stress","Long one ATM call + put, 30 days at entry, 23 days after the shock. P&L per share before fees; European Black–Scholes with 3% rate and zero dividends.","IV shock (vol points)",True,450)
    f.update_xaxes(title="Spot move (%)")
    x=np.linspace(spot*.7,spot*1.3,151)
    greeks=black_scholes(x,spot,30/365,.03,assumed_iv)
    f=go.Figure()
    f.add_trace(go.Scatter(x=x/spot,y=greeks['delta'],name="Call delta",line=dict(color=COLORS['mint'])))
    f.add_trace(go.Scatter(x=x/spot,y=greeks['vega_per_vol_point'],name="Vega / vol point",line=dict(color=COLORS['gold'])))
    add(f,"12 · Greeks through spot","Different units share this diagnostic panel; hover or toggle legend entries to isolate a curve.","Delta / vega")
    f.update_xaxes(title="Spot / strike")
    f=go.Figure()
    for fold in audit[::max(1,len(audit)//12)]:
        n=fold['fold']
        for start,end,label,color in [(0,fold['train_last_label_end'],'Train',COLORS['blue']),
                                      (fold['calibration_first_origin'],fold['calibration_last_label_end'],'Calibration',COLORS['gold']),
                                      (fold['test_first_origin'],fold['test_last_origin'],'Test',COLORS['mint'])]:
            f.add_trace(go.Scatter(x=[start,end],y=[n,n],mode="lines",name=label,line=dict(color=color,width=8),showlegend=False))
    add(f,"13 · The information boundary","Blue = expanding training; gold = calibration; green = test. Labels mature before the next block. Full split boundaries are saved in audit.json.","Refit index")
    f.update_xaxes(title="Original observation index")

    comparison=metrics['paired_qlike_comparison']
    means=[comparison[k]['mean_delta'] for k in ('linear','rv21')]
    f=go.Figure(go.Scatter(x=means,y=['ML − linear ridge','ML − trailing 21d'],mode='markers',
                          marker=dict(size=12,color=COLORS['mint']),error_x=dict(type='data',symmetric=False,
                          array=[comparison[k]['upper_95']-comparison[k]['mean_delta'] for k in ('linear','rv21')],
                          arrayminus=[comparison[k]['mean_delta']-comparison[k]['lower_95'] for k in ('linear','rv21')])))
    f.add_vline(x=0,line_color=COLORS['gold'],line_dash='dot')
    add(f,"14 · How uncertain is the comparison?","Paired QLIKE difference, ML minus baseline; negative favors ML. 1,000 circular moving-block bootstrap draws, 21-observation blocks, 95% intervals. Descriptive under changing regimes.",wide=True,height=280)
    f.update_xaxes(title="Mean QLIKE difference")

    cards=[]
    for i,(fig,title,note,wide) in enumerate(figures):
        body=fig.to_html(full_html=False,include_plotlyjs=False,div_id=f"chart-{i}",config={"displaylogo":False,"responsive":True})
        cards.append(f'<article class="panel {"wide" if wide else ""}">{body}<p class="caption">{html.escape(note)}</p></article>')
    coverage=metrics['interval']['observed_coverage']*100
    badge="SYNTHETIC MARKET · RESEARCH DEMO" if synthetic else "IMPORTED DAILY PRICES · RESEARCH RUN"
    date=p.date.iloc[-1].strftime('%d %b %Y')
    checks=f"{len(audit)} audited refits · {len(p):,} held-out origins · {horizon}-day target"
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Volterra Lab · Volatility Research</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#09111e;color:#d8e3f1;font-family:Arial,sans-serif}header,main,footer{max-width:1400px;margin:auto;padding:30px 36px}nav{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #25324a;padding-bottom:22px}.brand{font-weight:800;font-size:20px;letter-spacing:3px}.pill{color:#64e4c2;background:#14342f;border:1px solid #255548;padding:8px 13px;border-radius:24px;font-size:10px;letter-spacing:1px}.hero{padding:48px 0 22px;display:grid;grid-template-columns:2fr 1fr;gap:32px}.eyebrow{font-size:11px;letter-spacing:3px;color:#f3bf6a}h1{font-size:clamp(38px,5vw,70px);font-weight:500;letter-spacing:-3px;line-height:1.05;margin:20px 0}h1 em{font-style:normal;color:#64e4c2}.intro{max-width:650px;color:#98abc3;line-height:1.8;font-size:15px}.heroaside{align-self:end;border-left:2px solid #f3bf6a;padding:5px 0 5px 22px;line-height:1.8;font-size:13px;color:#9babc1}.heroaside strong{display:block;color:#d8e3f1}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:26px}.metric{background:#101a2b;border:1px solid #23314a;border-radius:14px;padding:22px}.metric label{font-size:10px;letter-spacing:1.5px;color:#91a3bb}.metric b{display:block;margin-top:14px;font-size:30px;font-weight:500;color:#64e4c2}.metric small{display:block;color:#8b9ab1;margin-top:10px;font-size:11px}.section{display:flex;justify-content:space-between;gap:20px;margin:22px 0;color:#91a3bb;font-size:12px;line-height:1.6}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}.panel{border:1px solid #223047;border-radius:16px;overflow:hidden;background:#101a2b;min-width:0}.wide{grid-column:1/-1}.caption{margin:0;padding:0 24px 22px;color:#8f9fb6;font-size:12px;line-height:1.7}footer{color:#8395ae;font-size:12px;line-height:1.8;border-top:1px solid #223047;margin-top:35px;padding-bottom:45px}a{color:#64e4c2}@media(max-width:780px){header,main,footer{padding:22px 15px}.hero{grid-template-columns:1fr;padding-top:30px}.metrics{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.wide{grid-column:auto}.section{flex-direction:column}.pill{font-size:8px}h1{letter-spacing:-2px}}
</style><script>''' + get_plotlyjs()+'''</script></head><body>'''
    page+=f'''<header><nav><div class="brand">VOLTERRA<span style="color:#64e4c2"> / </span>LAB</div><span class="pill">{badge}</span></nav><div class="hero"><div><div class="eyebrow">MACHINE LEARNING × OPTIONS RISK</div><h1>Make uncertainty<br><em>visible.</em></h1><p class="intro">A volatility research desk that connects forecasts to uncertainty, portfolio risk and the geometry of options. Every result comes with a benchmark and an information boundary.</p></div><aside class="heroaside"><strong>RESEARCH, WITH RECEIPTS</strong>{checks}<br>Historical scenario anchor: {date}<br>14 interactive research views<br>Self-contained · no external scripts</aside></div><div class="metrics">
<div class="metric"><label>FORECAST ERROR</label><b>{metrics['forecast']['mae_vol_points']:.2f}</b><small>MAE · annualized vol points</small></div>
<div class="metric"><label>INTERVAL COVERAGE</label><b>{coverage:.1f}%</b><small>90% nominal · empirical only</small></div>
<div class="metric"><label>NET MAX DRAWDOWN</label><b>{strategy['max_drawdown']*100:.1f}%</b><small>Illustrative vol-targeting strategy</small></div>
<div class="metric"><label>REALIZED NET VOL</label><b>{strategy['annualized_net_volatility']*100:.1f}%</b><small>15% portfolio target</small></div></div></header><main><div class="section"><span>THE RESEARCH CANVAS / Hover, zoom, isolate series and rotate the options surface.</span><span>Fixed model settings · delayed execution · explicit costs</span></div><div class="grid">'''
    page+=''.join(cards)+'''</div></main><footer><strong>Interpretation matters.</strong> Demo results describe a simulated market and do not establish tradable alpha. Forward volatility windows overlap; calibration coverage has no distribution-free time-series guarantee. Options surfaces are illustrative assumptions, not observed implied volatility; the smile has not been checked for arbitrage. Black–Scholes values European options and omits early exercise. Historical risk-targeting results exclude taxes, market impact and dividends not represented in the input price series. This is a research artifact, not an investment recommendation.</footer></body></html>'''
    Path(destination).write_text(page)
