"""Render a publication-style overview from an existing experiment's artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from volterra_lab.options import stress

p=argparse.ArgumentParser()
p.add_argument('--run',default='artifacts/demo')
p.add_argument('--out',default='docs/research-board.png')
a=p.parse_args()
root=Path(a.run)
data=pd.read_csv(root/'forecasts.csv',parse_dates=['date'])
meta=json.loads((root/'metrics.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':'#d8e3f1','axes.labelcolor':'#93a6c0','xtick.color':'#93a6c0','ytick.color':'#93a6c0','axes.edgecolor':'#253650','axes.titleweight':'medium'})
fig=plt.figure(figsize=(16,11),facecolor='#09111e',dpi=140)
fig.text(.055,.94,'VOLTERRA / LAB',fontsize=17,fontweight='bold',color='#64e4c2')
fig.text(.055,.874,'Make uncertainty visible.',fontsize=34,color='#eef4fc')
fig.text(.055,.832,'ML VOLATILITY  /  CALIBRATION  /  PORTFOLIO RISK  /  OPTIONS GEOMETRY',fontsize=10,color='#f3bf6a')
fig.text(.945,.94,'SYNTHETIC RESEARCH DEMO',ha='right',fontsize=10,color='#f3bf6a')
metrics=meta['forecast_metrics'];strategy=meta['strategy']
for x,label,value in [(.055,'FORECAST MAE',f"{metrics['forecast']['mae_vol_points']:.2f} vol pts"),(.30,'INTERVAL COVERAGE',f"{metrics['interval']['observed_coverage']*100:.1f}%"),(.55,'MAX DRAWDOWN',f"{strategy['max_drawdown']*100:.1f}%"),(.79,'NET VOLATILITY',f"{strategy['annualized_net_volatility']*100:.1f}%")]:
 fig.text(x,.777,label,fontsize=9,color='#8b9ab1');fig.text(x,.742,value,fontsize=21,color='#64e4c2')
axes=[fig.add_axes(rect,facecolor='#101a2b') for rect in [(.065,.407,.415,.255),(.56,.407,.39,.255),(.065,.089,.415,.235),(.56,.089,.39,.235)]]
for ax in axes:
 ax.grid(color='#23314a',alpha=.6,linewidth=.5);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
ax=axes[0];ax.set_title('01 / Forward volatility & uncertainty',loc='left',pad=14,color='#d8e3f1')
ax.fill_between(data.date,data.lower*100,data.upper*100,color='#64e4c2',alpha=.12)
ax.plot(data.date,data.target*100,color='#f3bf6a',lw=.6,alpha=.7,label='Forward realized')
ax.plot(data.date,data.forecast*100,color='#64e4c2',lw=1,label='ML forecast')
ax.set_ylabel('Annualized vol (%)');ax.legend(frameon=False,fontsize=8,labelcolor='#d8e3f1')
ax=axes[1];ax.set_title('02 / Wealth after trading friction',loc='left',pad=14,color='#d8e3f1')
ax.plot(data.date,data.nav,color='#64e4c2',lw=1.5,label='Vol targeting, net');ax.plot(data.date,data.buyhold,color='#f3bf6a',lw=1,label='Buy & hold')
ax.set_ylabel('Wealth / initial capital');ax.legend(frameon=False,fontsize=8,labelcolor='#d8e3f1')
ax=axes[2];ax.set_title('03 / Drawdowns stay in the picture',loc='left',pad=14,color='#d8e3f1')
ax.fill_between(data.date,data.drawdown*100,0,color='#f5829a',alpha=.25);ax.plot(data.date,data.drawdown*100,color='#f5829a',lw=.9);ax.set_ylabel('Drawdown (%)')
spot=float(data.close.iloc[-1]);vol=float(data.forecast.iloc[-1])+.04
moves,shocks,pnl=stress(spot,vol)
ax=axes[3];ax.grid(False);ax.set_title('04 / Seven-day straddle stress',loc='left',pad=14,color='#d8e3f1')
im=ax.pcolormesh(moves*100,shocks*100,pnl,cmap='RdBu',shading='auto');ax.set_xlabel('Spot move (%)');ax.set_ylabel('IV shock (vol points)')
cb=fig.colorbar(im,ax=ax,pad=.02);cb.ax.tick_params(labelsize=8);cb.set_label('P&L per share',fontsize=8)
fig.text(.055,.027,'14 interactive views in dashboard.html  ·  Simulated data, not tradable alpha  ·  Options IV is illustrative, not market calibrated',fontsize=9,color='#93a6c0')
Path(a.out).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.out,facecolor=fig.get_facecolor());plt.close(fig)
