import unittest
import numpy as np
from volterra_lab.research import simulate,features,walk_forward,risk_backtest,conformal_radius,FEATURES,block_interval
from volterra_lab.options import black_scholes


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=simulate(800)
        cls.frame=features(cls.data)
        cls.pred,cls.audit,cls.metrics=walk_forward(cls.frame,step=60)

    def test_target_matches_future_window(self):
        raw=np.log(self.data.close).diff()
        row=self.frame.iloc[0];t=int(row.origin)
        self.assertAlmostEqual(row.target,np.sqrt(np.mean(raw.iloc[t+1:t+6]**2)*252))

    def test_features_do_not_use_future(self):
        altered=self.data.copy();altered.loc[600:,'close']*=2
        a=features(altered)
        np.testing.assert_allclose(self.frame.loc[self.frame.origin<600,FEATURES],a.loc[a.origin<600,FEATURES])

    def test_audit_proves_label_purge(self):
        for fold in self.audit:
            self.assertLess(fold['train_last_label_end'],fold['calibration_first_origin'])
            self.assertLess(fold['calibration_last_label_end'],fold['test_first_origin'])

    def test_future_changes_do_not_change_earlier_forecasts(self):
        altered=self.data.copy();altered.loc[650:,'close']*=1.5
        p,_,_=walk_forward(features(altered),step=60)
        np.testing.assert_allclose(self.pred.loc[self.pred.origin<650,'forecast'],p.loc[p.origin<650,'forecast'])

    def test_conformal_finite_sample_order_statistic(self):
        self.assertEqual(conformal_radius(np.arange(1,11),alpha=.2),9.)
        self.assertTrue((self.pred.lower<=self.pred.upper).all())

    def test_execution_lag(self):
        p,_=risk_backtest(self.pred)
        self.assertEqual(p.weight.iloc[0],0)
        self.assertEqual(p.weight.iloc[1],0)
        self.assertAlmostEqual(p.weight.iloc[2],min(1.5,.15/max(.03,self.pred.forecast.iloc[0])))

    def test_costs_reduce_terminal_wealth(self):
        low,_=risk_backtest(self.pred,cost_bps=0)
        high,_=risk_backtest(self.pred,cost_bps=20)
        self.assertLess(high.nav.iloc[-1],low.nav.iloc[-1])

    def test_block_bootstrap_constant_difference(self):
        result=block_interval(np.ones(100)*2,draws=100)
        self.assertEqual(result['mean_delta'],2.)
        self.assertEqual(result['lower_95'],2.)
        self.assertEqual(result['upper_95'],2.)

    def test_input_rejects_duplicates(self):
        data=self.data.copy();data.loc[1,'date']=data.loc[0,'date']
        with self.assertRaises(ValueError): features(data)


class OptionTests(unittest.TestCase):
    def test_put_call_parity(self):
        p=black_scholes(100,np.array([80,100,120]),.5,.03,.25)
        np.testing.assert_allclose(p['call']-p['put'],100-np.array([80,100,120])*np.exp(-.03*.5))

    def test_greeks_against_finite_differences(self):
        s,k,t,r,v=100,105,.5,.03,.25
        p=black_scholes(s,k,t,r,v);eps=.01
        plus=black_scholes(s+eps,k,t,r,v)['call'];minus=black_scholes(s-eps,k,t,r,v)['call']
        self.assertAlmostEqual(float(p['delta']),float((plus-minus)/(2*eps)),places=6)
        self.assertAlmostEqual(float(p['gamma']),float((plus-2*p['call']+minus)/eps**2),places=6)
        a=black_scholes(s,k,t,r,v+1e-5)['call'];b=black_scholes(s,k,t,r,v-1e-5)['call']
        self.assertAlmostEqual(float(p['vega_per_vol_point']),float((a-b)/(2e-5)*.01),places=6)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): black_scholes(100,100,0,.03,.2)

if __name__=='__main__': unittest.main()
