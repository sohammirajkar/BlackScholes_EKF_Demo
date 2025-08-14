import React, { useMemo, useState } from "react";
import axios from "axios";
import "./App.css";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { Label } from "./components/ui/label";
import { Card, CardContent, CardHeader, CardTitle } from "./components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import { Slider } from "./components/ui/slider";
import { Badge } from "./components/ui/badge";
import { Play, LineChart, FlaskConical } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function number(v, d = 4) {
  if (v === null || v === undefined || Number.isNaN(v)) return "-";
  return Number(v).toFixed(d);
}

function MiniLineChart({ series = [], width = 840, height = 220, padding = 24 }) {
  // series: [{name, color, values: number[] }]
  const allVals = series.flatMap((s) => s.values);
  const [minV, maxV] = useMemo(() => {
    const mn = Math.min(...allVals);
    const mx = Math.max(...allVals);
    // if flat line
    if (!isFinite(mn) || !isFinite(mx) || mn === mx) return [mn || 0, (mx || 1) + 1e-6];
    return [mn, mx];
  }, [allVals.join(",")]);

  const w = width - padding * 2;
  const h = height - padding * 2;

  const scaleX = (i, n) => (i / Math.max(1, n - 1)) * w + padding;
  const scaleY = (v) => height - padding - ((v - minV) / (maxV - minV)) * h;

  return (
    <svg width={width} height={height} className="chart-svg">
      <defs>
        <linearGradient id="grad" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stopColor="var(--emerald-400)" />
          <stop offset="100%" stopColor="var(--cyan-400)" />
        </linearGradient>
      </defs>
      <rect x="0" y="0" width={width} height={height} rx="16" className="chart-bg" />
      {series.map((s, idx) => {
        const path = s.values
          .map((v, i) => `${i === 0 ? "M" : "L"}${scaleX(i, s.values.length)},${scaleY(v)}`)
          .join(" ");
        return (
          <g key={idx}>
            <path d={path} fill="none" stroke={s.color || "url(#grad)"} strokeWidth="2.2" />
          </g>
        );
      })}
    </svg>
  );
}

function App() {
  const [loading, setLoading] = useState(false);
  const [runId, setRunId] = useState(null);
  const [simData, setSimData] = useState(null);
  const [fitData, setFitData] = useState(null);

  const [params, setParams] = useState({
    n: 220,
    S0: 100,
    mu: 0.05,
    sigma_true: 0.2,
    K: undefined, // defaults to S0
    r: 0.02,
    T: 0.25,
    dt: 1 / 252,
    obs_noise_std: 0.5,
    seed: 42,
  });

  const [filterParams, setFilterParams] = useState({ sigma_init: 0.2, process_var: 1e-4, meas_var: undefined });

  const generate = async () => {
    setLoading(true);
    setFitData(null);
    try {
      const body = { ...params };
      const resp = await axios.post(`${API}/kalman/simulate`, body);
      setRunId(resp.data.run_id);
      setSimData(resp.data);
    } catch (e) {
      console.error(e);
      alert("Failed to generate synthetic series. Check backend logs.");
    } finally {
      setLoading(false);
    }
  };

  const fit = async () => {
    if (!runId) return;
    setLoading(true);
    try {
      const body = { run_id: runId, ...filterParams };
      const resp = await axios.post(`${API}/kalman/fit`, body);
      setFitData(resp.data);
    } catch (e) {
      console.error(e);
      alert("Failed to run Kalman fit.");
    } finally {
      setLoading(false);
    }
  };

  const trueVol = simData?.true_vol || [];
  const S = simData?.S || [];
  const callObs = simData?.call_price_obs || [];
  const callClean = simData?.call_price_clean || [];
  const estVol = fitData?.est_vol || [];
  const callEst = fitData?.call_price_est || [];

  const volError = useMemo(() => {
    if (!estVol.length || !trueVol.length) return null;
    const n = Math.min(estVol.length, trueVol.length);
    const mse = estVol.slice(0, n).reduce((acc, v, i) => acc + (v - trueVol[i]) ** 2, 0) / n;
    return { mse, rmse: Math.sqrt(mse) };
  }, [estVol.join(","), trueVol.join(",")]);

  return (
    <div className="app-wrap">
      <header className="app-header">
        <div className="brand">
          <div className="logo-dot" />
          <div>
            <h1>PE Options Lab</h1>
            <p>Kalman Filter pricing demo (synthetic)</p>
          </div>
        </div>
        <div className="actions">
          <Badge variant="outline">Backend: {BACKEND_URL ? "connected" : "missing URL"}</Badge>
        </div>
      </header>

      <main className="container">
        <div className="grid">
          <Card className="panel">
            <CardHeader>
              <CardTitle>Generate Synthetic Series</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="form-grid">
                <div className="field">
                  <Label>Steps (n)</Label>
                  <div className="row">
                    <Slider
                      value={[params.n]}
                      min={50}
                      max={1000}
                      step={10}
                      onValueChange={(v) => setParams((p) => ({ ...p, n: v[0] }))}
                      className="flex-1"
                    />
                    <Input
                      type="number"
                      value={params.n}
                      onChange={(e) => setParams((p) => ({ ...p, n: Number(e.target.value) }))}
                      className="w-24"
                    />
                  </div>
                </div>
                <div className="field">
                  <Label>S0</Label>
                  <Input type="number" value={params.S0} onChange={(e) => setParams((p) => ({ ...p, S0: Number(e.target.value) }))} />
                </div>
                <div className="field">
                  <Label>μ (drift)</Label>
                  <Input type="number" step="0.001" value={params.mu} onChange={(e) => setParams((p) => ({ ...p, mu: Number(e.target.value) }))} />
                </div>
                <div className="field">
                  <Label>σ (true vol)</Label>
                  <Input type="number" step="0.001" value={params.sigma_true} onChange={(e) => setParams((p) => ({ ...p, sigma_true: Number(e.target.value) }))} />
                </div>
                <div className="field">
                  <Label>K (strike, blank=S0)</Label>
                  <Input
                    placeholder="defaults to S0"
                    value={params.K ?? ""}
                    onChange={(e) => setParams((p) => ({ ...p, K: e.target.value === "" ? undefined : Number(e.target.value) }))}
                  />
                </div>
                <div className="field">
                  <Label>r (rate)</Label>
                  <Input type="number" step="0.001" value={params.r} onChange={(e) => setParams((p) => ({ ...p, r: Number(e.target.value) }))} />
                </div>
                <div className="field">
                  <Label>T (years)</Label>
                  <Input type="number" step="0.01" value={params.T} onChange={(e) => setParams((p) => ({ ...p, T: Number(e.target.value) }))} />
                </div>
                <div className="field">
                  <Label>Obs noise std</Label>
                  <Input type="number" step="0.01" value={params.obs_noise_std} onChange={(e) => setParams((p) => ({ ...p, obs_noise_std: Number(e.target.value) }))} />
                </div>
                <div className="row gap-8">
                  <Button disabled={loading} onClick={generate} className="btn-primary">
                    <Play size={16} />
                    Generate
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="panel">
            <CardHeader>
              <CardTitle>Run Kalman Filter</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="form-grid">
                <div className="field">
                  <Label>σ init</Label>
                  <Input type="number" step="0.001" value={filterParams.sigma_init} onChange={(e) => setFilterParams((p) => ({ ...p, sigma_init: Number(e.target.value) }))} />
                </div>
                <div className="field">
                  <Label>Process var</Label>
                  <Input type="number" step="0.00001" value={filterParams.process_var} onChange={(e) => setFilterParams((p) => ({ ...p, process_var: Number(e.target.value) }))} />
                </div>
                <div className="field">
                  <Label>Meas var (blank=auto)</Label>
                  <Input
                    placeholder="auto from noise std"
                    value={filterParams.meas_var ?? ""}
                    onChange={(e) => setFilterParams((p) => ({ ...p, meas_var: e.target.value === "" ? undefined : Number(e.target.value) }))}
                  />
                </div>
                <div className="row gap-8">
                  <Button disabled={loading || !simData} onClick={fit} className="btn-accent">
                    <FlaskConical size={16} />
                    Fit Filter
                  </Button>
                </div>
                {!simData && <p className="hint">Generate a series first.</p>}
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="spot" className="tabs">
          <TabsList>
            <TabsTrigger value="spot">
              <LineChart size={16} /> Spot
            </TabsTrigger>
            <TabsTrigger value="opt">Option Price</TabsTrigger>
            <TabsTrigger value="vol">Volatility</TabsTrigger>
          </TabsList>
          <TabsContent value="spot">
            <Card className="panel">
              <CardHeader>
                <CardTitle>Spot Path (S)</CardTitle>
              </CardHeader>
              <CardContent>
                {S.length ? (
                  <MiniLineChart series={[{ name: "S", color: "url(#grad)", values: S }]} />
                ) : (
                  <p className="hint">No data yet. Generate to preview.</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="opt">
            <Card className="panel">
              <CardHeader>
                <CardTitle>Option Prices (Observed vs Clean vs Est)</CardTitle>
              </CardHeader>
              <CardContent>
                {callObs.length ? (
                  <MiniLineChart
                    series={[
                      { name: "Observed", color: "#0ea5a2", values: callObs },
                      { name: "Clean", color: "#64748b", values: callClean },
                      ...(callEst.length ? [{ name: "Est", color: "#22c55e", values: callEst }] : []),
                    ]}
                  />
                ) : (
                  <p className="hint">No data yet. Generate to preview.</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="vol">
            <Card className="panel">
              <CardHeader>
                <CardTitle>Volatility (True vs Estimated)</CardTitle>
              </CardHeader>
              <CardContent>
                {trueVol.length ? (
                  <>
                    <MiniLineChart
                      series={[
                        { name: "True", color: "#64748b", values: trueVol },
                        ...(estVol.length ? [{ name: "Estimated", color: "#16a34a", values: estVol }] : []),
                      ]}
                    />
                    {volError && (
                      <div className="metrics">
                        <div className="metric">
                          <span className="label">RMSE</span>
                          <span className="value">{number(volError.rmse, 6)}</span>
                        </div>
                        <div className="metric">
                          <span className="label">Mean est σ</span>
                          <span className="value">{number(estVol.reduce((a, b) => a + b, 0) / (estVol.length || 1), 6)}</span>
                        </div>
                      </div>
                    )}
                  </>
                ) : (
                  <p className="hint">No data yet. Generate to preview.</p>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </main>

      <footer className="footer">
        <span>Black–Scholes + EKF demo · Stored to MongoDB by run</span>
      </footer>
    </div>
  );
}

export default App;