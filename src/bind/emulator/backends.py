"""Regression backends: 30 params → PCA-latent of one statistic, with uncertainty.

All three share the interface ``fit(X, Z) → self`` / ``predict(X) → (mean, std)``
over the unit-cube parameters ``X`` ``(n, 30)`` and the compressed latent ``Z``
``(n, k)`` from :class:`~bind.emulator.transforms.StatCompressor`:

* :class:`MLPEnsemble` — an ensemble of small torch MLPs; the spread across
  members is the epistemic 1σ.  CPU-fast, the default.
* :class:`GPBackend`   — an independent scikit-learn Gaussian process per latent
  component (ARD kernel); analytic 1σ.  The classic BCM-emulator backend.
* :class:`FlowBackend` — a conditional normalizing flow (zuko) modelling
  ``p(Z | X)``; ``predict`` returns the sample mean/std and ``sample`` draws mock
  latents (→ mock statistic vectors).  GPU-trained.

Backends standardize ``Z`` internally; state is returned as a picklable dict
(torch ``state_dict`` tensors / fitted sklearn objects) saved inside the Emulator
bundle via ``torch.save``.
"""

from __future__ import annotations

import numpy as np


def make_backend(name: str, **kw):
    name = name.lower()
    if name in ("mlp", "mlp_ensemble", "ensemble"):
        return MLPEnsemble(**kw)
    if name == "auto":
        import torch
        name = "gpgpu" if torch.cuda.is_available() else "gp"
    if name in ("gp", "gaussian_process"):
        return GPBackend(**kw)
    if name in ("gpgpu", "gp_gpu", "gpytorch"):
        return GPTorchBackend(**kw)
    if name in ("flow", "nflow", "normalizing_flow"):
        return FlowBackend(**kw)
    raise ValueError(f"unknown backend {name!r} (mlp|gp|gpgpu|flow|auto)")


# ── MLP ensemble ──────────────────────────────────────────────────────────────

class MLPEnsemble:
    kind = "mlp"

    def __init__(self, n_models: int = 8, hidden: int = 128, depth: int = 3,
                 epochs: int = 1500, lr: float = 3e-3, weight_decay: float = 1e-5,
                 dropout: float = 0.0, bootstrap: bool = False, device: str | None = None,
                 seed: int = 0, verbose: bool = False, **_):
        # `**_` swallows kwargs meant for a sibling backend (e.g. when backend="auto"
        # resolves across gp/gpgpu) so cross-backend defaults never raise.
        # bootstrap defaults OFF: with ~100 design points, resampling starves each
        # net (~63% unique) and the ensemble mean collapses toward the prior mean.
        # Seed-varied weight init gives the epistemic spread instead.
        self.cfg = dict(n_models=n_models, hidden=hidden, depth=depth, epochs=epochs,
                        lr=lr, weight_decay=weight_decay, dropout=dropout,
                        bootstrap=bootstrap, seed=seed)
        self.device = device
        self.verbose = verbose
        self.models: list = []
        self.zmean_ = None
        self.zstd_ = None
        self.in_dim = 0
        self.out_dim = 0

    def _build(self):
        import torch.nn as nn
        c = self.cfg
        layers, d = [], self.in_dim
        for _ in range(c["depth"]):
            layers += [nn.Linear(d, c["hidden"]), nn.SiLU()]
            if c["dropout"] > 0:
                layers += [nn.Dropout(c["dropout"])]
            d = c["hidden"]
        layers += [nn.Linear(d, self.out_dim)]
        return nn.Sequential(*layers)

    def fit(self, X, Z):
        import torch
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        X = np.asarray(X, float)
        Z = np.asarray(Z, float)
        self.in_dim, self.out_dim = X.shape[1], Z.shape[1]
        self.zmean_, self.zstd_ = Z.mean(0), Z.std(0) + 1e-8
        Zs = (Z - self.zmean_) / self.zstd_
        Xt = torch.tensor(X, dtype=torch.float32, device=dev)
        Zt = torch.tensor(Zs, dtype=torch.float32, device=dev)
        c = self.cfg
        self.models = []
        for m in range(c["n_models"]):
            torch.manual_seed(c["seed"] + m)
            if c["bootstrap"]:
                idx = np.random.default_rng(c["seed"] + m).integers(0, len(X), len(X))
            else:
                idx = np.arange(len(X))                     # full data, init-varied
            net = self._build().to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=c["lr"],
                                    weight_decay=c["weight_decay"])
            xb, zb = Xt[idx], Zt[idx]
            net.train()
            for ep in range(c["epochs"]):
                opt.zero_grad()
                loss = ((net(xb) - zb) ** 2).mean()
                loss.backward()
            opt.step()
            net.eval()
            self.models.append(net)
            if self.verbose:
                print(f"    mlp {m}: final loss {float(loss):.4g}")
        self._dev = dev
        return self

    def predict(self, X):
        import torch
        dev = getattr(self, "_dev", None) or self.device or "cpu"
        Xt = torch.tensor(np.asarray(X, float), dtype=torch.float32, device=dev)
        with torch.no_grad():
            preds = np.stack([m(Xt).cpu().numpy() for m in self.models])   # (M, n, k)
        mean = preds.mean(0) * self.zstd_ + self.zmean_
        std = preds.std(0) * self.zstd_
        return mean, std

    def state_dict(self) -> dict:
        return {"kind": self.kind, "cfg": self.cfg, "in_dim": self.in_dim,
                "out_dim": self.out_dim, "zmean": self.zmean_, "zstd": self.zstd_,
                "models": [{k: v.cpu() for k, v in m.state_dict().items()}
                           for m in self.models]}

    @classmethod
    def from_state(cls, s: dict) -> "MLPEnsemble":
        obj = cls(**s["cfg"])
        obj.in_dim, obj.out_dim = s["in_dim"], s["out_dim"]
        obj.zmean_, obj.zstd_ = s["zmean"], s["zstd"]
        obj.models = []
        for sd in s["models"]:
            net = obj._build()
            net.load_state_dict(sd)
            net.eval()
            obj.models.append(net)
        obj._dev = "cpu"
        return obj


# ── Gaussian process (per-component) ──────────────────────────────────────────

class GPBackend:
    kind = "gp"

    def __init__(self, n_restarts: int = 2, alpha: float = 1e-6, seed: int = 0, **_):
        self.cfg = dict(n_restarts=n_restarts, alpha=alpha, seed=seed)
        self.gps: list = []
        self.zmean_ = None
        self.zstd_ = None

    def fit(self, X, Z):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        X = np.asarray(X, float)
        Z = np.asarray(Z, float)
        self.zmean_, self.zstd_ = Z.mean(0), Z.std(0) + 1e-8
        Zs = (Z - self.zmean_) / self.zstd_
        self.gps = []
        for j in range(Zs.shape[1]):
            kernel = (ConstantKernel(1.0, (1e-2, 1e2))
                      * RBF(length_scale=np.ones(X.shape[1]),
                            length_scale_bounds=(1e-2, 1e2))
                      + WhiteKernel(1e-3, (1e-6, 1e0)))
            gp = GaussianProcessRegressor(kernel=kernel, alpha=self.cfg["alpha"],
                                          n_restarts_optimizer=self.cfg["n_restarts"],
                                          normalize_y=False, random_state=self.cfg["seed"])
            gp.fit(X, Zs[:, j])
            self.gps.append(gp)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        means, stds = [], []
        for gp in self.gps:
            m, s = gp.predict(X, return_std=True)
            means.append(m)
            stds.append(s)
        mean = np.stack(means, 1) * self.zstd_ + self.zmean_
        std = np.stack(stds, 1) * self.zstd_
        return mean, std

    def state_dict(self) -> dict:
        return {"kind": self.kind, "cfg": self.cfg, "gps": self.gps,
                "zmean": self.zmean_, "zstd": self.zstd_}

    @classmethod
    def from_state(cls, s: dict) -> "GPBackend":
        obj = cls(**s["cfg"])
        obj.gps = s["gps"]
        obj.zmean_, obj.zstd_ = s["zmean"], s["zstd"]
        return obj


# ── exact GP on GPU (gpytorch, batched over PCA components) ────────────────────

class GPTorchBackend:
    """GPU-accelerated exact GP — the recommended GP backend, batched over PCs.

    Identical model to :class:`GPBackend` (one independent ARD-RBF exact GP per PCA
    component) but fit jointly as a single batched gpytorch model, so all components
    and the marginal-likelihood optimization run on the GPU.  Use this on a GPU node
    (``run_emulator_train.sh``); ``backend="auto"`` selects it when CUDA is present.
    """

    kind = "gpgpu"

    def __init__(self, epochs: int = 400, lr: float = 0.1, device: str | None = None,
                 seed: int = 0, jitter: float = 1e-4, verbose: bool = False, **_):
        self.cfg = dict(epochs=epochs, lr=lr, seed=seed, jitter=jitter)
        self.device = device
        self.verbose = verbose
        self.model = self.likelihood = None
        self.train_x = self.train_y = None
        self.zmean_ = self.zstd_ = None
        self.in_dim = self.out_dim = 0

    def _build(self, train_x, train_y, dev):
        import gpytorch
        import torch
        k, d = self.out_dim, self.in_dim
        bshape = torch.Size([k])

        class _Batch(gpytorch.models.ExactGP):
            def __init__(self, x, y, lik):
                super().__init__(x, y, lik)
                self.mean = gpytorch.means.ConstantMean(batch_shape=bshape)
                self.covar = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(batch_shape=bshape, ard_num_dims=d),
                    batch_shape=bshape)

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(
                    self.mean(x), self.covar(x))

        lik = gpytorch.likelihoods.GaussianLikelihood(batch_shape=bshape).to(dev)
        model = _Batch(train_x, train_y, lik).to(dev)
        return model, lik

    def fit(self, X, Z):
        import gpytorch
        import torch
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        X = np.asarray(X, float)
        Z = np.asarray(Z, float)
        self.in_dim, self.out_dim = X.shape[1], Z.shape[1]
        self.zmean_, self.zstd_ = Z.mean(0), Z.std(0) + 1e-8
        Zs = ((Z - self.zmean_) / self.zstd_).T                  # (k, n)
        k = self.out_dim
        tx = torch.tensor(X, dtype=torch.float32, device=dev).unsqueeze(0).expand(k, -1, -1)
        ty = torch.tensor(Zs, dtype=torch.float32, device=dev).contiguous()
        torch.manual_seed(self.cfg["seed"])
        self.model, self.likelihood = self._build(tx.contiguous(), ty, dev)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg["lr"])
        self.model.train()
        self.likelihood.train()
        with gpytorch.settings.cholesky_jitter(self.cfg["jitter"]):
            for ep in range(self.cfg["epochs"]):
                opt.zero_grad()
                loss = -mll(self.model(tx), ty).sum()
                loss.backward()
                opt.step()
                if self.verbose and ep % 100 == 0:
                    print(f"    gpgpu ep {ep}: mll {-float(loss):.4g}")
        self.model.eval()
        self.likelihood.eval()
        self.train_x, self.train_y = tx.contiguous(), ty
        self._dev = dev
        return self

    def predict(self, X):
        import gpytorch
        import torch
        dev = getattr(self, "_dev", None) or self.device or "cpu"
        k = self.out_dim
        xt = torch.tensor(np.asarray(X, float), dtype=torch.float32,
                          device=dev).unsqueeze(0).expand(k, -1, -1).contiguous()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
                gpytorch.settings.cholesky_jitter(self.cfg["jitter"]):
            pred = self.likelihood(self.model(xt))
            mean = pred.mean.cpu().numpy().T                     # (m, k)
            std = pred.stddev.cpu().numpy().T
        return mean * self.zstd_ + self.zmean_, std * self.zstd_

    def state_dict(self) -> dict:
        return {"kind": self.kind, "cfg": self.cfg, "in_dim": self.in_dim,
                "out_dim": self.out_dim, "zmean": self.zmean_, "zstd": self.zstd_,
                "model": {k: v.cpu() for k, v in self.model.state_dict().items()},
                "likelihood": {k: v.cpu() for k, v in self.likelihood.state_dict().items()},
                "train_x": self.train_x.cpu(), "train_y": self.train_y.cpu()}

    @classmethod
    def from_state(cls, s: dict) -> "GPTorchBackend":
        obj = cls(**s["cfg"])
        obj.in_dim, obj.out_dim = s["in_dim"], s["out_dim"]
        obj.zmean_, obj.zstd_ = s["zmean"], s["zstd"]
        obj.train_x, obj.train_y = s["train_x"], s["train_y"]
        obj.model, obj.likelihood = obj._build(obj.train_x, obj.train_y, "cpu")
        obj.model.load_state_dict(s["model"])
        obj.likelihood.load_state_dict(s["likelihood"])
        obj.model.eval()
        obj.likelihood.eval()
        obj._dev = "cpu"
        return obj


# ── conditional normalizing flow (zuko) ───────────────────────────────────────

class FlowBackend:
    kind = "flow"

    def __init__(self, transforms: int = 4, hidden: int = 128, bins: int = 8,
                 epochs: int = 1500, lr: float = 2e-3, weight_decay: float = 1e-5,
                 n_samples: int = 256, device: str | None = None, seed: int = 0,
                 verbose: bool = False, **_):
        self.cfg = dict(transforms=transforms, hidden=hidden, bins=bins, epochs=epochs,
                        lr=lr, weight_decay=weight_decay, n_samples=n_samples, seed=seed)
        self.device = device
        self.verbose = verbose
        self.flow = None
        self.xmean_ = self.xstd_ = self.zmean_ = self.zstd_ = None
        self.in_dim = self.out_dim = 0

    def _build(self, dev):
        import zuko
        c = self.cfg
        # neural spline flow conditioned on the (standardized) parameters
        return zuko.flows.NSF(features=self.out_dim, context=self.in_dim,
                              transforms=c["transforms"], bins=c["bins"],
                              hidden_features=[c["hidden"]] * 2).to(dev)

    def fit(self, X, Z):
        import torch
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        X = np.asarray(X, float)
        Z = np.asarray(Z, float)
        self.in_dim, self.out_dim = X.shape[1], Z.shape[1]
        self.xmean_, self.xstd_ = X.mean(0), X.std(0) + 1e-8
        self.zmean_, self.zstd_ = Z.mean(0), Z.std(0) + 1e-8
        Xs = (X - self.xmean_) / self.xstd_
        Zs = (Z - self.zmean_) / self.zstd_
        torch.manual_seed(self.cfg["seed"])
        self.flow = self._build(dev)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=dev)
        Zt = torch.tensor(Zs, dtype=torch.float32, device=dev)
        opt = torch.optim.AdamW(self.flow.parameters(), lr=self.cfg["lr"],
                                weight_decay=self.cfg["weight_decay"])
        self.flow.train()
        for ep in range(self.cfg["epochs"]):
            opt.zero_grad()
            loss = -self.flow(Xt).log_prob(Zt).mean()
            loss.backward()
            opt.step()
            if self.verbose and ep % 200 == 0:
                print(f"    flow ep {ep}: nll {float(loss):.4g}")
        self.flow.eval()
        self._dev = dev
        return self

    def _ctx(self, X):
        import torch
        dev = getattr(self, "_dev", None) or self.device or "cpu"
        Xs = (np.asarray(X, float) - self.xmean_) / self.xstd_
        return torch.tensor(Xs, dtype=torch.float32, device=dev), dev

    def sample(self, X, n: int | None = None):
        """Draw ``n`` mock latents per parameter row → ``(n, m, k)`` physical-latent."""
        import torch
        n = n or self.cfg["n_samples"]
        ctx, _ = self._ctx(X)
        with torch.no_grad():
            s = self.flow(ctx).sample((n,))            # (n, m, k) standardized
        return s.cpu().numpy() * self.zstd_ + self.zmean_

    def predict(self, X):
        s = self.sample(X, self.cfg["n_samples"])      # (n, m, k)
        return s.mean(0), s.std(0)

    def state_dict(self) -> dict:
        return {"kind": self.kind, "cfg": self.cfg, "in_dim": self.in_dim,
                "out_dim": self.out_dim, "xmean": self.xmean_, "xstd": self.xstd_,
                "zmean": self.zmean_, "zstd": self.zstd_,
                "flow": {k: v.cpu() for k, v in self.flow.state_dict().items()}}

    @classmethod
    def from_state(cls, s: dict) -> "FlowBackend":
        obj = cls(**s["cfg"])
        obj.in_dim, obj.out_dim = s["in_dim"], s["out_dim"]
        obj.xmean_, obj.xstd_ = s["xmean"], s["xstd"]
        obj.zmean_, obj.zstd_ = s["zmean"], s["zstd"]
        obj.flow = obj._build("cpu")
        obj.flow.load_state_dict(s["flow"])
        obj.flow.eval()
        obj._dev = "cpu"
        return obj


BACKEND_REGISTRY = {"mlp": MLPEnsemble, "gp": GPBackend, "gpgpu": GPTorchBackend,
                    "flow": FlowBackend}
