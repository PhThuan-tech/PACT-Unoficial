"""Prior Anchoring loss (paper Eq. 9)."""

import torch


def prior_anchoring_loss(model, sigma_rel):
    """L_pa = KL(Q_t(theta) || Q_mlp(theta)) for the current adapter.

    ASSUMPTION: the paper does not state the form of Q. We take isotropic Gaussians of
    equal std centred on the current parameters theta and on their initial values theta_0
    (the MLP copy made by SAI), with the std proportional to the weight scale:
        sigma^2 = sigma_rel^2 * ||theta_0||^2 / n.
    The KL, averaged over the n parameters, is then
        L_pa = ||theta - theta_0||^2 / (2 sigma_rel^2 ||theta_0||^2),
    i.e. a relative drift, normalised like d_j in Eq. (12). sigma_rel is a relative drift
    tolerance: L_pa = 0.5 when the adapter has moved by sigma_rel of its own norm. This keeps
    the loss scale independent of the CLIP weight magnitude and comparable with
    d / tau (see config.py, `pa_sigma_rel`).
    """
    drift, norm = 0.0, 0.0
    for w in model.wrappers:
        params = w.adapter_params(w.adapters[-1])
        for p, p0 in zip(params, w.anchors[-1]):
            drift = drift + (p.float() - p0.float()).pow(2).sum()
            norm = norm + p0.float().pow(2).sum()
    if not torch.is_tensor(drift):
        return torch.zeros(())
    return drift / (norm + 1e-12) / (2 * sigma_rel ** 2)
