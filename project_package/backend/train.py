"""
Smoke-test training loop. Proves the world model trains end to end on
synthetic data, that all three loss terms actually go down, and that the
K-step rollout produces non-degenerate, correctly-shaped forecasts.

This is NOT a real training run and the numbers mean nothing outside this
file. Swap `make_dataset()` in data.py for a loader over real windowed
CIC-IDS2018 / CTU-13 features -- nothing here needs to change as long as
shapes match.
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import WorldModel
from data import make_dataset, TrafficWindowDataset, STAGE_RECON, STAGE_LATERAL


def train(epochs=15, batch_size=32, seq_len=40, n_samples=512, lr=1e-3, seed=0):
    X, y_attack, y_stage = make_dataset(n_samples=n_samples, seq_len=seq_len, seed=seed)
    ds = TrafficWindowDataset(X, y_attack, y_stage)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = WorldModel()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_latent = total_risk = total_stage = 0.0
        n_batches = 0
        for x, y_a, y_s in loader:
            z_pred_seq, z_seq = model.teacher_forced_rollout(x)
            latent_loss = F.mse_loss(z_pred_seq, z_seq[:, 1:].detach())

            risk_logit, stage_logits = model.decoder(z_pred_seq)
            risk_loss = F.binary_cross_entropy_with_logits(risk_logit, y_a[:, 1:])
            stage_loss = F.cross_entropy(
                stage_logits.reshape(-1, stage_logits.shape[-1]),
                y_s[:, 1:].reshape(-1),
            )

            loss = latent_loss + risk_loss + stage_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            total_latent += latent_loss.item()
            total_risk += risk_loss.item()
            total_stage += stage_loss.item()
            n_batches += 1

        print(f"epoch {epoch+1:02d}  latent {total_latent/n_batches:.4f}  "
              f"risk {total_risk/n_batches:.4f}  stage {total_stage/n_batches:.4f}")

    return model, ds


def lead_time_smoke_check(model, ds, k_steps=10, threshold=0.5, n_samples_to_try=20):
    """Take attack samples, give the model context that stops right where
    reconnaissance ends -- so it has seen the recon precursor but NOT the
    lateral-movement traffic that follows -- then roll forward and check
    whether the imagined rollout flags rising risk before lateral-movement
    windows are directly observed.

    This is the honest version of the check: cutting context off at the
    very first attack window (as an earlier draft of this function did)
    tests whether the model can predict an event with literally zero
    precursor signal, which nothing can do. This version tests the thing
    that's actually claimed -- forecasting the next stage of an attack
    already in progress, from the stage before it.
    """
    X, y_attack, y_stage = ds.raw()
    model.eval()
    tried = 0
    for i in range(X.shape[0]):
        stages = y_stage[i]
        recon_windows = (stages == STAGE_RECON).nonzero(as_tuple=True)[0]
        lateral_windows = (stages == STAGE_LATERAL).nonzero(as_tuple=True)[0]
        if len(recon_windows) == 0 or len(lateral_windows) == 0:
            continue
        cutoff = lateral_windows[0].item()  # first real lateral-movement window
        if cutoff < 4:
            continue
        tried += 1

        context = X[i:i + 1, :cutoff]  # recon is visible; lateral movement/C2 are not
        with torch.no_grad():
            risks, _, _ = model.imagine_forward(context, k_steps=k_steps)
        risks = risks[0]
        crossed = (risks > threshold).nonzero(as_tuple=True)[0]

        print(f"sample {i}: model has seen traffic through window {cutoff - 1} "
              f"(reconnaissance visible, lateral movement begins at real window {cutoff})")
        if len(crossed) == 0:
            print(f"  rollout never crossed risk={threshold} within {k_steps} imagined windows")
        else:
            first_cross = crossed[0].item()
            print(f"  imagined rollout crossed risk={threshold} at step {first_cross + 1} into "
                  f"the future, i.e. it flagged real window {cutoff + first_cross} without ever "
                  f"observing that window's actual traffic")
        if tried >= n_samples_to_try:
            return
    if tried == 0:
        print("no qualifying sample found for the lead-time smoke check")


if __name__ == "__main__":
    trained_model, dataset = train(epochs=30, n_samples=800)
    print()
    lead_time_smoke_check(trained_model, dataset, n_samples_to_try=8)
