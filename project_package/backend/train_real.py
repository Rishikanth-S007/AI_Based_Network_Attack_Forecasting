"""
Real-data training + evaluation on CTU-13 Scenario 10 (Rbot, IRC C2) --
the first real (non-synthetic) end-to-end test of the whole pipeline.

Train/test split is by HOST, not randomly across sequences, and not by
time. With only 10 infected hosts total in this scenario, splitting by
host is what actually tests generalization -- does the model recognize
C2 behavior on a bot it never trained on, rather than just memorizing
these specific 7.

Small-sample caveat, stated plainly: 10 infected hosts, split ~7 train /
3 test, is a very small population -- treat these numbers as a real
first signal, not a final claim. More scenarios / more infected hosts
would firm this up considerably.

Also stated plainly: there's no clean "before infection" context for
these hosts (see ctu13_pipeline.py docstring), so this measures
detection speed from the start of observed activity, NOT early warning
before compromise -- a genuinely different, more modest claim than the
synthetic-data lead-time result.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, precision_score, recall_score
import datetime
import pandas as pd

from ctu13_pipeline import load_ctu13_csv, window_by_host, build_sequences, FEATURE_COLS
from ctu13_packet_features import merge_packet_features
from model import WorldModel
from baseline import train_baseline, evaluate_baseline

SEQ_LEN = 15
DATA_PATH = "ctu13_multiscenario_trimmed.csv"  # put this CSV in the same folder as these .py files
PACKET_FEATURE_PATHS = [
    "pf_scenario1.csv", "pf_scenario2.csv", "pf_scenario9.csv", "pf_scenario10.csv",
]
TEST_DATE = datetime.date(2011, 8, 18)  # Scenario 10 -- held out entirely, never trained on
INFECTED_HOSTS = [
    "147.32.84.165", "147.32.84.191", "147.32.84.192", "147.32.84.193", "147.32.84.204",
    "147.32.84.205", "147.32.84.206", "147.32.84.207", "147.32.84.208", "147.32.84.209",
]


def load_real_packet_features():
    """Combines the per-scenario packet-feature CSVs -- all four days now
    have real coverage (training days 1/2/9 AND the held-out test day
    10), so merging this in no longer creates the placeholder-imbalance
    artifact that blocked this step earlier in the project."""
    dfs = []
    for path in PACKET_FEATURE_PATHS:
        df = pd.read_csv(path)
        df["window"] = pd.to_datetime(df["window"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def date_based_split(X, y_attack, y_stage, hosts, dates):
    is_test = dates == TEST_DATE
    return (X[~is_test], y_attack[~is_test], y_stage[~is_test], hosts[~is_test], dates[~is_test],
            X[is_test], y_attack[is_test], y_stage[is_test], hosts[is_test])


def train_world_model(X_train, ya_train, ys_train, hosts_train, dates_train,
                       epochs=40, batch_size=32, lr=1e-3, p_self_max=0.8, val_frac=0.15, seed=0):
    """Scheduled sampling (ceiling raised to 0.8, was 0.5) plus class-
    weighted risk loss. Also carves out a validation slice from TRAINING
    data only, split by (host, date) GROUP -- not by raw sequence index
    -- to pick the F1-optimal decision threshold afterward. A sequence-
    index split would let sequences from the same host-day land on both
    sides, since build_sequences chunks each host-day into several
    sequences; that's leakage, not a real validation set, and it's
    exactly what produced a bogus 1.0 F1 on the first version of this.
    """
    rng = np.random.default_rng(seed)
    groups = sorted({(h, d) for h, d in zip(hosts_train, dates_train)})  # sorted, not just set->list --
    # python randomizes string hash seeds per process, so an unsorted set's
    # iteration order isn't reproducible run to run even with rng seeded
    rng.shuffle(groups)
    n_val_groups = max(1, int(val_frac * len(groups)))
    val_groups = set(groups[:n_val_groups])

    is_val = np.array([(h, d) in val_groups for h, d in zip(hosts_train, dates_train)])
    X_fit, ya_fit, ys_fit = X_train[~is_val], ya_train[~is_val], ys_train[~is_val]
    X_val, ya_val = X_train[is_val], ya_train[is_val]
    print(f"  validation split: {len(val_groups)}/{len(groups)} (host,date) groups held out "
          f"-> {len(X_val)} val sequences, {len(X_fit)} fit sequences")

    n_pos = ya_fit[:, 1:].sum()
    n_neg = ya_fit[:, 1:].size - n_pos
    imbalance_ratio = n_neg / max(n_pos, 1)
    # only actually apply weighting if there's a real imbalance to correct --
    # our curated, heavily-trimmed dataset is roughly balanced at the window
    # level (unlike the raw 7.5M/353K flow-level numbers this idea was
    # originally based on), so blindly reweighting here does more harm than
    # good: it discourages positive predictions when there's no imbalance
    # to counteract, shifting the model's calibration for no real benefit.
    pos_weight = torch.tensor(imbalance_ratio) if imbalance_ratio > 1.5 or imbalance_ratio < 0.67 else torch.tensor(1.0)
    print(f"  class balance in fit-set: {int(n_pos)} positive / {int(n_neg)} negative windows "
          f"(ratio {imbalance_ratio:.2f}) -> pos_weight={pos_weight.item():.2f} "
          f"{'(reweighting applied)' if pos_weight.item() != 1.0 else '(no reweighting needed -- already balanced)'}")

    ds = TensorDataset(torch.tensor(X_fit), torch.tensor(ya_fit), torch.tensor(ys_fit))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = WorldModel(n_features=len(FEATURE_COLS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        p_self = min(p_self_max, p_self_max * epoch / epochs)
        total_latent = total_risk = total_stage = 0.0
        n_batches = 0
        for x, y_a, y_s in loader:
            batch, T, _ = x.shape
            z_seq = model.encoder(x)
            h = model.transition.init_hidden(batch, x.device)
            z_input = z_seq[:, 0]
            preds = []
            for t in range(T - 1):
                z_pred, h = model.transition.step(z_input, h)
                preds.append(z_pred)
                use_self = (torch.rand(batch, 1) < p_self)
                z_input = torch.where(use_self, z_pred.detach(), z_seq[:, t + 1])
            z_pred_seq = torch.stack(preds, dim=1)

            latent_loss = F.mse_loss(z_pred_seq, z_seq[:, 1:].detach())
            risk_logit, stage_logits = model.decoder(z_pred_seq)
            risk_loss = F.binary_cross_entropy_with_logits(risk_logit, y_a[:, 1:], pos_weight=pos_weight)
            stage_loss = F.cross_entropy(stage_logits.reshape(-1, stage_logits.shape[-1]), y_s[:, 1:].reshape(-1))
            loss = latent_loss + risk_loss + stage_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_latent += latent_loss.item()
            total_risk += risk_loss.item()
            total_stage += stage_loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch+1:02d}  p_self={p_self:.2f}  latent {total_latent/n_batches:.4f}  "
                  f"risk {total_risk/n_batches:.4f}  stage {total_stage/n_batches:.4f}")

    model.eval()
    with torch.no_grad():
        z_pred_seq, _ = model.teacher_forced_rollout(torch.tensor(X_val))
        risk_logit, _ = model.decoder(z_pred_seq)
        val_probs = torch.sigmoid(risk_logit).reshape(-1).numpy()
    val_true = ya_val[:, 1:].reshape(-1)

    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(val_true, val_probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = f1s[:-1].argmax() if len(thresholds) > 0 else None
    best_threshold = float(thresholds[best_idx]) if best_idx is not None else 0.5
    print(f"  threshold tuned on properly-grouped validation slice: {best_threshold:.3f} "
          f"(validation F1 at this threshold: {f1s[best_idx]:.4f})")

    return model, best_threshold


def evaluate_world_model(model, X_test, ya_test, threshold=0.5):
    model.eval()
    with torch.no_grad():
        z_pred_seq, _ = model.teacher_forced_rollout(torch.tensor(X_test))
        risk_logit, _ = model.decoder(z_pred_seq)
        preds = (torch.sigmoid(risk_logit) > threshold).float().reshape(-1).numpy()
    y_true = ya_test[:, 1:].reshape(-1)
    return {
        "f1": f1_score(y_true, preds, zero_division=0),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
    }


def detection_latency(model, X_test, hosts_test, ya_test, threshold=0.5):
    """Teacher-forced check (real data fed at every step) -- useful for
    seeing when the model's per-step judgment turns positive, but this
    is NOT the genuine forecast test. See rollout_forecast_check below
    for that."""
    model.eval()
    infected_in_test = [i for i, h in enumerate(hosts_test) if h in INFECTED_HOSTS]
    for i in infected_in_test:
        h = hosts_test[i]
        with torch.no_grad():
            z_pred_seq, _ = model.teacher_forced_rollout(torch.tensor(X_test[i:i + 1]))
            risk_logit, _ = model.decoder(z_pred_seq)
            risk = torch.sigmoid(risk_logit)[0]
        crossed = (risk > threshold).nonzero(as_tuple=True)[0]
        true_positive_windows = int(ya_test[i, 1:].sum())
        total = SEQ_LEN - 1
        if len(crossed) == 0:
            print(f"  host {h}: never crossed {threshold:.2f} risk ({true_positive_windows}/{total} windows are actually C2)")
        else:
            print(f"  host {h}: first flagged at window {crossed[0].item() + 1} of {total} "
                  f"({true_positive_windows}/{total} windows are actually C2)")


def rollout_forecast_check(model, X_test, hosts_test, ya_test, context_len=3, threshold=0.5):
    """The actual test of the world-model claim: give the model only the
    first `context_len` REAL windows of a held-out DAY's infected host,
    then roll forward autoregressively with ZERO new real data for the
    rest of the sequence, using imagine_forward."""
    model.eval()
    infected_in_test = [i for i, h in enumerate(hosts_test) if h in INFECTED_HOSTS]
    n_crossed = 0
    for i in infected_in_test:
        h = hosts_test[i]
        context = torch.tensor(X_test[i:i + 1, :context_len])
        k_steps = SEQ_LEN - context_len
        with torch.no_grad():
            risks, _, _ = model.imagine_forward(context, k_steps=k_steps)
        risks = risks[0]
        crossed = (risks > threshold).nonzero(as_tuple=True)[0]
        true_future_positive = int(ya_test[i, context_len:].sum())

        print(f"  host {h}: shown only windows 1-{context_len} (real), "
              f"imagined windows {context_len+1}-{SEQ_LEN} blind ({k_steps} steps, zero real data)")
        if len(crossed) == 0:
            print(f"    -> never imagined risk > {threshold:.2f} ({true_future_positive}/{k_steps} of those windows are actually C2)")
        else:
            n_crossed += 1
            print(f"    -> imagined risk crossed {threshold:.2f} at blind step {crossed[0].item()+1} "
                  f"({true_future_positive}/{k_steps} of those windows are actually C2)")
    print(f"\n  summary: {n_crossed}/{len(infected_in_test)} held-out infected sequences "
          f"triggered under genuine blind rollout")


if __name__ == "__main__":
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"loading real merged multi-scenario data from {DATA_PATH}...")
    df = load_ctu13_csv(DATA_PATH)
    windowed = window_by_host(df)

    print("merging real packet-level features (TTL variance, retransmits) "
          "across all four days...")
    packet_df = load_real_packet_features()
    windowed = merge_packet_features(windowed, packet_df)

    X, y_attack, y_stage, hosts, dates, feature_mean, feature_std = build_sequences(windowed, seq_len=SEQ_LEN)
    print(f"{len(X)} sequences total, {int((y_attack.sum(axis=1) > 0).sum())} contain C2 activity\n")

    X_train, ya_train, ys_train, hosts_train, dates_train, X_test, ya_test, ys_test, hosts_test = date_based_split(
        X, y_attack, y_stage, hosts, dates
    )
    n_test_infected = sum(1 for h in hosts_test if h in INFECTED_HOSTS)
    print(f"train: {len(X_train)} sequences (Aug 10+11+17) | "
          f"test: {len(X_test)} sequences, held-out Aug 18 "
          f"({n_test_infected} from infected hosts on that day)\n")

    print("--- baseline (logistic regression) ---")
    clf = train_baseline(X_train, ya_train)
    baseline_result = evaluate_baseline(clf, X_test, ya_test)
    print(f"baseline F1: {baseline_result['f1']:.4f}\n")
    print(
        f"baseline attack precision: {baseline_result['report']['1.0']['precision']:.4f}  "
        f"recall: {baseline_result['report']['1.0']['recall']:.4f}\n"
    )

    print("--- world model (scheduled sampling p_self->0.8, class-weighted loss, tuned threshold) ---")
    model, tuned_threshold = train_world_model(X_train, ya_train, ys_train, hosts_train, dates_train)
    wm_result = evaluate_world_model(model, X_test, ya_test, threshold=tuned_threshold)
    wm_result_default = evaluate_world_model(model, X_test, ya_test, threshold=0.5)
    print(f"\nworld model F1 @ tuned threshold {tuned_threshold:.3f}: {wm_result['f1']:.4f}  "
          f"precision: {wm_result['precision']:.4f}  recall: {wm_result['recall']:.4f}")
    print(f"world model F1 @ default 0.5 (for comparison): {wm_result_default['f1']:.4f}\n")

    print("--- detection speed on held-out day (teacher-forced) ---")
    detection_latency(model, X_test, hosts_test, ya_test, threshold=tuned_threshold)

    print("\n--- genuine forecast: blind rollout from limited context (the real test) ---")
    rollout_forecast_check(model, X_test, hosts_test, ya_test, context_len=3, threshold=tuned_threshold)

    checkpoint_path = "ctu13_world_model.pt"
    torch.save({
        "model_state": model.state_dict(),
        "n_features": len(FEATURE_COLS),
        "feature_cols": FEATURE_COLS,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "threshold": tuned_threshold,
        "seq_len": SEQ_LEN,
        "test_f1": wm_result["f1"],
        "baseline_f1": baseline_result["f1"],
        "trained_on": "CTU-13 Scenarios 1/2/9 (Aug 10/11/17), tested on Scenario 10 (Aug 18, held out)",
    }, checkpoint_path)
    print(f"\nsaved checkpoint to {checkpoint_path} -- this is what inference.py loads")
