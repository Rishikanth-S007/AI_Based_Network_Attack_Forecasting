"""
The core architecture: encoder -> transition model -> risk decoder, plus
the K-step autoregressive rollout that makes this a world model rather
than a classifier with a lookback window.

The one design rule everything else follows: after the real observed
context is consumed, the transition model is only ever fed ITS OWN
previous output. No real traffic is looked at again during a rollout.
That's what `imagine_forward` does, and it's the thing to point to if
anyone asks how this is different from a sequence classifier.
"""
import torch
import torch.nn as nn

N_FEATURES = 10
N_STAGES = 5       # benign, reconnaissance, lateral_movement, command_and_control, exfiltration
LATENT_DIM = 32
HIDDEN_DIM = 64


class Encoder(nn.Module):
    """Compresses one observed traffic window into a latent state."""
    def __init__(self, n_features=N_FEATURES, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransitionModel(nn.Module):
    """Predicts the next latent state from the current one. This is the
    piece that gets rolled forward K times at inference with no new real
    data -- swap this for a Transformer decoder block later if you want,
    the rest of the code doesn't care what's inside `step`.
    """
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.gru = nn.GRUCell(latent_dim, hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(self, z, h):
        h = self.gru(z, h)
        z_next = self.to_latent(h)
        return z_next, h


class RiskDecoder(nn.Module):
    """Reads infiltration probability and MITRE stage off a latent state --
    real or imagined, it doesn't know the difference, which is the point."""
    def __init__(self, latent_dim=LATENT_DIM, n_stages=N_STAGES):
        super().__init__()
        self.risk_head = nn.Linear(latent_dim, 1)
        self.stage_head = nn.Linear(latent_dim, n_stages)

    def forward(self, z):
        risk_logit = self.risk_head(z).squeeze(-1)
        stage_logits = self.stage_head(z)
        return risk_logit, stage_logits


class WorldModel(nn.Module):
    def __init__(self, n_features=N_FEATURES):
        super().__init__()
        self.encoder = Encoder(n_features=n_features)
        self.transition = TransitionModel()
        self.decoder = RiskDecoder()

    def teacher_forced_rollout(self, x_seq):
        """Training pass. At every step, feed the transition model the
        REAL encoded state and ask it to predict the REAL next encoded
        state -- this is the self-supervised P(z_t+1 | z_t) objective,
        separate from the risk/stage labels.

        x_seq: [batch, T, n_features] -> returns
          z_pred_seq: [batch, T-1, latent_dim]  (predictions for steps 1..T-1)
          z_seq:      [batch, T,   latent_dim]  (the encoder's own real latents)
        """
        batch, T, _ = x_seq.shape
        z_seq = self.encoder(x_seq)
        h = self.transition.init_hidden(batch, x_seq.device)
        preds = []
        for t in range(T - 1):
            z_pred, h = self.transition.step(z_seq[:, t], h)
            preds.append(z_pred)
        z_pred_seq = torch.stack(preds, dim=1)
        return z_pred_seq, z_seq

    def imagine_forward(self, x_context, k_steps):
        """Inference-time forecast. Encode the REAL context window(s),
        walk the transition model through them once (teacher forced), then
        roll forward k_steps using ONLY the model's own output -- this is
        the actual forward simulation the PS is asking for.

        x_context: [batch, T_context, n_features] -> returns
          risks:    [batch, k_steps]              sigmoid infiltration probability
          stages:   [batch, k_steps, n_stages]     MITRE stage logits
          imagined: [batch, k_steps, latent_dim]   the imagined latents themselves
        """
        batch, T_context, _ = x_context.shape
        z_seq = self.encoder(x_context)
        h = self.transition.init_hidden(batch, x_context.device)

        z = z_seq[:, 0]
        for t in range(T_context - 1):
            z, h = self.transition.step(z_seq[:, t], h)   # still real data, teacher forced

        risks, stages, imagined = [], [], []
        for _ in range(k_steps):
            z, h = self.transition.step(z, h)              # <-- z is now the model's OWN output
            imagined.append(z)
            risk_logit, stage_logits = self.decoder(z)
            risks.append(torch.sigmoid(risk_logit))
            stages.append(stage_logits)

        return (torch.stack(risks, dim=1),
                torch.stack(stages, dim=1),
                torch.stack(imagined, dim=1))
