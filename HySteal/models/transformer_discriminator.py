


import torch
import torch.nn as nn
import torch.nn.functional as F

class MILD2TransformerDiscriminator(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        n_agents: int,
        d_model: int = 128,
        nhead: int = 4,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_agent_id_embedding: bool = True,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.n_agents = int(n_agents)
        self.d_model = int(d_model)

        self.obs_embed = nn.Sequential(
            nn.Linear(self.obs_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Tanh(),
        )
        self.act_embed = nn.Sequential(
            nn.Linear(self.act_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Tanh(),
        )

        self.use_agent_id_embedding = bool(use_agent_id_embedding)
        self.agent_id_embed = nn.Embedding(self.n_agents, self.d_model) if self.use_agent_id_embedding else None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_enc_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_dec_layers)

        self.out_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, obs_tokens: torch.Tensor, act_tokens: torch.Tensor) -> torch.Tensor:
        assert obs_tokens.dim() == 3 and act_tokens.dim() == 3
        B, N, _ = obs_tokens.shape
        assert N == self.n_agents, f"expected N={self.n_agents}, got {N}"

        x_obs = self.obs_embed(obs_tokens)
        x_act = self.act_embed(act_tokens)

        if self.use_agent_id_embedding:
            ids = torch.arange(N, device=obs_tokens.device).view(1, N).expand(B, N)
            x_obs = x_obs + self.agent_id_embed(ids)
            x_act = x_act + self.agent_id_embed(ids)

        memory = self.encoder(x_obs)
        dec_out = self.decoder(tgt=x_act, memory=memory)

        logits = self.out_head(dec_out).squeeze(-1)
        return logits

    @staticmethod
    def reward_from_logits(logits: torch.Tensor) -> torch.Tensor:
        return F.softplus(logits)

    @staticmethod
    def bce_loss(logits_exp: torch.Tensor, logits_gen: torch.Tensor, label_smooth: float = 0.0):
        loss_fn = nn.BCEWithLogitsLoss()
        exp_y = torch.ones_like(logits_exp)
        gen_y = torch.zeros_like(logits_gen)

        if label_smooth > 0:
            exp_y = exp_y * (1.0 - label_smooth)
            gen_y = gen_y + label_smooth

        loss_exp = loss_fn(logits_exp, exp_y)
        loss_gen = loss_fn(logits_gen, gen_y)
        return loss_exp + loss_gen, loss_exp, loss_gen
