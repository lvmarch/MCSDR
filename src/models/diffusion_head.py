# src/models/diffusion_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from thop import profile
from src.losses.other_loss import wasserstein_loss_1d
from src.losses.other_loss import DynamicFocusingL1Loss
from tqdm import tqdm  # <-- 添加 tqdm 用于 DDIM 循环


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        # Corrected a potential division by zero if half_dim is 1
        denominator = half_dim - 1 if half_dim > 1 else 1
        emb = math.log(10000) / denominator
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class NoisePredictorMLP(nn.Module):
    def __init__(self, condition_dim, time_emb_dim, hidden_dim=256, **kwargs):
        super().__init__()
        total_input_dim = condition_dim + time_emb_dim + 1  # y_dim is 1
        self.mlp = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, noisy_y, t_emb, condition_x):
        if noisy_y.ndim == 1:
            noisy_y = noisy_y.unsqueeze(1)
        x_combined = torch.cat([noisy_y, t_emb, condition_x], dim=1)
        return self.mlp(x_combined)


class LegacyCrossAttentionPredictor(nn.Module):
    def __init__(self, condition_dim, time_emb_dim, embed_dim=256, n_head=8, **kwargs):
        super().__init__()
        self.map_y_t = nn.Linear(1 + time_emb_dim, embed_dim)
        self.map_cond = nn.Linear(condition_dim, embed_dim)
        self.cross_attention = nn.MultiheadAttention(embed_dim, n_head, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.final_layer = nn.Linear(embed_dim, 1)

    def forward(self, noisy_y, t_emb, condition_x):
        if noisy_y.ndim == 1:
            noisy_y = noisy_y.unsqueeze(1)

        q_in = torch.cat([noisy_y, t_emb], dim=1)
        q = self.map_y_t(q_in).unsqueeze(1)
        kv = self.map_cond(condition_x).unsqueeze(1)

        attn_output, _ = self.cross_attention(query=q, key=kv, value=kv)
        h = self.norm1(q + attn_output)

        ffn_output = self.ffn(h)
        h = self.norm2(h + ffn_output)

        return self.final_layer(h.squeeze(1))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, n_head):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4), nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, context):
        attn_output, _ = self.cross_attention(query=query, key=context, value=context)
        query = self.norm1(query + attn_output)
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        return output


class DeepCrossAttentionPredictor(nn.Module):
    def __init__(self, condition_dim, time_emb_dim, hidden_dim=256, n_head=8, depth=4, **kwargs):
        super().__init__()

        self.query_encoder = nn.Linear(1 + time_emb_dim, hidden_dim)
        self.context_encoder = nn.Linear(condition_dim, hidden_dim)

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim=hidden_dim, n_head=n_head)
            for _ in range(depth)
        ])

        self.out_layer = nn.Linear(hidden_dim, 1)

    def forward(self, noisy_y, t_emb, condition_x):
        if noisy_y.ndim == 1:
            noisy_y = noisy_y.unsqueeze(1)

        q_in = torch.cat([noisy_y, t_emb], dim=1)
        q = self.query_encoder(q_in).unsqueeze(1)
        context = self.context_encoder(condition_x).unsqueeze(1)

        for layer in self.layers:
            q = layer(q, context)

        return self.out_layer(q.squeeze(1))


PREDICTOR_MAP = {
    "MLP": NoisePredictorMLP,
    "LegacyAttention": LegacyCrossAttentionPredictor,
    "DeepAttention": DeepCrossAttentionPredictor,
}


class DiffusionRegressorHead(nn.Module):
    def __init__(self, condition_dim, n_diffusion_steps=1000, time_emb_dim=32, predictor_config=None,
                 tabular_config=None):
        super().__init__()
        self.condition_dim = condition_dim
        self.n_diffusion_steps = n_diffusion_steps

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU()
        )

        self.use_tabular_data = tabular_config is not None
        self.tabular_encoder = None
        final_condition_dim = self.condition_dim

        if self.use_tabular_data:
            print("INFO: Diffusion head is configured to use tabular data.")
            tabular_dim = tabular_config.get("dim")
            tabular_hidden_dim = tabular_config.get("hidden_dim")
            if not isinstance(tabular_dim, int) or not isinstance(tabular_hidden_dim, int):
                raise ValueError("tabular_config must contain integer 'dim' and 'hidden_dim'")

            self.tabular_encoder = nn.Sequential(
                nn.Linear(tabular_dim, tabular_hidden_dim),
                nn.ReLU(),
                nn.Linear(tabular_hidden_dim, tabular_hidden_dim)
            )
            final_condition_dim += tabular_hidden_dim

        if predictor_config is None:
            predictor_config = {"name": "DeepAttention"}

        predictor_name = predictor_config["name"]
        predictor_params = predictor_config.get("params", {})

        if predictor_name not in PREDICTOR_MAP:
            raise ValueError(f"Unknown predictor name: {predictor_name}. Available: {list(PREDICTOR_MAP.keys())}")

        PredictorClass = PREDICTOR_MAP[predictor_name]

        self.noise_predictor = PredictorClass(
            condition_dim=final_condition_dim,
            time_emb_dim=time_emb_dim,
            **predictor_params
        )

        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        res = arr.to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def q_sample(self, y0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(y0)
        if y0.ndim == 1:
            y0 = y0.unsqueeze(1)
        sqrt_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, y0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, y0.shape)
        return sqrt_alphas_cumprod_t * y0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, combined_condition, y0):
        batch_size = y0.shape[0]
        if y0.ndim == 1:
            y0 = y0.unsqueeze(1)

        t = torch.randint(0, self.n_diffusion_steps, (batch_size,), device=y0.device).long()
        noise = torch.randn_like(y0)
        yt = self.q_sample(y0=y0, t=t, noise=noise)
        t_emb = self.time_mlp(t)

        predicted_noise = self.noise_predictor(noisy_y=yt, t_emb=t_emb, condition_x=combined_condition)

        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def p_sample(self, yt, t_scalar, combined_condition):
        batch_size_expanded = yt.shape[0]
        t_tensor = torch.full((batch_size_expanded,), t_scalar, device=yt.device, dtype=torch.long)

        betas_t = self._extract_into_tensor(self.betas, t_tensor, yt.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t_tensor,
                                                                    yt.shape)
        sqrt_alphas_t = self._extract_into_tensor(torch.sqrt(1. - self.betas), t_tensor, yt.shape)

        t_emb = self.time_mlp(t_tensor)

        if yt.ndim == 1:
            yt = yt.unsqueeze(1)

        predicted_noise = self.noise_predictor(noisy_y=yt, t_emb=t_emb, condition_x=combined_condition)

        if predicted_noise.ndim == 1:
            predicted_noise = predicted_noise.unsqueeze(1)

        model_mean = (1 / sqrt_alphas_t) * (yt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t)

        if t_scalar == 0:
            return model_mean
        else:
            posterior_variance_t = self._extract_into_tensor(self.posterior_variance, t_tensor, yt.shape)
            noise = torch.randn_like(yt)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_ddim(self, yt, t_scalar, t_prev_scalar, combined_condition, eta):
        batch_size_expanded = yt.shape[0]
        t_tensor = torch.full((batch_size_expanded,), t_scalar, device=yt.device, dtype=torch.long)
        t_prev_tensor = torch.full((batch_size_expanded,), t_prev_scalar, device=yt.device, dtype=torch.long)

        t_emb = self.time_mlp(t_tensor)

        if yt.ndim == 1:
            yt = yt.unsqueeze(1)

        predicted_noise = self.noise_predictor(noisy_y=yt, t_emb=t_emb, condition_x=combined_condition)

        if predicted_noise.ndim == 1:
            predicted_noise = predicted_noise.unsqueeze(1)

        sqrt_recip_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t_tensor, yt.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t_tensor, yt.shape)
        pred_y0 = sqrt_recip_alphas_cumprod_t * yt - sqrt_recipm1_alphas_cumprod_t * predicted_noise

        if t_prev_scalar < 0:
            return pred_y0  # 已在 t=0

        alphas_cumprod_prev_t = self._extract_into_tensor(self.alphas_cumprod, t_prev_tensor, yt.shape)

        sigma_t = eta * torch.sqrt(
            (1 - alphas_cumprod_prev_t) / (1 - self.alphas_cumprod[t_scalar]) * self.betas[t_scalar]
        )

        dir_yt = torch.sqrt(1. - alphas_cumprod_prev_t - sigma_t ** 2) * predicted_noise

        noise = torch.randn_like(yt) * sigma_t
        y_prev = torch.sqrt(alphas_cumprod_prev_t) * pred_y0 + dir_yt + noise

        return y_prev


    @torch.no_grad()
    def p_sample_loop(self, combined_condition, n_samples_per_condition=1, return_all_timesteps=False,
                      sample_every_n_steps=100,
                      sampler='ddpm', ddim_steps=1000, ddim_eta=0.0):  # <-- 添加新参数

        device = self.betas.device
        original_batch_size = combined_condition.shape[0]

        total_parallel_samples = original_batch_size * n_samples_per_condition

        y_t = torch.randn((total_parallel_samples, 1), device=device)

        condition_x_expanded = combined_condition.repeat_interleave(n_samples_per_condition, dim=0)

        intermediate_y_t_steps = []
        recorded_timesteps_list = []

        if return_all_timesteps:
            intermediate_y_t_steps.append(y_t.clone().cpu().reshape(original_batch_size, n_samples_per_condition, 1))
            recorded_timesteps_list.append(self.n_diffusion_steps)

        if sampler == 'ddim':
            times = torch.linspace(-1, self.n_diffusion_steps - 1, ddim_steps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))  # [(1000, 900), (900, 800), ...]

            desc = "DDIM Sampling"
            # for t, t_prev in tqdm(time_pairs, desc=desc, leave=False, dynamic_ncols=True):
            for t, t_prev in time_pairs:
                y_t = self.p_sample_ddim(y_t, t, t_prev, condition_x_expanded, ddim_eta)

                is_last_step = (t_prev < 0)
                if return_all_timesteps and ((t_prev % sample_every_n_steps == 0 and t_prev >= 0) or is_last_step):
                    t_to_record = 0 if is_last_step else t_prev
                    if t_to_record not in recorded_timesteps_list:
                        intermediate_y_t_steps.append(
                            y_t.clone().cpu().reshape(original_batch_size, n_samples_per_condition, 1))
                        recorded_timesteps_list.append(t_to_record)

        elif sampler == 'ddpm':
            desc = "DDPM Sampling (Slow)"
            for i in reversed(range(0, self.n_diffusion_steps)):
                # for i in tqdm(reversed(range(0, self.n_diffusion_steps)), desc=desc, leave=False, dynamic_ncols=True):
                y_t = self.p_sample(y_t, i, condition_x_expanded)

                if return_all_timesteps and (i % sample_every_n_steps == 0 or i == 0):
                    if i not in recorded_timesteps_list:  # 避免重复添加 t=0
                        intermediate_y_t_steps.append(
                            y_t.clone().cpu().reshape(original_batch_size, n_samples_per_condition, 1))
                        recorded_timesteps_list.append(i)
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Choose 'ddpm' or 'ddim'.")

        final_samples = y_t.reshape(original_batch_size, n_samples_per_condition)

        if return_all_timesteps:
            all_y_t_trajectory = torch.stack(intermediate_y_t_steps, dim=0)
            all_y_t_trajectory = all_y_t_trajectory.squeeze(-1)
            recorded_timesteps_tensor = torch.tensor(recorded_timesteps_list, dtype=torch.int, device='cpu')
            return final_samples, all_y_t_trajectory, recorded_timesteps_tensor
        else:
            return final_samples

    @torch.no_grad()
    def p_sample_loop_visualize(self, combined_condition, n_samples_per_condition=100, sample_every_n_steps=50,
                                sampler='ddpm', ddim_steps=1000, ddim_eta=0.0):  # <-- 添加 DDIM 参数
        device = self.betas.device
        if combined_condition.shape[0] != 1:
            raise ValueError("此可视化模式仅支持批次大小为1。")

        y_t = torch.randn((n_samples_per_condition, 1), device=device)
        condition_expanded = combined_condition.repeat(n_samples_per_condition, 1)

        trajectory_distributions = []
        recorded_timesteps_list = []

        # --- MODIFICATION START: 选择采样器 ---
        if sampler == 'ddim':
            times = torch.linspace(-1, self.n_diffusion_steps - 1, ddim_steps + 1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:]))

            # --- FIX: 总是保存初始状态 (t=999 或 t=1000) ---
            trajectory_distributions.append(y_t.clone().cpu())
            recorded_timesteps_list.append(times[0])

            for t, t_prev in tqdm(time_pairs, desc="DDIM GIF Sampling", leave=False, dynamic_ncols=True):
                y_t = self.p_sample_ddim(y_t, t, t_prev, condition_expanded, ddim_eta)

                # --- FIX: 检查 t_prev，并始终保存最后一步 ---
                is_last_step = (t_prev < 0)
                if (t_prev % sample_every_n_steps == 0 and t_prev >= 0) or is_last_step:
                    t_to_record = 0 if is_last_step else t_prev
                    if t_to_record not in recorded_timesteps_list:
                        trajectory_distributions.append(y_t.clone().cpu())
                        recorded_timesteps_list.append(t_to_record)

        elif sampler == 'ddpm':
            # --- FIX: 总是保存初始状态 ---
            trajectory_distributions.append(y_t.clone().cpu())
            recorded_timesteps_list.append(self.n_diffusion_steps)

            for i in tqdm(reversed(range(0, self.n_diffusion_steps)), desc="DDPM GIF Sampling", leave=False,
                          dynamic_ncols=True):
                y_t = self.p_sample(y_t, i, condition_expanded)

                # 原始逻辑是正确的，因为它会捕获 i=0
                if i % sample_every_n_steps == 0 and i >= 0:
                    if i not in recorded_timesteps_list:  # 避免重复
                        trajectory_distributions.append(y_t.clone().cpu())
                        recorded_timesteps_list.append(i)

        else:
            raise ValueError(f"Unknown sampler: {sampler}. Choose 'ddpm' or 'ddim'.")
        # --- MODIFICATION END ---

        all_distributions = torch.stack(trajectory_distributions, dim=0)
        recorded_timesteps_tensor = torch.tensor(recorded_timesteps_list, dtype=torch.int, device='cpu')

        return all_distributions.squeeze(-1), recorded_timesteps_tensor

    def forward(self, condition_x, y0=None, tabular_data=None, mode='train',
                **inference_kwargs):

        combined_condition = condition_x
        if self.use_tabular_data:
            if tabular_data is None:
                raise ValueError("tabular_data must be provided when use_tabular_data is True.")
            tabular_data = tabular_data.to(condition_x.device)
            tabular_embedding = self.tabular_encoder(tabular_data)
            combined_condition = torch.cat([condition_x, tabular_embedding], dim=1)

        if mode == 'train':
            if y0 is None:
                raise ValueError("Ground truth y0 must be provided for training.")
            return self.p_losses(combined_condition, y0)

        elif mode == 'inference':
            n_samples = inference_kwargs.get('n_samples', 10)
            return_diffusion_trajectory = inference_kwargs.get('return_diffusion_trajectory', False)
            diffusion_sample_every_n_steps = inference_kwargs.get('diffusion_sample_every_n_steps', 100)
            sampler = inference_kwargs.get('sampler', 'ddpm')
            ddim_steps = inference_kwargs.get('ddim_steps', 1000)
            ddim_eta = inference_kwargs.get('ddim_eta', 0.0)

            return self.p_sample_loop(
                combined_condition,
                n_samples_per_condition=n_samples,
                return_all_timesteps=return_diffusion_trajectory,
                sample_every_n_steps=diffusion_sample_every_n_steps,
                sampler=sampler,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta
            )

        elif mode == 'visualize_gif':
            n_samples = inference_kwargs.get('n_samples', 100)
            diffusion_sample_every_n_steps = inference_kwargs.get('diffusion_sample_every_n_steps', 50)
            sampler = inference_kwargs.get('sampler', 'ddpm')
            ddim_steps = inference_kwargs.get('ddim_steps', 1000)
            ddim_eta = inference_kwargs.get('ddim_eta', 0.0)

            return self.p_sample_loop_visualize(
                combined_condition,
                n_samples_per_condition=n_samples,
                sample_every_n_steps=diffusion_sample_every_n_steps,
                sampler=sampler,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta
            )

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'train', 'inference', or 'visualize_gif'.")
