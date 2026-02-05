# src/utils/helpers.py
import numpy as np
import random
import torch
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import io
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
import math
import matplotlib.gridspec as gridspec

FONT_SIZE_TITLE = 14 - 1
FONT_SIZE_LABEL = 12 - 1
FONT_SIZE_LEGEND = 10 - 2
FONT_SIZE_TICKS = 8


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0


def set_seed(seed):
    if seed is None:
        logging.warning("Seed is None. Reproducibility is not guaranteed.")
        return

    seed = int(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    logging.info(f"Seed set to {seed} for reproducibility.")


def plot_predictions_vs_actuals(y_true, y_pred, title="Predictions vs. Actuals"):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    plt.figure(figsize=(3.5, 3.2))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, edgecolor=None, s=20)  # 稍微增大点的大小

    min_val = 0
    max_val = 100
    if len(y_true) > 0 and len(y_pred) > 0:
        min_val = min(np.min(y_true), np.min(y_pred), 0)
        max_val = max(np.max(y_true), np.max(y_pred), 100)

    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=2.5, label='Ideal (y=x)')
    plt.xlim(min_val - 5, max_val + 5)
    plt.ylim(min_val - 5, max_val + 5)

    plt.xlabel("Actual LVEF (%)", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Predicted LVEF (%)", fontsize=FONT_SIZE_LABEL)
    # plt.title(title, fontsize=FONT_SIZE_TITLE)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    # --------------------

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.subplots_adjust(left=0.16, bottom=0.15)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)  # 提高DPI以获得更高分辨率
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img


def plot_mean_predictions_distribution(
        mean_predictions,
        true_lvef=None,
        title="Distribution of Mean Predictions vs. True LVEFs"
):
    if isinstance(mean_predictions, torch.Tensor):
        mean_predictions = mean_predictions.cpu().numpy()
    if isinstance(true_lvef, torch.Tensor):
        true_lvef = true_lvef.cpu().numpy()

    plt.figure(figsize=(4, 3.2))

    sns.histplot(mean_predictions, color="skyblue", label="Mean Predictions ($\hat{y}_{final}$)", kde=True,
                 stat="density", element="step", line_kws={'linewidth': 2.5})
    if true_lvef is not None:
        sns.histplot(true_lvef, color="salmon", label="True LVEFs ($y_{true}$)", kde=True, stat="density",
                     element="step", line_kws={'linewidth': 2.5})

    # plt.title(title, fontsize=FONT_SIZE_TITLE)
    plt.xlabel("LVEF (%)", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Density", fontsize=FONT_SIZE_LABEL)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    # plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img


def plot_single_sample_diffusion_trajectory(
        trajectory_for_one_sample,
        recorded_timesteps,
        true_lvef,
        sample_index=0,
        title_suffix=""
):
    if isinstance(trajectory_for_one_sample, torch.Tensor):
        trajectory = trajectory_for_one_sample.cpu().numpy()
    else:
        trajectory = trajectory_for_one_sample
    if isinstance(recorded_timesteps, torch.Tensor):
        timesteps = recorded_timesteps.cpu().numpy()
    else:
        timesteps = recorded_timesteps

    sorted_indices = np.argsort(timesteps)[::-1]
    timesteps, trajectory = timesteps[sorted_indices], trajectory[sorted_indices, :]
    num_steps, k_samples = trajectory.shape
    start_noise_dist, final_pred_dist = trajectory[0, :], trajectory[-1, :]
    mean_trajectory, std_trajectory = np.mean(trajectory, axis=1), np.std(trajectory, axis=1)
    upper_bound, lower_bound = mean_trajectory + std_trajectory, mean_trajectory - std_trajectory

    fig = plt.figure(figsize=(5, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 5, 1], wspace=0.05)
    ax_start, ax_main, ax_end = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])

    for k in range(k_samples):
        ax_main.plot(timesteps, trajectory[:, k], color='gray', alpha=0.1, linewidth=0.75)
    ax_main.scatter(timesteps[0] * np.ones(k_samples), start_noise_dist, color='orange', s=15, alpha=0.2, zorder=3)
    ax_main.scatter(timesteps[-1] * np.ones(k_samples), final_pred_dist, color='limegreen', s=15, alpha=0.5, zorder=3)
    ax_main.plot(timesteps, mean_trajectory, color='blue', linewidth=1.75, label='Mean Trajectory')
    ax_main.fill_between(timesteps, lower_bound, upper_bound, color='dodgerblue', alpha=0.6, label='Standard Deviation')
    ax_main.axhline(y=true_lvef, color='red', linestyle='--', linewidth=2, label=f'True LVEF: {true_lvef:.2f}')

    sns.violinplot(y=start_noise_dist, ax=ax_start, color='orange', inner=None)
    sns.stripplot(y=start_noise_dist, ax=ax_start, color='black', size=1, alpha=0.4)
    ax_start.axhline(y=0, color='black', linestyle=':', linewidth=1.5)
    # ax_start.set_title(f"Start (t={timesteps[0]})\nNoise Distribution", fontsize=FONT_SIZE_TITLE - 2)
    ax_start.set_ylabel("LVEF Value", fontsize=FONT_SIZE_LABEL)
    ax_start.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    ax_start.set_xticks([])
    ax_start.grid(True, linestyle=':', alpha=0.7)

    sns.violinplot(y=final_pred_dist, ax=ax_end, color='limegreen', inner=None)
    sns.stripplot(y=final_pred_dist, ax=ax_end, color='black', size=1, alpha=0.4)
    ax_end.axhline(y=np.mean(final_pred_dist), color='blue', linestyle='-', linewidth=1.75)
    ax_end.axhline(y=true_lvef, color='red', linestyle='--', linewidth=1.5)
    # ax_end.set_title(f"End (t=0)\nPredicted Distribution", fontsize=FONT_SIZE_TITLE - 2)
    # mean_pred = np.mean(final_pred_dist)
    # ax_end.axhline(y=mean_pred, color='blue', linestyle='--', linewidth=2.5, label=f'Mean Pred: {mean_pred:.2f}')
    # ax_end.axhline(y=true_lvef, color='red', linestyle='--', linewidth=2.5, label=f'True LVEF: {true_lvef:.2f}')
    # ax_end.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    ax_end.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    ax_end.set_xticks([])
    # ax_end.set_yticks([])
    ax_end.grid(True, linestyle=':', alpha=0.7)

    y_min, y_max = min(lower_bound.min(), true_lvef) - 5, max(upper_bound.max(), true_lvef) + 5
    ax_main.set_ylim(y_min, y_max)
    ax_start.set_ylim(y_min, y_max)
    ax_end.set_ylim(y_min, y_max)
    ax_main.invert_xaxis()
    ax_main.set_xlabel("Diffusion Timestep (t)", fontsize=FONT_SIZE_LABEL)
    ax_main.legend(fontsize=FONT_SIZE_LEGEND, loc='best')
    ax_main.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
    ax_main.set_yticklabels([])
    ax_main.grid(True, linestyle=':', alpha=0.7)

    # fig.suptitle(f"Full Denoising Process for Sample {sample_index + 1} {title_suffix}".strip(), fontsize=FONT_SIZE_TITLE - 2)
    # plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(bottom=0.15)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def plot_batch_diffusion_heatmap_series(
        trajectories,
        recorded_timesteps,
        n_samples_k_to_plot=0,
        title_suffix=""
):
    if isinstance(trajectories, torch.Tensor):
        trajectories = trajectories.cpu().numpy()
    if isinstance(recorded_timesteps, torch.Tensor):
        recorded_timesteps = recorded_timesteps.cpu().numpy()

    num_recorded_steps, batch_size, n_samples_k_total = trajectories.shape

    if n_samples_k_to_plot >= n_samples_k_total:
        logging.warning(
            f"n_samples_k_to_plot ({n_samples_k_to_plot}) is out of bounds for K={n_samples_k_total}. Defaulting to 0.")
        n_samples_k_to_plot = 0

    data_to_plot = trajectories[:, :, n_samples_k_to_plot][::-1]
    plot_timesteps = recorded_timesteps[::-1]

    if num_recorded_steps == 0:
        logging.warning("No recorded steps to plot for heatmap series.")
        return None

    n_cols = math.ceil(math.sqrt(num_recorded_steps))
    n_rows = math.ceil(num_recorded_steps / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 3.5), squeeze=False)
    axes = axes.flatten()

    vmin, vmax = np.min(data_to_plot), np.max(data_to_plot)

    for i in range(num_recorded_steps):
        ax = axes[i]
        heatmap_data = data_to_plot[i, :].reshape(-1, 1)

        sns.heatmap(heatmap_data, ax=ax, cmap="viridis", annot=False, cbar=True, vmin=vmin, vmax=vmax)

        ax.set_title(f"t={plot_timesteps[i]}", fontsize=FONT_SIZE_LABEL)
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)  # 调整y轴刻度字体
        if i % n_cols == 0:
            ax.set_ylabel("Batch Item Index", fontsize=FONT_SIZE_LABEL)
        else:
            ax.set_yticks([])

    for j in range(num_recorded_steps, n_cols * n_rows):
        fig.delaxes(axes[j])

    fig.suptitle(f"Batch Denoising: $y_t$ (k={n_samples_k_to_plot}-th sample) {title_suffix}".strip(),
                 fontsize=FONT_SIZE_TITLE)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def plot_conditional_distribution(
        final_samples_k,
        true_lvef=None,
        max_batch_items_to_plot=4,
        title_suffix=""
):
    if isinstance(final_samples_k, torch.Tensor):
        final_samples_k = final_samples_k.cpu().numpy()
    if true_lvef is not None and isinstance(true_lvef, torch.Tensor):
        true_lvef = true_lvef.cpu().numpy()

    batch_size, n_samples_k = final_samples_k.shape
    num_to_plot = min(batch_size, max_batch_items_to_plot)

    if num_to_plot == 0:
        logging.warning("No samples to plot for conditional distribution.")
        return None

    fig, axes = plt.subplots(num_to_plot, 1, figsize=(5, 3 * num_to_plot), squeeze=False)
    axes = axes.flatten()

    for i in range(num_to_plot):
        ax = axes[i]
        sns.histplot(final_samples_k[i, :], ax=ax, kde=True, stat="density",
                     bins=min(n_samples_k // 2 if n_samples_k > 1 else 1, 25),
                     label=f"k={n_samples_k} Sampling Count",
                     line_kws={'linewidth': 2.5})

        current_mean = np.mean(final_samples_k[i, :])
        current_std = np.std(final_samples_k[i, :])
        ax.axvline(current_mean, color='blue', linestyle='--', lw=2, label=f"Mean Pred: {current_mean:.2f}")

        if true_lvef is not None and i < len(true_lvef):
            ax.axvline(true_lvef[i], color='red', linestyle='-', lw=2.5, label=f"True LVEF: {true_lvef[i]:.2f}")

        ax.set_title(f"t=0: Predicted LVEF: {current_mean:.1f}% ± {current_std:.1f}%",
                     fontsize=FONT_SIZE_TITLE)
        ax.set_ylabel("Density", fontsize=FONT_SIZE_LABEL + 2)
        ax.set_xlabel("Predicted LVEF (%)", fontsize=FONT_SIZE_LABEL + 2)  # 为每个子图添加X轴标签
        ax.legend(fontsize=FONT_SIZE_LEGEND + 4, loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS + 2)  # 同时调整X和Y轴刻度字体
        ax.grid(True, linestyle=':', alpha=0.7)

    fig.suptitle(f"Learned Conditional Distributions $p(y|c)$ {title_suffix}".strip(), fontsize=FONT_SIZE_TITLE + 2,
                 y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


import matplotlib.animation as animation


def create_diffusion_animation(
        distributions_over_time,
        recorded_timesteps,
        true_lvef,
        output_filename="diffusion_animation.gif"
):
    if isinstance(distributions_over_time, torch.Tensor):
        distributions = distributions_over_time.cpu().numpy()
    else:
        distributions = distributions_over_time

    if isinstance(recorded_timesteps, torch.Tensor):
        timesteps = recorded_timesteps.cpu().numpy()
    else:
        timesteps = recorded_timesteps

    sorted_indices = np.argsort(timesteps)[::-1]

    timesteps = timesteps[sorted_indices]
    distributions = distributions[sorted_indices, :]

    num_frames, n_samples_k = distributions.shape

    x_min = min(np.min(distributions), true_lvef) - 10
    x_max = max(np.max(distributions), true_lvef) + 10

    fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)

    def update(frame_index):
        ax.clear()

        current_t = timesteps[frame_index]
        current_dist = distributions[frame_index, :]

        sns.histplot(
            current_dist, ax=ax, kde=True, stat="density",
            bins=min(n_samples_k // 2 if n_samples_k > 1 else 1, 25),
            label=f"K={n_samples_k} Sampling Count",
            line_kws={'linewidth': 2.5}
        )

        current_mean = np.mean(current_dist)
        current_std = np.std(current_dist)

        ax.axvline(current_mean, color='blue', linestyle='--', lw=2, label=f"Mean Pred: {current_mean:.2f}")
        ax.axvline(true_lvef, color='red', linestyle='-', lw=2.5, label=f"True LVEF: {true_lvef:.2f}")

        ax.set_title(f"t={current_t}: Predicted EF: {current_mean:.1f}% ± {current_std:.1f}%",
                     fontsize=FONT_SIZE_TITLE + 2)
        ax.set_xlabel("Predicted LVEF (%)", fontsize=FONT_SIZE_LABEL + 2)
        ax.set_ylabel("Density", fontsize=FONT_SIZE_LABEL + 2)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS + 2)
        ax.legend(fontsize=FONT_SIZE_LEGEND + 2, loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, max(0.2, ax.get_ylim()[1]))  # 动态调整Y轴，但设置一个最小值
        # -----------------------------------------------

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=200)

    print(f"正在渲染并保存GIF到 {output_filename}...")
    ani.save(output_filename, writer='pillow', fps=5, dpi=150)
    plt.close(fig)
    print("GIF保存成功！")



def create_dynamic_trajectory_animation(
        distributions_over_time,
        recorded_timesteps,
        true_lvef,
        sample_index=0,
        output_filename="generative_trajectory_animation.gif"
):
    distributions_np = distributions_over_time.cpu().numpy() if isinstance(distributions_over_time,
                                                                           torch.Tensor) else np.array(
        distributions_over_time)
    timesteps_np = recorded_timesteps.cpu().numpy() if isinstance(recorded_timesteps, torch.Tensor) else np.array(
        recorded_timesteps)

    sorted_indices = np.argsort(timesteps_np)[::-1]
    timesteps = timesteps_np[sorted_indices]
    distributions = distributions_np[sorted_indices, :]

    num_frames, k_samples = distributions.shape
    mean_trajectory = np.mean(distributions, axis=1)
    std_trajectory = np.std(distributions, axis=1)
    upper_bound = mean_trajectory + std_trajectory
    lower_bound = mean_trajectory - std_trajectory

    start_noise_dist = distributions[0, :]

    fig = plt.figure(figsize=(12, 8), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 5, 1], wspace=0.05, figure=fig)
    ax_start = plt.subplot(gs[0])
    ax_main = plt.subplot(gs[1])
    ax_dist = plt.subplot(gs[2])

    sns.violinplot(y=start_noise_dist, ax=ax_start, color='orange', inner=None)
    sns.stripplot(y=start_noise_dist, ax=ax_start, color='black', size=3, alpha=0.4)
    ax_start.axhline(y=0, color='black', linestyle=':', linewidth=2)
    ax_start.set_title(f"Start (t={timesteps[0]})", fontsize=FONT_SIZE_TITLE)
    ax_start.set_ylabel("LVEF Value", fontsize=FONT_SIZE_LABEL)
    ax_start.tick_params(axis='y', labelsize=FONT_SIZE_TICKS)
    ax_start.set_xticks([])
    ax_start.grid(True, linestyle=':', alpha=0.7)

    ax_main.invert_xaxis()
    ax_main.set_xlabel("Diffusion Timestep (t)", fontsize=FONT_SIZE_LABEL)
    ax_main.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
    ax_main.set_yticklabels([])
    ax_main.grid(True, linestyle=':', alpha=0.7)

    y_min, y_max = min(np.min(distributions), true_lvef) - 10, max(np.max(distributions), true_lvef) + 10
    ax_main.set_ylim(y_min, y_max)
    ax_start.set_ylim(y_min, y_max)

    def update(frame_index):
        for artist in getattr(ax_main, "artists_to_clear", []): artist.remove()
        ax_dist.clear()

        current_step_index = frame_index + 1

        artists = []
        for k in range(k_samples):
            line, = ax_main.plot(timesteps[:current_step_index], distributions[:current_step_index, k], color='gray',
                                 alpha=0.1, linewidth=1.5)
            artists.append(line)
        mean_line, = ax_main.plot(timesteps[:current_step_index], mean_trajectory[:current_step_index],
                                  color='dodgerblue', linewidth=3.5, label='Mean Trajectory')
        artists.append(mean_line)
        fill = ax_main.fill_between(timesteps[:current_step_index], lower_bound[:current_step_index],
                                    upper_bound[:current_step_index], color='dodgerblue', alpha=0.2,
                                    label='Standard Deviation')
        artists.append(fill)
        true_line, = ax_main.plot([timesteps[0], timesteps[-1]], [true_lvef, true_lvef], color='red', linestyle='--',
                                  lw=3, label=f'True LVEF: {true_lvef:.1f}')
        if frame_index == 0: ax_main.legend(fontsize=FONT_SIZE_LEGEND, loc='lower left')
        artists.append(true_line)

        ax_main.artists_to_clear = artists

        current_t = timesteps[frame_index]
        current_dist = distributions[frame_index, :]
        sns.violinplot(y=current_dist, ax=ax_dist, color="skyblue" if current_t > 0 else "limegreen", inner=None)
        sns.stripplot(y=current_dist, ax=ax_dist, color='black', size=3, alpha=0.3)
        ax_dist.axhline(y=np.mean(current_dist), color='blue', linestyle='--', linewidth=2.5)
        ax_dist.axhline(y=true_lvef, color='red', linestyle='--', linewidth=2.5)

        ax_dist.set_title(f"Dist. at t={current_t}", fontsize=FONT_SIZE_TITLE)
        ax_dist.set_ylabel("")
        ax_dist.set_xlabel("")
        ax_dist.set_xticks([])
        ax_dist.set_yticks([])
        ax_dist.grid(True, linestyle=':', alpha=0.7)
        ax_dist.set_ylim(y_min, y_max)

        return artists

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
    fig.suptitle(f"Generative Process for Sample {sample_index + 1}", fontsize=FONT_SIZE_TITLE + 4)

    print(f"正在渲染并保存GIF到 {output_filename}...")
    ani.save(output_filename, writer='pillow', fps=5, dpi=150)
    plt.close(fig)
    print("GIF保存成功！")
