# evaluate.py
import yaml
from argparse import ArgumentParser
import torch
import torch.nn as nn
import os
import logging
import sys
import numpy as np
from tqdm import tqdm
from torcheval.metrics.functional import r2_score as torcheval_r2_score
import wandb

from src.utils.helpers import set_seed, plot_predictions_vs_actuals, \
    plot_batch_diffusion_heatmap_series, \
    plot_conditional_distribution, \
    plot_mean_predictions_distribution, \
    plot_single_sample_diffusion_trajectory
from src.data.dataset import EchoNet, EchoNetRNC

try:
    from src.data.dataset import EchoNetCAMUS, EchoNetCAMUSRNC
except ImportError:
    EchoNetCAMUS = None
    EchoNetCAMUSRNC = None
from src.utils.metrics import calculate_crps
from src.models.uniformer import uniformer_small
from src.models.diffusion_head import DiffusionRegressorHead
from src.models.mlp_head import get_shallow_mlp_head

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)


def load_trained_model(cfg, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = cfg['model']
    data_cfg = cfg['data']

    feature_extractor_name = model_cfg['feature_extractor']['name']
    feature_extractor, output_dim, is_video_model_flag = None, 0, True

    logging.info(f"Attempting to build feature extractor: {feature_extractor_name}")
    if feature_extractor_name == "uniformer_small":
        feature_extractor = uniformer_small()
        output_dim = feature_extractor.head.in_features
        feature_extractor.head = nn.Identity()
        is_video_model_flag = True
    elif feature_extractor_name == "resnet50_torch_hub":
        try:
            hub_pretrained_option = cfg['model']['feature_extractor'].get('pretrained_weights', False)
            hub_pretrained = hub_pretrained_option == "ImageNet" or hub_pretrained_option is True
            feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=hub_pretrained)
            output_dim = feature_extractor.fc.in_features
            feature_extractor.fc = nn.Identity()
            is_video_model_flag = False
        except Exception as e:
            logging.error(f"Error loading resnet50 for evaluation: {e}");
            raise
    elif feature_extractor_name == "pan_echo":
        import pandas as pd
        from src.models.PanEcho.src.models import FrameTransformer, MultiTaskModel
        from src.models.PanEcho.hubconf import Task
        task_dict = pd.read_pickle(model_cfg['feature_extractor'].get('task_dict'))
        all_tasks = list(task_dict.keys())
        task_list = [Task(t, task_dict[t]['task_type'], task_dict[t]['class_names'], task_dict[t]['mean']) for t in
                     all_tasks]
        encoder = FrameTransformer(model_cfg['feature_extractor'].get('arch'),
                                   model_cfg['feature_extractor'].get('n_heads'),
                                   model_cfg['feature_extractor'].get('n_layers'),
                                   model_cfg['feature_extractor'].get('transformer_dropout'),
                                   model_cfg['feature_extractor'].get('pooling'),
                                   model_cfg['feature_extractor'].get('clip_len'))
        encoder_dim = encoder.encoder.n_features
        model = MultiTaskModel(encoder, encoder_dim, task_list, 0.25, True)
        tasks = model_cfg['feature_extractor'].get('tasks')
        if tasks != 'all':
            for t in all_tasks:
                if t not in tasks: delattr(model, t + '_head')
            model.tasks = [t for t in task_list if t.task_name in tasks]
        local_weights_path = model_cfg['feature_extractor'].get('pretrained_weights')
        print(f"Loading pretrained weights from local file: {local_weights_path}")
        loaded_object = torch.load(local_weights_path, map_location='cpu', weights_only=False)
        if isinstance(loaded_object, dict) and 'weights' in loaded_object:
            weights = loaded_object['weights']
        else:
            weights = loaded_object
        pe_key = 'encoder.time_encoder.pe'
        if pe_key in weights:
            del weights[pe_key]
        model.load_state_dict(weights, strict=False)
        feature_extractor = model.encoder
        output_dim = 768
        is_video_model_flag = True
        logging.info(f"PanEcho feature dimension set to: {output_dim}")
    else:
        raise ValueError(f"Unsupported feature_extractor: {feature_extractor_name}")
    logging.info(f"Feature extractor '{feature_extractor_name}' built. Output dim: {output_dim}")

    final_condition_dim = output_dim
    dataset_class_name = data_cfg.get('dataset_class', 'EchoNet')
    view_fusion_mode = data_cfg.get('view_fusion_mode', 'single')

    DATASET_MAP = {
        "EchoNet": EchoNet,
        "EchoNetRNC": EchoNetRNC,
        "EchoNetCAMUS": EchoNetCAMUS,
        "EchoNetCAMUSRNC": EchoNetCAMUSRNC
    }

    if dataset_class_name == "EchoNetCAMUS" and view_fusion_mode in ['cat', 'add']:
        if view_fusion_mode == 'cat':
            final_condition_dim = output_dim * 2
            logging.info(f"Multi-view fusion mode: 'cat'. Base condition dim = {final_condition_dim}")
        else:  # 'add'
            final_condition_dim = output_dim
            logging.info(f"Multi-view fusion mode: 'add'. Base condition dim = {final_condition_dim}")
    else:
        logging.info(f"Single-view model. Base condition dim = {final_condition_dim}")

    tabular_config = data_cfg.get('tabular_config', None)
    use_tabular_data = data_cfg.get('use_tabular_data', False)

    regressor_head_cfg = model_cfg['regressor_head']
    if regressor_head_cfg['name'] == 'diffusion':
        diffusion_args = regressor_head_cfg.get('diffusion', {})
        init_kwargs = {}
        if 'n_diffusion_steps' in diffusion_args:
            init_kwargs['n_diffusion_steps'] = diffusion_args['n_diffusion_steps']
        if 'time_emb_dim' in diffusion_args:
            init_kwargs['time_emb_dim'] = diffusion_args['time_emb_dim']

        if 'predictor_config' in diffusion_args:
            init_kwargs['predictor_config'] = diffusion_args['predictor_config']
        else:
            pred_params = {}
            if 'hidden_dim' in diffusion_args:
                pred_params['hidden_dim'] = diffusion_args['hidden_dim']
            init_kwargs['predictor_config'] = {
                "name": diffusion_args.get("predictor_name", "DeepAttention"),
                "params": pred_params
            }

        regressor = DiffusionRegressorHead(
            condition_dim=final_condition_dim,
            tabular_config=tabular_config if use_tabular_data else None,
            **init_kwargs
        )
    elif regressor_head_cfg['name'] == 'mlp':
        regressor = get_shallow_mlp_head(dim_in=final_condition_dim, **regressor_head_cfg.get('mlp', {}))
    else:
        raise ValueError(f"Unknown regressor head: {regressor_head_cfg['name']}")
    logging.info(f"Regressor head '{regressor_head_cfg['name']}' built.")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    fe_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
    reg_key = 'regressor_state_dict' if 'regressor_state_dict' in checkpoint else 'regressor'
    if fe_key not in checkpoint or reg_key not in checkpoint:
        raise KeyError(f"Checkpoint must contain '{fe_key}' and '{reg_key}' keys.")

    feature_extractor.load_state_dict(checkpoint[fe_key])
    regressor.load_state_dict(checkpoint[reg_key])
    logging.info(f"Weights loaded successfully from epoch {checkpoint.get('epoch', 'N/A')}.")

    feature_extractor.to(device)
    regressor.to(device)

    num_gpus_config = len(cfg['run'].get('gpu_ids', "0").split(','))
    if torch.cuda.device_count() > 1 and num_gpus_config > 1:
        logging.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs for eval.")
        if not isinstance(feature_extractor, nn.DataParallel): feature_extractor = nn.DataParallel(feature_extractor)
        if not isinstance(regressor, nn.DataParallel): regressor = nn.DataParallel(regressor)

    return feature_extractor, regressor, is_video_model_flag, device


def main_evaluate():
    parser = ArgumentParser(description="Evaluation Script for LVEF Estimation Models")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint.')
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--gpu', type=str, default=None)
    parser.add_argument('--visualize_diffusion_process', action='store_true')
    parser.add_argument('--visualize_single_trajectory', action='store_true')
    parser.add_argument('--visualize_k_sample_conditional_dist', action='store_true')
    parser.add_argument('--visualize_mean_pred_dist', action='store_true')
    parser.add_argument('--diffusion_viz_sample_every', type=int, default=100)
    parser.add_argument('--diffusion_viz_max_batch_items', type=int, default=4)
    parser.add_argument('--diffusion_viz_k_sample_idx', type=int, default=0)
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading YAML: {e}");
        return

    if args.data_folder: cfg['data']['data_folder'] = args.data_folder
    cfg_batch_size = cfg['data'].get('batch_size', 32)
    args.batch_size = args.batch_size if args.batch_size is not None else cfg_batch_size
    if args.gpu: cfg['run']['gpu_ids'] = args.gpu
    for k, v in vars(args).items():
        if k not in ['config', 'checkpoint', 'data_folder', 'batch_size', 'gpu']:
            cfg['evaluation'][k] = v

    if cfg['run'].get('gpu_ids') is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['run']['gpu_ids'])
        logging.info(f"CUDA_VISIBLE_DEVICES set to '{cfg['run']['gpu_ids']}'")

    set_seed(cfg['run'].get('seed', 42))
    feature_extractor, regressor, is_video_model_flag, device = load_trained_model(cfg, args.checkpoint)

    data_cfg = cfg['data']

    dataset_class_name = data_cfg.get('dataset_class', 'EchoNet')
    DATASET_MAP = {
        "EchoNet": EchoNet,
        "EchoNetRNC": EchoNetRNC,
        "EchoNetCAMUS": EchoNetCAMUS,
        "EchoNetCAMUSRNC": EchoNetCAMUSRNC
    }
    BaseDatasetClass = DATASET_MAP.get(dataset_class_name)
    if BaseDatasetClass is None:
        raise ValueError(f"Config error: 'dataset_class' {dataset_class_name} not found.")

    test_dataset = BaseDatasetClass(cfg=cfg, split="test")

    eval_batch_size = args.batch_size
    eval_batch_size = max(1, eval_batch_size)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False,
                                              num_workers=data_cfg.get('num_workers', 4), pin_memory=True)
    logging.info(
        f"Test data loader created (Base: {dataset_class_name}). Size: {len(test_dataset)}. Eval Batch Size: {eval_batch_size}")

    feature_extractor.eval()
    regressor.eval()

    all_preds_list, all_labels_list = [], []
    all_preds_k_samples_list = []

    first_batch_viz_done = False
    eval_wandb_run = None
    logging.info("Starting evaluation on the test set...")
    current_test_loader = test_loader

    use_tabular = data_cfg.get('use_tabular_data', False)
    view_fusion_mode = data_cfg.get('view_fusion_mode', 'single')
    eval_cfg = cfg['evaluation']

    if cfg['wandb'].get('use_for_evaluation', False):
        try:
            if wandb.run is None:
                eval_wandb_run = wandb.init(
                    project=cfg['wandb'].get('project', 'CDMR-Evaluation'),
                    name=f"EVAL_{cfg['run']['name']}",
                    config=cfg,
                    notes=f"Evaluation run for {cfg['run']['name']} using checkpoint {args.checkpoint}"
                )
            logging.info("W&B initialized for evaluation.")
        except Exception as e:
            logging.error(f"Failed to initialize W&B for evaluation: {e}")
            cfg['wandb']['use_for_evaluation'] = False

    for batch_idx, batch_data in enumerate(
            tqdm(current_test_loader, desc="Evaluating Test Set", leave=False, dynamic_ncols=True)):

        images_or_views = None
        labels = None
        tabular_data = None
        if use_tabular:
            if len(batch_data) != 3:
                logging.error("Data loader expected 3 items (images, labels, tabular) but got {len(batch_data)}")
                continue
            images_or_views, labels, tabular_data = batch_data
            tabular_data = tabular_data.to(device, non_blocking=True)
        else:
            if len(batch_data) == 3:
                images_or_views, labels, _ = batch_data
            elif len(batch_data) == 2:
                images_or_views, labels = batch_data
            else:
                logging.error(f"Data loader expected 2 or 3 items but got {len(batch_data)}")
                continue
            tabular_data = None

        bsz = labels.shape[0]

        with torch.no_grad():
            features = None
            if view_fusion_mode in ['cat', 'add']:
                if not isinstance(images_or_views, (list, tuple)):
                    logging.error("Fusion mode 'cat'/'add' expected list of views but got single tensor.")
                    continue
                images_2ch, images_4ch = images_or_views
                images_2ch = images_2ch.to(device, non_blocking=True)
                images_4ch = images_4ch.to(device, non_blocking=True)

                processed_2ch = images_2ch
                processed_4ch = images_4ch

                features_2ch = feature_extractor(processed_2ch)
                features_4ch = feature_extractor(processed_4ch)

                if view_fusion_mode == 'cat':
                    features = torch.cat([features_2ch, features_4ch], dim=1)
                else:  # 'add'
                    features = features_2ch + features_4ch

            else:
                if isinstance(images_or_views, (list, tuple)):
                    logging.error("Single view mode expected a single tensor but got list/tuple.")
                    continue
                images = images_or_views.to(device, non_blocking=True)
                processed_images = images
                if not is_video_model_flag:
                    b_img, c_img, t_img, h_img, w_img = images.shape
                    processed_images = images.permute(0, 2, 1, 3, 4).reshape(b_img * t_img, c_img, h_img, w_img)

                features_raw = feature_extractor(processed_images)
                features = features_raw
                if not is_video_model_flag:
                    output_dim = feature_extractor.fc.in_features if hasattr(feature_extractor,
                                                                             'fc') else feature_extractor.classifier.in_features
                    features = features_raw.view(bsz, -1, output_dim).mean(dim=1)

            head_name = cfg['model']['regressor_head']['name']
            preds_k_samples = None
            diffusion_trajectories = None
            recorded_timesteps_for_traj = None

            if head_name == 'diffusion':
                reg_module = regressor.module if isinstance(regressor, nn.DataParallel) else regressor

                get_traj_for_this_batch = (eval_cfg['visualize_diffusion_process'] or
                                           eval_cfg['visualize_single_trajectory']) and \
                                          not first_batch_viz_done

                inference_params = {
                    'n_samples': eval_cfg.get('inference_samples', 10),
                    'return_diffusion_trajectory': get_traj_for_this_batch,
                    'diffusion_sample_every_n_steps': eval_cfg.get('diffusion_viz_sample_every', 100),
                    'sampler': eval_cfg.get('sampler', 'ddpm'),
                    'ddim_steps': eval_cfg.get('ddim_steps', 1000),
                    'ddim_eta': eval_cfg.get('ddim_eta', 0.0)
                }

                diffusion_output = reg_module(
                    condition_x=features,
                    tabular_data=tabular_data,
                    mode='inference',
                    **inference_params
                )

                if get_traj_for_this_batch:
                    preds_k_samples, diffusion_trajectories, recorded_timesteps_for_traj = diffusion_output
                else:
                    preds_k_samples = diffusion_output

                preds_final_mean = preds_k_samples.mean(dim=1).cpu()
            else:
                preds_final_mean = regressor(features).cpu()
                preds_k_samples = preds_final_mean.unsqueeze(1)  # [B] -> [B, 1]

        all_preds_list.append(preds_final_mean.view(-1))
        all_labels_list.append(labels.view(-1).cpu())
        all_preds_k_samples_list.append(preds_k_samples.cpu())

        if not first_batch_viz_done and cfg['wandb'].get('use_for_evaluation', False):
            log_payload_batch0 = {}
            if eval_cfg['visualize_diffusion_process'] and diffusion_trajectories is not None:
                heatmap_series_img = plot_batch_diffusion_heatmap_series(
                    trajectories=diffusion_trajectories,
                    recorded_timesteps=recorded_timesteps_for_traj,
                    n_samples_k_to_plot=eval_cfg['diffusion_viz_k_sample_idx'],
                    title_suffix=f"(Test Batch {batch_idx})"
                )
                if heatmap_series_img: log_payload_batch0["eval/diffusion_heatmap_series_batch0"] = wandb.Image(
                    heatmap_series_img)
                logging.info(f"Generated diffusion heatmap series for W&B.")

            if eval_cfg['visualize_k_sample_conditional_dist'] and preds_k_samples is not None:
                num_items_for_dist_plot = min(bsz, eval_cfg['diffusion_viz_max_batch_items'])
                k_sample_conditional_dist_img = plot_conditional_distribution(
                    final_samples_k=preds_k_samples[:num_items_for_dist_plot],
                    true_lvef=labels[:num_items_for_dist_plot].squeeze(),
                    max_batch_items_to_plot=num_items_for_dist_plot,
                    title_suffix=f"(Test Batch {batch_idx}, First {num_items_for_dist_plot} items, K samples)"
                )
                if k_sample_conditional_dist_img: log_payload_batch0[
                    "eval/k_sample_conditional_dist_batch0"] = wandb.Image(k_sample_conditional_dist_img)
                logging.info(f"Generated K-sample conditional distributions for W&B.")

            if eval_cfg['visualize_single_trajectory'] and diffusion_trajectories is not None:
                logging.info("Generating trajectory beam diagrams for the first few samples in the first batch...")
                num_items_for_traj_plot = min(bsz, eval_cfg['diffusion_viz_max_batch_items'])

                for i in range(num_items_for_traj_plot):
                    trajectory_one_sample = diffusion_trajectories[:, i, :]
                    trajectory_img = plot_single_sample_diffusion_trajectory(
                        trajectory_for_one_sample=trajectory_one_sample,
                        recorded_timesteps=recorded_timesteps_for_traj,
                        true_lvef=labels[i].item(),
                        sample_index=i,
                        title_suffix=f"(Test Batch {batch_idx})"
                    )
                    if trajectory_img:
                        log_payload_batch0[f"eval/sample_{i}_trajectory"] = wandb.Image(trajectory_img)

            if log_payload_batch0 and eval_wandb_run:
                wandb.log(log_payload_batch0)

            if eval_cfg['visualize_diffusion_process'] or eval_cfg['visualize_k_sample_conditional_dist']:
                first_batch_viz_done = True
                if len(test_dataset) > eval_batch_size:
                    logging.info("Resetting data loader to full batch size for remaining evaluation.")
                    current_test_loader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=args.batch_size,  # 使用 args 中最终的 batch_size
                        shuffle=False,
                        num_workers=data_cfg.get('num_workers', 4),
                        pin_memory=True,
                        sampler=torch.utils.data.SequentialSampler(range(eval_batch_size, len(test_dataset)))
                    )
                    pbar = tqdm(current_test_loader, desc="Evaluating Test Set (Full BS)", leave=False,
                                dynamic_ncols=True)

    all_preds = torch.cat(all_preds_list)
    all_labels = torch.cat(all_labels_list)
    all_preds_k_samples = torch.cat(all_preds_k_samples_list)  # [N_total, K]

    mae = torch.abs(all_preds - all_labels).mean().item()
    rmse = torch.sqrt(((all_preds - all_labels) ** 2).mean()).item()
    r2 = 0.0
    if all_preds.shape[0] > 1 and all_labels.shape[0] > 1 and torch.var(all_preds) > 1e-6 and torch.var(
            all_labels) > 1e-6:
        try:
            r2 = torcheval_r2_score(all_preds, all_labels).item()
        except Exception:
            try:
                corr = np.corrcoef(all_preds.numpy(), all_labels.numpy())[0, 1];
                r2 = corr ** 2 if not np.isnan(corr) else 0.0
            except Exception:
                pass

    mape_sum = torch.abs((all_labels - all_preds) / all_labels[all_labels > 1e-8]).sum()
    valid_count = len(all_labels[all_labels > 1e-8])
    mape = (mape_sum / valid_count * 100).item() if valid_count > 0 else float('inf')

    crps = 0.0

    head_name = cfg['model']['regressor_head']['name']
    n_samples_k = all_preds_k_samples.shape[1]

    if head_name == 'diffusion' and n_samples_k > 1:
        logging.info(f"Calculating probabilistic metrics (CRPS) using K={n_samples_k} samples...")
        try:
            crps = calculate_crps(all_preds_k_samples, all_labels)
        except Exception as e:
            logging.error(f"Failed to calculate CRPS: {e}")
    else:
        logging.info(f"Probabilistic metrics (CRPS) skipped. (Model: {head_name}, K={n_samples_k})")

    logging.info("======= Evaluation on Test Set Finished =======")
    logging.info(f"Config: {args.config} | Checkpoint: {args.checkpoint}")
    logging.info(f"Total Samples: {len(all_labels)}")
    logging.info(f"--- Standard Metrics ---")
    logging.info(f"Test MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | MAPE: {mape:.2f}%")
    logging.info(f"--- Probabilistic Metrics (K={n_samples_k}) ---")
    logging.info(f"Test CRPS: {crps:.4f}")
    logging.info("===============================================")

    if eval_wandb_run:
        final_log_data = {
            'test/MAE': mae,
            'test/RMSE': rmse,
            'test/R2': r2,
            'test/MAPE': mape,
            'test/CRPS': crps,
            'test/config_K_samples': n_samples_k
        }

        scatter_plot_img = plot_predictions_vs_actuals(all_labels, all_preds,
                                                       title="Final Test: Predictions vs. Actuals")
        if scatter_plot_img:
            final_log_data['test/predictions_vs_actuals_scatter'] = wandb.Image(scatter_plot_img)

        if eval_cfg.get('visualize_mean_pred_dist', False):
            mean_dist_img = plot_mean_predictions_distribution(all_preds, all_labels,
                                                               title="Test Set: Distribution of Mean Predictions")
            if mean_dist_img:
                final_log_data['test/mean_predictions_distribution'] = wandb.Image(mean_dist_img)

        wandb.log(final_log_data)
        wandb.finish()
        logging.info("Final metrics and plots logged to W&B.")


if __name__ == '__main__':
    main_evaluate()
