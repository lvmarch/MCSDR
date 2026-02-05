# src/engine/trainer.py
import torch
import torch.nn as nn
import os
import wandb
import numpy as np
import logging
import sys
from tqdm import tqdm
from torcheval.metrics.functional import r2_score as torcheval_r2_score

import time
from thop import profile

# from ..utils.helpers import AverageMeter, set_seed, plot_predictions_vs_actuals
# from ..data.dataset import EchoNet, EchoNetRNC
# from ..models.uniformer import uniformer_small
# from ..models.diffusion_head import DiffusionRegressorHead
# from ..models.mlp_head import get_shallow_mlp_head
# from ..losses.rnc_loss import RnCLoss
# from ..losses.other_loss import DynamicFocusingL1Loss
# # --- MODIFICATION START: 导入新的 METRICS ---
# from ..utils.metrics import calculate_crps

from src.utils.helpers import AverageMeter, set_seed, plot_predictions_vs_actuals
from src.data.dataset import EchoNet, EchoNetRNC
from src.models.uniformer import uniformer_small
from src.models.diffusion_head import DiffusionRegressorHead
from src.models.mlp_head import get_shallow_mlp_head
from src.losses.rnc_loss import RnCLoss
from src.losses.other_loss import DynamicFocusingL1Loss
# --- MODIFICATION START: 导入新的 METRICS ---
from src.utils.metrics import calculate_crps


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save_dir = os.path.join(cfg['run']['save_dir'], cfg['run']['name'])
        os.makedirs(self.save_dir, exist_ok=True)

        self._init_logging()
        set_seed(cfg['run']['seed'])
        self._init_wandb()

        self.data_cfg = self.cfg['data']
        self.model_cfg = self.cfg['model']
        self.train_cfg = self.cfg['training']
        self.use_tabular_data = self.data_cfg.get('use_tabular_data', False)

        self._build_dataloaders()
        self._build_models()
        self._build_optimizers()

        if self.train_cfg['stage'] == 1:
            self.rnc_criterion = RnCLoss(temperature=self.train_cfg['rnc_loss']['temp']).to(self.device)
            self.l1_criterion = nn.L1Loss().to(self.device)
        elif self.train_cfg['stage'] == 2 and self.model_cfg['regressor_head']['name'] == 'mlp':
            self.l1_criterion = nn.L1Loss().to(self.device)

    def _init_logging(self):
        logging.root.handlers = []
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.save_dir, 'training.log')),
                logging.StreamHandler(sys.stdout)
            ])
        logging.info(f"Full Configuration: {self.cfg}")
        logging.info(f"Using device: {self.device}")
        if torch.cuda.is_available():
            logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
            logging.info(f"CUDA Device Count: {torch.cuda.device_count()}")

    def _init_wandb(self):
        if self.cfg['wandb']['use']:
            try:
                wandb.login()
                wandb.init(
                    project=self.cfg['wandb']['project'],
                    name=self.cfg['run']['name'],
                    config=self.cfg,
                    notes=self.cfg['run'].get('notes', '')
                )
                logging.info("Weights & Biases initialized successfully.")
            except Exception as e:
                logging.error(f"Failed to initialize wandb: {e}. Disabling W&B for this run.")
                self.cfg['wandb']['use'] = False

    def _build_dataloaders(self):
        data_cfg = self.cfg['data']
        dataset_class_train = EchoNetRNC if data_cfg.get('data_loader_type') == 'rnc' else EchoNet

        train_dataset = dataset_class_train(cfg=self.cfg, split="train")
        val_dataset = EchoNet(cfg=self.cfg, split="val")
        test_dataset = dataset_class_train(cfg=self.cfg, split="test")

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=data_cfg['batch_size'], shuffle=True,
            num_workers=data_cfg['num_workers'], pin_memory=True, drop_last=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=data_cfg['batch_size'], shuffle=False,
            num_workers=data_cfg['num_workers'], pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=data_cfg['batch_size'], shuffle=False,
            num_workers=data_cfg['num_workers'], pin_memory=True
        )
        logging.info(
            f"Data loaders created (Base: EchoNet). Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")


    def _build_models(self):
        feature_extractor_name = self.model_cfg['feature_extractor']['name']
        pretrained_weights_path = self.model_cfg['feature_extractor'].get('pretrained_weights')

        if feature_extractor_name == "uniformer_small":
            self.feature_extractor = uniformer_small()
            if pretrained_weights_path and os.path.exists(pretrained_weights_path):
                logging.info(f"Loading UniFormer pretrained weights from: {pretrained_weights_path}")
                try:
                    ckpt = torch.load(pretrained_weights_path, map_location='cpu', weights_only=False)
                    if 'model' in ckpt and 'regressor_state_dict' in ckpt:
                        model_state_dict = {k.replace("module.", ""): v for k, v in ckpt['model_state_dict'].items()}
                    elif 'model' in ckpt:
                        model_state_dict = {k.replace("module.", ""): v for k, v in ckpt['model'].items()}
                    else:
                        model_state_dict = ckpt
                    model_state_dict_cleaned = {k.replace("fc.", "head.") if "fc." in k else k: v for k, v in
                                                model_state_dict.items()}
                    self.feature_extractor.load_state_dict(model_state_dict_cleaned, strict=False)
                    logging.info("Successfully loaded UniFormer weights.")
                except Exception as e:
                    logging.error(f"Error loading UniFormer weights from {pretrained_weights_path}: {e}.")
            else:
                logging.info(f"No valid pretrained_weights path for UniFormer, using random initialization.")

            self.output_dim = self.feature_extractor.output_dim

            self.feature_extractor.head = nn.Identity()
            self.is_video_model = True

        elif feature_extractor_name == "resnet50_torch_hub":
            try:
                use_pretrained_hub = pretrained_weights_path == "ImageNet" or pretrained_weights_path is True
                self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50',
                                                        pretrained=use_pretrained_hub)
                logging.info(f"ResNet50 loaded from torch.hub. Pretrained (ImageNet)={use_pretrained_hub}")
                if pretrained_weights_path and not use_pretrained_hub and os.path.exists(pretrained_weights_path):
                    logging.info(f"Loading custom ResNet50 weights from {pretrained_weights_path}")
                    ckpt = torch.load(pretrained_weights_path, map_location='cpu', weights_only=False)
                    model_state_dict_loaded = ckpt.get('model_state_dict',
                                                       ckpt.get('model', ckpt.get('state_dict', ckpt)))
                    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict_loaded.items()}
                    self.feature_extractor.load_state_dict(model_state_dict, strict=False)
            except Exception as e:
                logging.error(f"Error loading ResNet50: {e}. Attempting with pretrained=False.")
                self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
            self.output_dim = self.feature_extractor.fc.in_features
            self.feature_extractor.fc = nn.Identity()
            self.is_video_model = False
            logging.info(f"ResNet50 feature dimension set to: {self.output_dim}")

        elif feature_extractor_name == "pan_echo":
            import pandas as pd
            from src.models.PanEcho.src.models import FrameTransformer, MultiTaskModel
            from src.models.PanEcho.hubconf import Task
            task_dict = pd.read_pickle(self.model_cfg['feature_extractor'].get('task_dict'))
            all_tasks = list(task_dict.keys())
            task_list = [Task(t, task_dict[t]['task_type'], task_dict[t]['class_names'], task_dict[t]['mean']) for t in
                         all_tasks]
            encoder = FrameTransformer(self.model_cfg['feature_extractor'].get('arch'),
                                       self.model_cfg['feature_extractor'].get('n_heads'),
                                       self.model_cfg['feature_extractor'].get('n_layers'),
                                       self.model_cfg['feature_extractor'].get('transformer_dropout'),
                                       self.model_cfg['feature_extractor'].get('pooling'),
                                       self.model_cfg['feature_extractor'].get('clip_len'))
            encoder_dim = encoder.encoder.n_features
            model = MultiTaskModel(encoder, encoder_dim, task_list, 0.25, True)
            tasks = self.model_cfg['feature_extractor'].get('tasks')
            if tasks != 'all':
                for t in all_tasks:
                    if t not in tasks: delattr(model, t + '_head')
                model.tasks = [t for t in task_list if t.task_name in tasks]
            weights = torch.load(pretrained_weights_path, map_location='cpu', weights_only=False).get('weights')
            del weights['encoder.time_encoder.pe']
            model.load_state_dict(weights, strict=False)
            self.feature_extractor = model.encoder
            self.output_dim = encoder_dim
            self.is_video_model = True
            logging.info(f"PanEcho feature dimension set to: {self.output_dim}")
        else:
            raise ValueError(f"Unsupported feature_extractor name: {feature_extractor_name}")

        logging.info(f"Feature Extractor '{feature_extractor_name}' built. Output dim: {self.output_dim}")

        final_condition_dim = self.output_dim
        logging.info(f"Base condition dim = {final_condition_dim}")

        tabular_config = None
        if self.use_tabular_data:
            tabular_config = self.data_cfg.get('tabular_config', {})
            if 'dim' not in tabular_config or 'hidden_dim' not in tabular_config:
                raise ValueError("'data.tabular_config' 必须包含 'dim' 和 'hidden_dim' 键。")
            logging.info(f"Tabular data enabled. Diffusion head will handle embedding.")

        regressor_head_cfg = self.model_cfg['regressor_head']
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

            self.regressor = DiffusionRegressorHead(
                condition_dim=final_condition_dim,
                tabular_config=tabular_config,
                **init_kwargs
            )
        elif regressor_head_cfg['name'] == 'mlp':
            if self.use_tabular_data:
                logging.warning("MLP head + tabular data a-priori fusion not implemented. "
                                "MLP head will ONLY receive video features.")
            self.regressor = get_shallow_mlp_head(dim_in=final_condition_dim, **regressor_head_cfg.get('mlp', {}))
        else:
            raise ValueError(f"Unknown regressor head: {regressor_head_cfg['name']}")

        self.feature_extractor.to(self.device)
        self.regressor.to(self.device)
        if torch.cuda.device_count() > 1 and len(self.cfg['run'].get('gpu_ids', '').split(',')) > 1:
            self.feature_extractor = nn.DataParallel(self.feature_extractor)
            self.regressor = nn.DataParallel(self.regressor)
        logging.info("Models built and moved to device.")

    def _build_optimizers(self):
        params_to_optimize = []
        if self.train_cfg['stage'] == 2:
            finetune_lr_multiplier = self.train_cfg.get('finetune_lr_multiplier', 1.0)

            for param in self.feature_extractor.parameters(): param.requires_grad = True
            for param in self.regressor.parameters(): param.requires_grad = True

            if finetune_lr_multiplier < 1.0:
                logging.info("Optimizer Setup: Stage 2 with Differential Learning Rate.")
                base_lr = self.train_cfg['learning_rate']
                params_to_optimize = [
                    {"params": self.regressor.parameters(), "lr": base_lr},
                    {"params": self.feature_extractor.parameters(), "lr": base_lr * finetune_lr_multiplier}
                ]
            else:
                logging.info("Optimizer Setup: Stage 2. Training feature extractor and regressor at same LR.")
                params_to_optimize = list(self.feature_extractor.parameters()) + list(self.regressor.parameters())

        else:
            logging.info("Optimizer Setup: Stage 1. Training feature extractor and regressor.")
            for param in self.feature_extractor.parameters(): param.requires_grad = True
            for param in self.regressor.parameters(): param.requires_grad = True
            params_to_optimize = list(self.feature_extractor.parameters()) + list(self.regressor.parameters())

        self.optimizer = torch.optim.AdamW(params_to_optimize, lr=self.train_cfg['learning_rate'],
                                           weight_decay=self.train_cfg['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                    T_max=self.train_cfg['epochs'], eta_min=1e-6)
        num_trainable_params = sum(
            p.numel() for group in self.optimizer.param_groups for p in group['params'] if p.requires_grad)
        logging.info(f"Optimizer created (AdamW). Optimizing {num_trainable_params:,} parameters.")

    def run(self):
        logging.info(
            f"======= Training Started: Stage {self.train_cfg['stage']} - Run Name: {self.cfg['run']['name']} =======")
        best_metric = float('inf')
        for epoch in range(1, self.train_cfg['epochs'] + 1):
            logging.info(f"--- Epoch {epoch}/{self.train_cfg['epochs']} ---")
            if self.train_cfg['stage'] == 1:
                train_metrics = self._train_epoch_stage1(epoch)
            elif self.train_cfg['stage'] == 2:
                train_metrics = self._train_epoch_stage2(epoch)
            else:
                raise ValueError(f"Invalid training stage: {self.train_cfg['stage']}")
            val_metrics = self._evaluate(self.val_loader, epoch, "Validation")

            logging.info(
                f"Epoch {epoch} Summary | Train Loss: {train_metrics['loss']:.4f} | Val MAE: {val_metrics['MAE']:.4f} | Val RMSE: {val_metrics['RMSE']:.4f} | Val R2: {val_metrics['R2']:.4f}")
            if self.model_cfg['regressor_head']['name'] == 'diffusion':
                logging.info(
                    f"Epoch {epoch} Proba. | CRPS: {val_metrics['CRPS']:.4f}")

            if self.cfg['wandb']['use']:
                log_data = {f"train/{k.replace('_', '/')}": v for k, v in train_metrics.items()}
                log_data.update({f"val/{k}": v for k, v in val_metrics.items()})
                log_data['epoch'] = epoch
                scatter_plot_img = plot_predictions_vs_actuals(val_metrics['all_labels_for_plot'],
                                                               val_metrics['all_preds_for_plot'],
                                                               title=f"Epoch {epoch}: Validation Preds vs Actuals")
                log_data['val/predictions_vs_actuals_scatter'] = wandb.Image(scatter_plot_img)
                wandb.log(log_data)

            current_mae = val_metrics['MAE']
            if current_mae < best_metric:
                best_metric = current_mae
                self._save_checkpoint('best.pth', epoch, best_metric)
                logging.info(f"New best model saved with MAE: {best_metric:.4f}")
            if epoch % 10 == 0 or epoch == self.train_cfg['epochs']:
                self._save_checkpoint(f'epoch_{epoch}.pth', epoch, current_mae)

        logging.info(f"======= Training Finished: Stage {self.train_cfg['stage']} =======")
        logging.info("======= Final Evaluation on Test Set =======")
        best_ckpt_path = os.path.join(self.save_dir, 'best.pth')
        if os.path.exists(best_ckpt_path):
            logging.info(f"Loading best model for final test from: {best_ckpt_path}")
            checkpoint = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)

            fe_key = 'model_state_dict' if 'model_state_dict' in checkpoint else 'model'
            reg_key = 'regressor_state_dict' if 'regressor_state_dict' in checkpoint else 'regressor'

            if fe_key not in checkpoint or reg_key not in checkpoint:
                logging.error(f"Checkpoint 必须包含 '{fe_key}' 和 '{reg_key}' 键。")
                fe_state_dict, reg_state_dict = checkpoint, checkpoint
            else:
                fe_state_dict, reg_state_dict = checkpoint[fe_key], checkpoint[reg_key]

            target_feature_extractor = self.feature_extractor.module if isinstance(self.feature_extractor,
                                                                                   nn.DataParallel) else self.feature_extractor
            target_regressor = self.regressor.module if isinstance(self.regressor, nn.DataParallel) else self.regressor
            target_feature_extractor.load_state_dict(fe_state_dict)
            target_regressor.load_state_dict(reg_state_dict)
            logging.info(f"Best model (epoch {checkpoint.get('epoch', 'N/A')}) loaded.")
        else:
            logging.warning("No best model checkpoint found for final test.")

        test_metrics = self._evaluate(self.test_loader, "Final", "Test")

        logging.info(f"--- Final Test Metrics ---")
        logging.info(
            f"Standard: MAE: {test_metrics['MAE']:.4f} | RMSE: {test_metrics['RMSE']:.4f} | R2: {test_metrics['R2']:.4f} | MAPE: {test_metrics['MAPE']:.2f}%")
        if self.model_cfg['regressor_head']['name'] == 'diffusion':
            logging.info(f"Probabilistic: CRPS: {test_metrics['CRPS']:.4f}")

        if self.cfg['wandb']['use']:
            final_log_data = {f"test/{k}": v for k, v in test_metrics.items()}
            scatter_plot_img_test = plot_predictions_vs_actuals(test_metrics['all_labels_for_plot'],
                                                                test_metrics['all_preds_for_plot'],
                                                                title=f"Final Test: Preds vs Actuals")
            final_log_data['test/predictions_vs_actuals_scatter'] = wandb.Image(scatter_plot_img_test)
            wandb.log(final_log_data)

    def _process_input_for_feature_extractor(self, images_bcthw):
        if not self.is_video_model:
            b, c, t, h, w = images_bcthw.shape
            return images_bcthw.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        return images_bcthw

    def _process_input_for_feature_extractor_panecho(self, images_bcthw):
        b, c, t, h, w = images_bcthw.shape
        return images_bcthw.reshape(b, c, t, h, w)

    def _aggregate_frame_features(self, frame_or_video_features, original_batch_size):
        if not self.is_video_model:
            feature_dim = frame_or_video_features.shape[-1]
            return frame_or_video_features.view(original_batch_size, -1, feature_dim).mean(dim=1)
        return frame_or_video_features

    def _train_epoch_stage1(self, epoch):
        self.feature_extractor.train()
        self.regressor.train()
        losses_meter, rnc_losses_meter, l1_losses_meter = AverageMeter(), AverageMeter(), AverageMeter()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Stage 1 Training]", leave=False, dynamic_ncols=True)

        for batch_data in pbar:
            images_or_views = None
            labels = None

            if self.use_tabular_data:
                images_or_views, labels, _ = batch_data
            else:
                if len(batch_data) == 3:
                    images_or_views, labels, _ = batch_data
                elif len(batch_data) == 2:
                    images_or_views, labels = batch_data
                else:
                    logging.error(f"数据加载器返回了 {len(batch_data)} 个元素，需要 2 或 3 个。")
                    continue

            if not isinstance(images_or_views, (list, tuple)):
                logging.error("Stage 1 RNC 期望数据加载器返回一个视图元组/列表。")
                continue

            images_v1, images_v2 = images_or_views[0].to(self.device), images_or_views[1].to(self.device)
            labels = labels.to(self.device)
            bsz = labels.shape[0]

            input_v1 = self._process_input_for_feature_extractor(images_v1)
            input_v2 = self._process_input_for_feature_extractor(images_v2)

            features_v1 = self.feature_extractor(input_v1)  # (B, 1024)
            features_v2 = self.feature_extractor(input_v2)  # (B, 1024)

            loss_rnc = self.rnc_criterion(torch.stack([features_v1, features_v2], dim=1), labels)
            y_preds_mlp = self.regressor(torch.cat([features_v1.detach(), features_v2.detach()], dim=0))

            loss_l1 = self.l1_criterion(y_preds_mlp.squeeze(), labels.repeat(2, 1).squeeze())

            loss = loss_rnc + loss_l1
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses_meter.update(loss.item(), bsz)
            pbar.set_postfix(loss=losses_meter.avg)
        return {"loss": losses_meter.avg}

    def _train_epoch_stage2(self, epoch):
        finetune_lr_multiplier = self.train_cfg.get('finetune_lr_multiplier', 1.0)
        if finetune_lr_multiplier < 1.0:
            self.feature_extractor.train()
            if epoch == 1: logging.info("Setting feature extractor to TRAIN mode for fine-tuning.")
        else:
            self.feature_extractor.eval()
            if epoch == 1: logging.info("Setting feature extractor to EVAL mode (frozen).")

        self.regressor.train()
        losses_meter = AverageMeter()
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Stage 2 Training]", leave=False, dynamic_ncols=True)

        for batch_data in pbar:
            images = None
            labels = None
            tabular_data = None

            if self.use_tabular_data:
                if len(batch_data) != 3:
                    logging.error(f"数据加载器返回了 {len(batch_data)} 个元素，但 use_tabular_data=True 需要 3 个。")
                    continue
                images, labels, tabular_data = batch_data
                tabular_data = tabular_data.to(self.device, non_blocking=True)
            else:
                if len(batch_data) == 3:
                    images, labels, _ = batch_data
                elif len(batch_data) == 2:
                    images, labels = batch_data
                else:
                    logging.error(f"数据加载器返回了 {len(batch_data)} 个元素，需要 2 或 3 个。")
                    continue
                tabular_data = None

            labels = labels.to(self.device, non_blocking=True)
            images = images.to(self.device, non_blocking=True)
            bsz = labels.shape[0]

            features = None
            feature_extractor_context = torch.enable_grad() if finetune_lr_multiplier < 1.0 else torch.no_grad()

            with feature_extractor_context:
                if isinstance(images, (list, tuple)):
                    logging.error("Single-view 模式期望数据加载器返回一个张量，但收到了元组。")
                    continue
                processed_images = self._process_input_for_feature_extractor(images)
                features = self.feature_extractor(processed_images)  # (B, 1024)
                features = self._aggregate_frame_features(features, bsz)  # (此行现在无效，但保留以防万一)

            if self.model_cfg['regressor_head']['name'] == 'diffusion':
                loss = self.regressor(condition_x=features, y0=labels, tabular_data=tabular_data, mode='train')
            else:
                y_preds = self.regressor(features)
                loss = self.l1_criterion(y_preds.squeeze(), labels.squeeze())

            losses_meter.update(loss.item(), bsz)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.set_postfix(loss=losses_meter.avg)

        self.scheduler.step()
        return {"loss": losses_meter.avg, "lr": self.optimizer.param_groups[0]['lr']}

    @torch.no_grad()
    def _evaluate(self, data_loader, epoch_or_context_str, eval_phase_name="Validation"):
        self.feature_extractor.eval()
        self.regressor.eval()
        all_preds_list, all_labels_list = [], []
        all_preds_k_samples_list = []

        desc = f"Epoch {epoch_or_context_str} [{eval_phase_name}]"
        pbar = tqdm(data_loader, desc=desc, leave=False, dynamic_ncols=True)

        for batch_data in pbar:
            images = None
            labels = None
            tabular_data = None

            if self.use_tabular_data:
                if len(batch_data) != 3:
                    logging.error(f"数据加载器返回了 {len(batch_data)} 个元素，但 use_tabular_data=True 需要 3 个。")
                    continue
                images, labels, tabular_data = batch_data
                tabular_data = tabular_data.to(self.device, non_blocking=True)
            else:
                if len(batch_data) == 3:
                    images, labels, _ = batch_data
                elif len(batch_data) == 2:
                    images, labels = batch_data
                else:
                    logging.error(f"数据加载器返回了 {len(batch_data)} 个元素，需要 2 或 3 个。")
                    continue
                tabular_data = None

            labels = labels.cpu()
            images = images.to(self.device, non_blocking=True)
            bsz = labels.shape[0]

            features = None
            if isinstance(images, (list, tuple)):
                logging.error("Single-view 模式期望数据加载器返回一个张量，但收到了元组。")
                continue

            expected_frames = self.cfg['data'].get('frames', 32)
            if 'frame_check_warning_logged' not in self.__dict__:
                if images.shape[2] != expected_frames:
                    logging.warning(
                        f"数据加载器帧数 ({images.shape[2]}) 与配置帧数 ({expected_frames}) 不匹配。")
                    self.frame_check_warning_logged = True

            processed_images = self._process_input_for_feature_extractor(images)
            features = self.feature_extractor(processed_images)  # (B, 1024)
            features = self._aggregate_frame_features(features, bsz)  # (此行现在无效)

            # (保留 DDIM 逻辑)
            inference_params = {
                'n_samples': self.cfg['evaluation'].get('inference_samples', 10),
                'sampler': self.cfg['evaluation'].get('sampler', 'ddpm'),
                'ddim_steps': self.cfg['evaluation'].get('ddim_steps', 1000),
                'ddim_eta': self.cfg['evaluation'].get('ddim_eta', 0.0)
            }

            preds = None
            if self.model_cfg['regressor_head']['name'] == 'diffusion':
                reg_module = self.regressor.module if isinstance(self.regressor, nn.DataParallel) else self.regressor
                samples = reg_module(
                    condition_x=features,
                    tabular_data=tabular_data,
                    mode='inference',
                    **inference_params
                )
                preds = samples.mean(dim=1).cpu()
                all_preds_k_samples_list.append(samples.cpu())
            else:
                preds = self.regressor(features).cpu()
                all_preds_k_samples_list.append(preds.unsqueeze(1).cpu())

            all_preds_list.append(preds.view(-1))
            all_labels_list.append(labels.view(-1))

        all_preds = torch.cat(all_preds_list)
        all_labels = torch.cat(all_labels_list)
        all_preds_k_samples = torch.cat(all_preds_k_samples_list)

        mae = torch.abs(all_preds - all_labels).mean().item()
        rmse = torch.sqrt(((all_preds - all_labels) ** 2).mean()).item()
        r2 = torcheval_r2_score(all_preds, all_labels).item() if all_labels.numel() > 1 else 0.0

        mape_sum = torch.abs((all_labels - all_preds) / all_labels[all_labels > 1e-8]).sum()
        valid_count = len(all_labels[all_labels > 1e-8])
        mape = (mape_sum / valid_count * 100).item() if valid_count > 0 else float('inf')

        crps = 0.0
        head_name = self.model_cfg['regressor_head']['name']
        n_samples_k = all_preds_k_samples.shape[1]

        if head_name == 'diffusion' and n_samples_k > 1:
            if eval_phase_name == "Validation":
                logging.info(
                    f"[{eval_phase_name}] Calculating probabilistic metrics (K={n_samples_k}) on first 1000 samples...")
                sample_size = min(1000, len(all_labels))
                try:
                    crps = calculate_crps(all_preds_k_samples[:sample_size], all_labels[:sample_size])
                except Exception as e:
                    logging.error(f"Failed to calculate partial probabilistic metrics: {e}")
            else:  # 在 "Test" 阶段，计算所有
                logging.info(
                    f"[{eval_phase_name}] Calculating probabilistic metrics (K={n_samples_k}) on all samples...")
                try:
                    crps = calculate_crps(all_preds_k_samples, all_labels)
                except Exception as e:
                    logging.error(f"Failed to calculate full probabilistic metrics: {e}")
        else:
            if eval_phase_name == "Test":
                logging.info(f"Probabilistic metrics skipped (Model: {head_name}, K={n_samples_k})")

        metrics = {
            'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE': mape,
            'CRPS': crps,  # <-- 添加新指标
            'all_labels_for_plot': all_labels,
            'all_preds_for_plot': all_preds
        }
        return metrics


    def _save_checkpoint(self, filename, epoch, current_metric_val):
        fe_state_dict = self.feature_extractor.module.state_dict() if isinstance(self.feature_extractor,
                                                                                 nn.DataParallel) else self.feature_extractor.state_dict()
        reg_state_dict = self.regressor.module.state_dict() if isinstance(self.regressor,
                                                                          nn.DataParallel) else self.regressor.state_dict()
        state = {'epoch': epoch, 'model_state_dict': fe_state_dict, 'regressor_state_dict': reg_state_dict,
                 'optimizer_state_dict': self.optimizer.state_dict(), 'current_metric_val': current_metric_val,
                 'config': self.cfg}
        path = os.path.join(self.save_dir, filename)
        torch.save(state, path)
        logging.info(f"Checkpoint saved to {path} (Epoch: {epoch}, Current Val MAE: {current_metric_val:.4f})")

    def _get_inference_params(self):
        n_samples_k = self.cfg['evaluation'].get('inference_samples', 10)
        return {
            'n_samples': n_samples_k,
            'sampler': self.cfg['evaluation'].get('sampler', 'ddpm'),
            'ddim_steps': self.cfg['evaluation'].get('ddim_steps', 1000),
            'ddim_eta': self.cfg['evaluation'].get('ddim_eta', 0.0)
        }

    @torch.no_grad()
    def analyze_inference_performance(self):
        logging.info("======= 开始性能分析（单个视频） =======")
        self.feature_extractor.eval()
        self.regressor.eval()

        fe_model = self.feature_extractor.module if isinstance(self.feature_extractor, nn.DataParallel) else self.feature_extractor
        reg_model = self.regressor.module if isinstance(self.regressor, nn.DataParallel) else self.regressor

        head_name = self.model_cfg['regressor_head']['name']

        fe_params = sum(p.numel() for p in fe_model.parameters())
        reg_params = sum(p.numel() for p in reg_model.parameters())
        total_params = fe_params + reg_params

        logging.info(f"--- 1. 参数量 (Params) ---")
        logging.info(f"特征提取器: {fe_params / 1e6:.3f} M")
        logging.info(f"回归头:      {reg_params / 1e6:.3f} M")
        logging.info(f"总计:          {total_params / 1e6:.3f} M")
        logging.info("-" * 40)

        T = self.cfg['data'].get('frames', 32)
        H = self.cfg['data'].get('crop_size', 112)
        W = self.cfg['data'].get('crop_size', 112)

        dummy_video = torch.randn(1, 3, T, H, W).to(self.device)
        dummy_tabular = None

        if self.use_tabular_data:
            tabular_dim = self.data_cfg.get('tabular_config', {}).get('dim')
            if tabular_dim:
                dummy_tabular = torch.randn(1, tabular_dim).to(self.device)
            else:
                logging.error("use_tabular_data=True 但未在配置中找到 'tabular_config.dim'")

        processed_video = self._process_input_for_feature_extractor(dummy_video)

        inference_params = self._get_inference_params()
        n_samples_k = inference_params['n_samples']

        logging.info(f"--- 2. 计算量 (FLOPs) ---")

        fe_macs, _ = profile(fe_model, inputs=(processed_video, ), verbose=False)
        fe_flops = fe_macs * 2
        logging.info(f"特征提取器 (单次运行): {fe_flops / 1e9:.3f} GFLOPs")

        total_reg_flops = 0

        if head_name == 'diffusion':
            sampler = inference_params['sampler']
            num_steps = inference_params['ddim_steps'] if sampler == 'ddim' else reg_model.n_diffusion_steps

            dummy_features = torch.randn(1, self.output_dim).to(self.device)
            dummy_noisy_y = torch.randn(1, 1).to(self.device)
            dummy_t_int = torch.randint(0, reg_model.n_diffusion_steps, (1,), device=self.device)
            dummy_t_emb = reg_model.time_mlp(dummy_t_int)

            combined_cond = dummy_features
            if reg_model.use_tabular_data and dummy_tabular is not None:
                # 必须在 no_grad() 上下文之外运行，否则 tabular_encoder 不会被跟踪
                with torch.no_grad():
                    tabular_emb = reg_model.tabular_encoder(dummy_tabular)
                combined_cond = torch.cat([dummy_features, tabular_emb], dim=1)

            predictor_macs, _ = profile(
                reg_model.noise_predictor,
                inputs=(dummy_noisy_y, dummy_t_emb, combined_cond),
                verbose=False
            )
            flops_per_step = predictor_macs * 2
            total_reg_flops = flops_per_step * num_steps * n_samples_k

            logging.info(f"回归头 (Diffusion):")
            logging.info(f"  - 预测器 (单步): {flops_per_step / 1e9:.3f} GFLOPs")
            logging.info(f"  - 采样器: {sampler.upper()} @ {num_steps} 步, K={n_samples_k}")
            logging.info(f"  - 总计 (回归头): {total_reg_flops / 1e9:.3f} GFLOPs")

        else: # MLP Head
            dummy_features = torch.randn(1, self.output_dim).to(self.device)
            reg_macs, _ = profile(reg_model, inputs=(dummy_features,), verbose=False)
            total_reg_flops = reg_macs * 2
            n_samples_k = 1 # MLP K=1
            logging.info(f"回归头 (MLP) (单次运行): {total_reg_flops / 1e9:.3f} GFLOPs")

        # 3.3. 总 FLOPs
        total_flops = fe_flops + total_reg_flops
        logging.info(f"总计 (FLOPs): {total_flops / 1e12:.4f} TFLOPs")
        logging.info("-" * 40)

        logging.info(f"--- 4. GPU 内存统计 (VRAM) ---")
        vram_stats = {
            "model_vram_mb": 0,
            "peak_inference_vram_mb": 0,
            "peak_activation_vram_mb": 0
        }

        if self.device.type != 'cuda':
            logging.warning("未检测到 CUDA，跳过 VRAM 统计。")
        else:
            torch.cuda.synchronize(self.device)
            model_vram_bytes = torch.cuda.memory_allocated(self.device)
            model_vram_mb = model_vram_bytes / (1024 * 1024)
            logging.info(f"模型参数显存 (VRAM): {model_vram_mb:.2f} MB")

            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)

            features = fe_model(processed_video)

            if head_name == 'diffusion':
                _ = reg_model(
                    condition_x=features,
                    tabular_data=dummy_tabular,
                    mode='inference',
                    **inference_params
                )
            else:
                _ = reg_model(features)

            torch.cuda.synchronize(self.device)

            peak_vram_bytes = torch.cuda.max_memory_allocated(self.device)
            peak_vram_mb = peak_vram_bytes / (1024 * 1024)
            peak_activation_vram_mb = peak_vram_mb - model_vram_mb

            logging.info(f"峰值推理显存 (总计): {peak_vram_mb:.2f} MB")
            logging.info(f"峰值激活/缓存显存: {peak_activation_vram_mb:.2f} MB")

            vram_stats = {
                "model_vram_mb": model_vram_mb,
                "peak_inference_vram_mb": peak_vram_mb,
                "peak_activation_vram_mb": peak_activation_vram_mb
            }
            torch.cuda.reset_peak_memory_stats(self.device)

        logging.info("-" * 40)


        logging.info(f"--- 5. 推理时间 (Wall Clock) ---")

        logging.info(f"正在预热 GPU (5 次运行, K={n_samples_k})...")
        for _ in range(5):
            _features = fe_model(processed_video)

            if head_name == 'diffusion':
                _ = reg_model(
                    condition_x=_features,
                    tabular_data=dummy_tabular,
                    mode='inference',
                    **inference_params
                )
            else: # 'mlp'
                _ = reg_model(_features)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        start_time = time.time()

        features = fe_model(processed_video)

        if head_name == 'diffusion':
            _ = reg_model(
                condition_x=features,
                tabular_data=dummy_tabular,
                mode='inference',
                **inference_params
            )
        else: # 'mlp'
            _ = reg_model(features)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        end_time = time.time()
        total_time = end_time - start_time

        logging.info(f"总推理时间 (B=1, K={n_samples_k}): {total_time:.4f} 秒")
        logging.info("======= 性能分析结束 =======")

        final_results = {
            "total_params_M": total_params / 1e6,
            "fe_params_M": fe_params / 1e6,
            "reg_params_M": reg_params / 1e6,
            "total_flops_T": total_flops / 1e12,
            "fe_flops_G": fe_flops / 1e9,
            "reg_flops_G": total_reg_flops / 1e9,
            "total_time_s": total_time
        }

        final_results.update(vram_stats)

        return final_results


if __name__ == '__main__':
    import yaml

    config = 'configs/pediatric/stage2_uniformer_diffusion.yaml'

    try:
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        print(f"成功加载配置文件: {config}")
    except Exception as e:
        print(f"加载配置文件 {config} 失败: {e}")
        sys.exit(1)

    try:
        cfg['wandb']['use'] = False
        print("正在初始化 Trainer (将构建模型)...")
        trainer = Trainer(cfg)
        print("Trainer 初始化完毕。")
    except Exception as e:
        print(f"初始化 Trainer 失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    try:
        print("\n" + "=" * 50)
        trainer.analyze_inference_performance()
        print("=" * 50 + "\n")
    except Exception as e:
        print(f"运行分析时出错: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
