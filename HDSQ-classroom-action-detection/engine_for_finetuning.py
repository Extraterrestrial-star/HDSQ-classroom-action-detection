from itertools import count
import os
import numpy as np
import math
import sys
import time
import datetime
import logging
from typing import Iterable, Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import utils
from alphaction.modeling.utils import cat
from alphaction.structures.bounding_box import BoxList
from data.ava_eval import do_ava_evaluation
import pdb


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """KL divergence between independent Bernoulli distributions for multi-label distillation.
    p, q are probabilities in (0,1), same shape.
    Returns mean KL over all elements.
    """
  p = p.clamp(eps, 1.0 - eps)
  q = q.clamp(eps, 1.0 - eps)
  kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
  return kl.mean()


def train_class_batch(model, samples, boxes, criterion):
  """
    Supports:
    - Baseline single-head models: model(samples, boxes) -> logits [N, K]
    - Two-stream models (training mode): model(samples, boxes) -> dict with keys:
        logits_fused / logits_slow / logits_fast / aux
    """
  outputs = model(samples, boxes)
  labels = cat([proposal.get_field("labels") for proposal in boxes], dim=0).to(dtype=torch.float32)

  # DDP unwrap for reading hyper-params
  base_model = model.module if hasattr(model, "module") else model

  if isinstance(outputs, dict):
    z_fused = outputs["logits_fused"]
    z_fast = outputs["logits_fast"]

    assert z_fused.shape[1] == labels.shape[1], \
      f"Shape mismatch: fused logits {z_fused.shape} vs labels {labels.shape}. Check --nb_classes."
    assert z_fast.shape[1] == labels.shape[1], \
      f"Shape mismatch: fast logits {z_fast.shape} vs labels {labels.shape}. Check --nb_classes."

    # v3: L = ASL(z_fused) + lambda_fast*ASL(z_fast) + lambda_cons*KL(stopgrad(z_fused) || z_fast)
    loss_fused = criterion(z_fused, labels)
    loss_fast = criterion(z_fast, labels)
    lambda_fast = float(getattr(base_model, "lambda_fast", 0.0))
    lambda_cons = float(getattr(base_model, "lambda_cons", 0.0))
    loss = loss_fused + lambda_fast * loss_fast
    loss_dict = {
      "loss_fused": float(loss_fused.detach().item()),
      "loss_fast": float(loss_fast.detach().item()),
      "lambda_fast": lambda_fast,
    }
    if lambda_cons > 0:
      p_teacher = torch.sigmoid(z_fused.detach()).clamp(1e-6, 1.0 - 1e-6)
      p_fast = torch.sigmoid(z_fast).clamp(1e-6, 1.0 - 1e-6)
      loss_cons = bernoulli_kl(p_teacher, p_fast)
      loss = loss + lambda_cons * loss_cons
      loss_dict["loss_cons"] = float(loss_cons.detach().item())
      loss_dict["lambda_cons"] = lambda_cons
    aux = outputs.get("aux", {})
    if isinstance(aux, dict):
      for k, v in aux.items():
        try:
          loss_dict[k] = float(v.detach().item()) if torch.is_tensor(v) else float(v)
        except Exception:
          pass
      if "loss_codaq_total" in aux and torch.is_tensor(aux["loss_codaq_total"]):
        loss = loss + aux["loss_codaq_total"]
        loss_dict["loss_codaq_total"] = float(aux["loss_codaq_total"].detach().item())

    return loss, z_fused, loss_dict

  # ---- CoDA-Q single-stream: (logits, aux_losses) ----
  if isinstance(outputs, (tuple, list)) and len(outputs) == 2:
    outputs, aux_losses = outputs
    labels = cat([proposal.get_field("labels") for proposal in boxes], dim=0)
    assert outputs.shape[1] == labels.shape[1]
    loss_main = criterion(outputs, labels.to(dtype=torch.float32))
    loss_extra = aux_losses.get("loss_codaq_total", outputs.new_tensor(0.0))
    if torch.is_tensor(loss_extra):
      loss = loss_main + loss_extra
    else:
      loss = loss_main
    loss_dict = {"loss_main": float(loss_main.detach().item()),
                 "loss_extra": float(loss_extra.detach().item()) if torch.is_tensor(loss_extra) else 0.0}
    if isinstance(aux_losses, dict):
      for k in ("loss_codaq_cd", "loss_codaq_div", "lambda_cd_eff", "lambda_div_eff"):
        if k in aux_losses and torch.is_tensor(aux_losses[k]):
          loss_dict[k] = float(aux_losses[k].detach().item())
    return loss, outputs, loss_dict

  # ---- baseline single-head ----
  labels = cat([proposal.get_field("labels") for proposal in boxes], dim=0)
  assert outputs.shape[1] == labels.shape[1], \
    f"Shape mismatch: Model output has {outputs.shape[1]} classes, but labels have {labels.shape[1]} classes. " \
    f"Please check the --nb_classes argument in your training script."
  loss = criterion(outputs, labels.to(dtype=torch.float32))
  return loss, outputs, {"loss_fused": float(loss.detach().item())}


def get_loss_scale_for_deepspeed(model):
  optimizer = model.optimizer
  return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def _tensor_to_uint8_image(img_chw: torch.Tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
  if img_chw is None or (getattr(img_chw, "ndim", 0) != 3 or getattr(img_chw, "shape", [0])[0] != 3):
    return None
  img = img_chw.detach().float().cpu().clone() if hasattr(img_chw, "detach") else torch.as_tensor(
    img_chw).float().cpu().clone()
  for c in range(3):
    img[c] = img[c] * float(std[c]) + float(mean[c])
  img = img.clamp(0.0, 1.0)
  return (img * 255.0).byte().permute(1, 2, 0).numpy()


def _attn_vec_to_heatmap(attn_vec: np.ndarray, num_time: int, pooler_resolution: int,
                         num_local_tokens: int) -> np.ndarray:
  P, S = int(pooler_resolution), int(max(1, num_time))
  if getattr(attn_vec, "ndim", 0) != 1:
    attn_vec = attn_vec.reshape(-1)
  local = attn_vec[:min(int(num_local_tokens), attn_vec.size)].astype(np.float32)
  expected = S * P * P
  if local.shape[0] < expected:
    local = np.concatenate([local, np.zeros((expected - local.shape[0],), dtype=np.float32)], axis=0)
  local = local[:expected].reshape(S, P, P)
  return local.sum(axis=0)


def _save_overlay(frame_u8: np.ndarray, box_xyxy: np.ndarray, heat: np.ndarray, save_path: str, title: str = "",
                  alpha: float = 0.5) -> Optional[str]:
  """Produce a clear ROI-cropped overlay: student image with semi-transparent jet heatmap.

    The output is a tightly cropped ROI image (like the reference: clear person with
    colorful attention heatmap on top), saved at high resolution for paper quality.
    """
  try:
    import cv2
  except Exception:
    return None
  if frame_u8 is None or heat is None:
    return None
  H, W = frame_u8.shape[:2]
  x1, y1, x2, y2 = [float(v) for v in box_xyxy.tolist()]
  x1, x2 = int(max(0, min(W - 1, x1))), int(max(0, min(W, x2)))
  y1, y2 = int(max(0, min(H - 1, y1))), int(max(0, min(H, y2)))
  if x2 <= x1 + 1 or y2 <= y1 + 1:
    return None

  # Crop the ROI region from the original frame
  roi_crop = frame_u8[y1:y2, x1:x2].copy()
  crop_h, crop_w = roi_crop.shape[:2]

  # Normalize heatmap to [0, 1]
  heat = heat.astype(np.float32)
  heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

  # Upsample heatmap to crop size with smooth cubic interpolation
  heat_up = cv2.resize(heat, (crop_w, crop_h), interpolation=cv2.INTER_CUBIC)
  heat_up = np.clip(heat_up, 0.0, 1.0)

  # Apply jet colormap (produces BGR uint8)
  heat_color = cv2.applyColorMap((heat_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
  heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)  # to RGB

  # Blend: overlay = (1-alpha)*original + alpha*heatmap
  blended = (roi_crop.astype(np.float32) * (1.0 - alpha) + heat_color.astype(np.float32) * alpha)
  blended = np.clip(blended, 0, 255).astype(np.uint8)

  # Optionally upscale small crops for visual clarity (min 128px on short side)
  short_side = min(crop_h, crop_w)
  if short_side < 128:
    scale = 128.0 / short_side
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    blended = cv2.resize(blended, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

  # Save as high-quality PNG via cv2 (no matplotlib dependency for cleaner output)
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
  return save_path


def _save_codaq_overlay_images(debug_state: Dict[str, Any], debug_meta: Dict[str, Any], output_dir: str,
                               global_step: int, roi_index: int = 0) -> List[str]:
  """Generate per-query-type and combined attention overlay images for CoDA-Q.

    Produces:
    - overlay_local_corr_mean: correlation queries mean attention on local tokens
    - overlay_local_disc_mean: discriminative queries mean attention on local tokens
    - overlay_local_combined: combined (corr+disc) attention on local tokens
    - per-query overlays: overlay_corr_q{i}, overlay_disc_q{i}
    All cropped to the ROI bounding box for a clear student-level view.
    """
  paths: List[str] = []
  if not debug_state or not debug_meta:
    return paths
  attn_corr = debug_state.get("attn_corr")
  attn_disc = debug_state.get("attn_disc")
  if attn_corr is None or attn_disc is None:
    return paths
  attn_corr = attn_corr.numpy() if hasattr(attn_corr, "numpy") else attn_corr
  attn_disc = attn_disc.numpy() if hasattr(attn_disc, "numpy") else attn_disc
  frames = debug_meta.get("frames")
  boxes_xyxy = debug_meta.get("boxes_xyxy")
  if frames is None or boxes_xyxy is None or frames.shape[0] == 0:
    return paths
  roi_index = max(0, min(roi_index, int(frames.shape[0]) - 1))
  frame_u8 = _tensor_to_uint8_image(frames[roi_index])
  box_xyxy = boxes_xyxy[roi_index].numpy() if hasattr(boxes_xyxy[roi_index], "numpy") else np.asarray(
    boxes_xyxy[roi_index])
  P = int(debug_state.get("pooler_resolution", 7))
  S = int(debug_state.get("num_time", 1))
  num_local_tokens = int(debug_state.get("num_local_tokens", S * P * P))
  vis_dir = os.path.join(output_dir, "codaq_vis")
  os.makedirs(vis_dir, exist_ok=True)

  # Mean attention across queries
  corr_vec = attn_corr[roi_index].mean(axis=0)
  disc_vec = attn_disc[roi_index].mean(axis=0)
  heat_corr = _attn_vec_to_heatmap(corr_vec, S, P, num_local_tokens)
  heat_disc = _attn_vec_to_heatmap(disc_vec, S, P, num_local_tokens)
  # Combined attention: average of corr and disc
  heat_combined = (heat_corr + heat_disc) / 2.0

  for name, heat, title in [
    ("overlay_local_corr_mean", heat_corr, "Corr"),
    ("overlay_local_disc_mean", heat_disc, "Disc"),
    ("overlay_local_combined", heat_combined, "Combined"),
  ]:
    p = _save_overlay(frame_u8, box_xyxy, heat,
                      os.path.join(vis_dir, f"{name}_step{global_step}_roi{roi_index}.png"),
                      title=f"step={global_step} roi={roi_index} {title}")
    if p:
      paths.append(p)

  # Per-query overlays (individual corr / disc queries)
  num_corr_q = attn_corr[roi_index].shape[0]
  num_disc_q = attn_disc[roi_index].shape[0]
  for qi in range(num_corr_q):
    heat_q = _attn_vec_to_heatmap(attn_corr[roi_index][qi], S, P, num_local_tokens)
    p = _save_overlay(frame_u8, box_xyxy, heat_q,
                      os.path.join(vis_dir, f"overlay_corr_q{qi}_step{global_step}_roi{roi_index}.png"),
                      title=f"Corr-Q{qi}")
    if p:
      paths.append(p)
  for qi in range(num_disc_q):
    heat_q = _attn_vec_to_heatmap(attn_disc[roi_index][qi], S, P, num_local_tokens)
    p = _save_overlay(frame_u8, box_xyxy, heat_q,
                      os.path.join(vis_dir, f"overlay_disc_q{qi}_step{global_step}_roi{roi_index}.png"),
                      title=f"Disc-Q{qi}")
    if p:
      paths.append(p)

  return paths


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, criterion=None, args=None,
                    lambda_fast_schedule=None, lambda_cons_schedule=None):
  model.train(True)
  metric_logger = utils.MetricLogger(delimiter="  ")
  metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
  metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
  header = 'Epoch: [{}]'.format(epoch)
  print_freq = 10

  optimizer.zero_grad()

  for data_iter_step, (samples, boxes, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    step = data_iter_step // update_freq
    if step >= num_training_steps_per_epoch:
      continue
    it = start_steps + step

    if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
      for i, param_group in enumerate(optimizer.param_groups):
        if lr_schedule_values is not None:
          param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
        if wd_schedule_values is not None and param_group["weight_decay"] > 0:
          param_group["weight_decay"] = wd_schedule_values[it]

    samples = samples.to(device, non_blocking=True)
    boxes = [box.to(device=device) for box in boxes]

    model_without_ddp = model.module if hasattr(model, "module") else model
    if hasattr(model_without_ddp, "set_codaq_step"):
      model_without_ddp.set_codaq_step(it)
    enable_vis = (args is not None and getattr(args, "codaq_vis_freq", 0) > 0 and utils.is_main_process() and (
              it % int(getattr(args, "codaq_vis_freq", 0)) == 0))
    if hasattr(model_without_ddp, "set_codaq_debug"):
      model_without_ddp.set_codaq_debug(enable_vis, max_rois=int(getattr(args, "codaq_vis_max_rois", 4)) if args else 4)

    if loss_scaler is not None:
      with torch.cuda.amp.autocast():
        loss, _, loss_dict = train_class_batch(model, samples, boxes, criterion)
    else:
      loss, _, loss_dict = train_class_batch(model, samples, boxes, criterion)

    if enable_vis and args and getattr(args, "output_dir", ""):
      debug_state = model_without_ddp.pop_codaq_debug_state() if hasattr(model_without_ddp,
                                                                         "pop_codaq_debug_state") else None
      debug_meta = model_without_ddp.pop_codaq_debug_meta() if hasattr(model_without_ddp,
                                                                       "pop_codaq_debug_meta") else None
      if debug_state and debug_meta:
        _save_codaq_overlay_images(debug_state, debug_meta, args.output_dir, it,
                                   int(getattr(args, "codaq_vis_roi_index", 0)))

    loss_value = loss.item()

    if not math.isfinite(loss_value):
      print("Loss is {}, stopping training".format(loss_value))
      sys.exit(1)

    if loss_scaler is not None:
      # AMP backward pass
      is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
      loss /= update_freq
      grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                              parameters=model.parameters(), create_graph=is_second_order,
                              update_grad=(data_iter_step + 1) % update_freq == 0)
      if (data_iter_step + 1) % update_freq == 0:
        optimizer.zero_grad()
        if model_ema is not None:
          model_ema.update(model)
      loss_scale_value = loss_scaler.state_dict()["scale"]
    else:
      # 手动 FP32 反向传播路径
      loss /= update_freq
      loss.backward()
      if (data_iter_step + 1) % update_freq == 0:
        if max_norm is not None:
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        optimizer.zero_grad()
        if model_ema is not None:
          model_ema.update(model)
      grad_norm = None
      loss_scale_value = None  # 在FP32模式下，没有loss_scale_value

    torch.cuda.synchronize()

    metric_logger.update(loss=loss_value)
    # extra metrics for two-stream training
    if 'loss_dict' in locals() and isinstance(loss_dict, dict):
      metric_logger.update(**loss_dict)
    if loss_scale_value is not None:
      metric_logger.update(loss_scale=loss_scale_value)

    min_lr, max_lr = 10., 0.
    for group in optimizer.param_groups:
      min_lr = min(min_lr, group["lr"])
      max_lr = max(max_lr, group["lr"])

    metric_logger.update(lr=max_lr)
    metric_logger.update(min_lr=min_lr)
    weight_decay_value = None
    for group in optimizer.param_groups:
      if group["weight_decay"] > 0:
        weight_decay_value = group["weight_decay"]
    metric_logger.update(weight_decay=weight_decay_value)
    metric_logger.update(grad_norm=grad_norm)

    if log_writer is not None:
      log_writer.update(loss=loss_value, head="loss")
      if 'loss_dict' in locals() and isinstance(loss_dict, dict):
        for k in ['loss_fused', 'loss_fast', 'loss_cons', 'loss_main', 'loss_extra', 'gate_mean', 'gamma', 'gamma_map',
                  'token_norm_scale', 'drop_fast', 'loss_codaq_cd', 'loss_codaq_div', 'lambda_cd_eff',
                  'lambda_div_eff']:
          if k in loss_dict:
            log_writer.update(**{k: loss_dict[k]}, head='loss')
      if loss_scale_value is not None:
        log_writer.update(loss_scale=loss_scale_value, head="opt")
      log_writer.update(lr=max_lr, head="opt")
      log_writer.update(min_lr=min_lr, head="opt")
      log_writer.update(weight_decay=weight_decay_value, head="opt")
      log_writer.update(grad_norm=grad_norm, head="opt")
      log_writer.set_step()

  metric_logger.synchronize_between_processes()
  print("Averaged stats:", metric_logger)
  return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class PostProcessor(nn.Module):
  def forward(self, class_logits, boxes):
    class_logits = torch.sigmoid(class_logits)
    box_scores = cat([box.get_field("scores") for box in boxes], dim=0)
    box_scores = box_scores.reshape(class_logits.shape[0], 1)
    action_prob = class_logits * box_scores

    image_shapes = [box.size for box in boxes]
    boxes_per_image = [len(box) for box in boxes]
    box_tensors = [a.bbox for a in boxes]

    action_prob = action_prob.split(boxes_per_image, dim=0)

    results = []
    for prob, boxes_per_image, image_shape in zip(
            action_prob, box_tensors, image_shapes
    ):
      boxlist = self.prepare_boxlist(boxes_per_image, prob, image_shape)
      results.append(boxlist)
    return results

  def prepare_boxlist(self, boxes, scores, image_shape):
    boxlist = BoxList(boxes, image_shape, mode="xyxy")
    boxlist.add_field("scores", scores)
    return boxlist


@torch.no_grad()
def validation_one_epoch(data_loader, model, device, output_dir, epoch, log_writer=None):
  if not utils.is_main_process():
    return

  metric_logger = utils.MetricLogger(delimiter="  ")
  header = 'Val:'
  model.eval()

  logging.info("Start evaluation on ava_v2.2 dataset({} videos).".format(data_loader.num_samples))
  start_time = time.time()

  cpu_device = torch.device("cpu")
  results_dict = {}
  # Two-stream: separate result dicts for fused / slow / fast
  results_dict_slow = {}
  results_dict_fast = {}
  postprocess = PostProcessor()
  for batch in metric_logger.log_every(data_loader, 10, header):
    videos = batch[0]
    boxes = batch[1]
    video_ids = batch[2]

    videos = videos.to(device, non_blocking=True)
    boxes = [box.to(device=device) for box in boxes]

    output = model(videos, boxes)
    if isinstance(output, dict) and "logits_fused" in output and "logits_fast" in output:
      # Two-stream (v2: fused + fast only)
      out_fused = postprocess(output["logits_fused"], boxes)
      out_fast = postprocess(output["logits_fast"], boxes)
      out_fused = [o.to(cpu_device) for o in out_fused]
      out_fast = [o.to(cpu_device) for o in out_fast]
      results_dict.update({vid: r for vid, r in zip(video_ids, out_fused)})
      results_dict_fast.update({vid: r for vid, r in zip(video_ids, out_fast)})
      if "logits_slow" in output:
        out_slow = postprocess(output["logits_slow"], boxes)
        out_slow = [o.to(cpu_device) for o in out_slow]
        results_dict_slow.update({vid: r for vid, r in zip(video_ids, out_slow)})
    else:
      # Single-head baseline or CoDA-Q (logits, aux)
      if isinstance(output, (tuple, list)) and len(output) == 2:
        output = output[0]
      output = postprocess(output, boxes)
      output = [o.to(cpu_device) for o in output]
      results_dict.update({video_id: result for video_id, result in zip(video_ids, output)})

  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=total_time))
  logging.info(
    "Total inference time: {}".format(total_time_str)
  )

  video_ids = list(sorted(results_dict.keys()))
  if len(video_ids) != video_ids[-1] + 1:
    logging.warning(
      "Number of videos that were gathered from multiple processes is not "
      "a contiguous set. Some images might be missing from the evaluation"
    )
  predictions = [results_dict[i] for i in video_ids]

  logging.info("Performing ava evaluation")

  output_folder = os.path.join(output_dir, "inference")
  os.makedirs(output_folder, exist_ok=True)

  eval_res = do_ava_evaluation(
    dataset=data_loader.dataset,
    predictions=predictions,
    output_folder=output_folder,
  )

  # Two-stream v3: evaluate fused + slow + fast mAP
  if results_dict_fast:
    predictions_fast = [results_dict_fast[i] for i in video_ids]
    output_folder_fast = os.path.join(output_dir, "inference_fast")
    os.makedirs(output_folder_fast, exist_ok=True)
    eval_res_fast = do_ava_evaluation(
      dataset=data_loader.dataset,
      predictions=predictions_fast,
      output_folder=output_folder_fast,
    )
    eval_res, _ = eval_res
    eval_res_fast, _ = eval_res_fast
    total_mAP = eval_res['PascalBoxes_Precision/mAP@0.5IOU']
    mAP_fast = eval_res_fast['PascalBoxes_Precision/mAP@0.5IOU']
    if results_dict_slow:
      predictions_slow = [results_dict_slow[i] for i in video_ids]
      output_folder_slow = os.path.join(output_dir, "inference_slow")
      os.makedirs(output_folder_slow, exist_ok=True)
      eval_res_slow = do_ava_evaluation(
        dataset=data_loader.dataset,
        predictions=predictions_slow,
        output_folder=output_folder_slow,
      )
      eval_res_slow, _ = eval_res_slow
      mAP_slow = eval_res_slow['PascalBoxes_Precision/mAP@0.5IOU']
      logging.info("mAP fused=%.4f | slow=%.4f | fast=%.4f", total_mAP, mAP_slow, mAP_fast)
      if log_writer is not None:
        log_writer.update(map=total_mAP, map_slow=mAP_slow, map_fast=mAP_fast, head="perf", step=epoch)
    else:
      logging.info("mAP fused=%.4f | fast=%.4f", total_mAP, mAP_fast)
      if log_writer is not None:
        log_writer.update(map=total_mAP, map_fast=mAP_fast, head="perf", step=epoch)
  else:
    if log_writer is not None:
      eval_res, _ = eval_res
      total_mAP = eval_res['PascalBoxes_Precision/mAP@0.5IOU']
      log_writer.update(map=total_mAP, head="perf", step=epoch)


# --- START: 重新添加的函数 ---
# 以下是之前被省略，但 run_class_finetuning.py 需要导入的函数
@torch.no_grad()
def final_test(data_loader, model, device, file):
  criterion = torch.nn.CrossEntropyLoss()

  metric_logger = utils.MetricLogger(delimiter="  ")
  header = 'Test:'

  # switch to evaluation mode
  model.eval()
  final_result = []

  for batch in metric_logger.log_every(data_loader, 10, header):
    videos = batch[0]
    target = batch[1]
    ids = batch[2]
    chunk_nb = batch[3]
    split_nb = batch[4]
    videos = videos.to(device, non_blocking=True)
    target = target.to(device, non_blocking=True)

    # compute output
    with torch.cuda.amp.autocast():
      output = model(videos)
      loss = criterion(output, target)

    for i in range(output.size(0)):
      string = "{} {} {} {} {}\n".format(ids[i], \
                                         str(output.data[i].cpu().numpy().tolist()), \
                                         str(int(target[i].cpu().numpy())), \
                                         str(int(chunk_nb[i].cpu().numpy())), \
                                         str(int(split_nb[i].cpu().numpy())))
      final_result.append(string)

    acc1, acc5 = accuracy(output, target, topk=(1, 5))

    batch_size = videos.shape[0]
    metric_logger.update(loss=loss.item())
    metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
    metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

  if not os.path.exists(file):
    os.mknod(file)
  with open(file, 'w') as f:
    f.write("{}, {}\n".format(acc1, acc5))
    for line in final_result:
      f.write(line)
  # gather the stats from all processes
  metric_logger.synchronize_between_processes()
  print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

  return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
  dict_feats = {}
  dict_label = {}
  dict_pos = {}
  print("Reading individual output files")

  for x in range(num_tasks):
    file = os.path.join(eval_path, str(x) + '.txt')
    lines = open(file, 'r').readlines()[1:]
    for line in lines:
      line = line.strip()
      name = line.split('[')[0]
      label = line.split(']')[1].split(' ')[1]
      chunk_nb = line.split(']')[1].split(' ')[2]
      split_nb = line.split(']')[1].split(' ')[3]
      data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
      if not name in dict_feats:
        dict_feats[name] = []
        dict_label[name] = 0
        dict_pos[name] = []
      if chunk_nb + split_nb in dict_pos[name]:
        continue
      dict_feats[name].append(data)
      dict_pos[name].append(chunk_nb + split_nb)
      dict_label[name] = label
  print("Computing final results")

  input_lst = []
  print(len(dict_feats))
  for i, item in enumerate(dict_feats):
    input_lst.append([i, item, dict_feats[item], dict_label[item]])
  from multiprocessing import Pool
  p = Pool(64)
  ans = p.map(compute_video, input_lst)
  top1 = [x[1] for x in ans]
  top5 = [x[2] for x in ans]
  pred = [x[0] for x in ans]
  label = [x[3] for x in ans]
  final_top1, final_top5 = np.mean(top1), np.mean(top5)
  return final_top1 * 100, final_top5 * 100


def compute_video(lst):
  i, video_id, data, label = lst
  feat = [x for x in data]
  feat = np.mean(feat, axis=0)
  pred = np.argmax(feat)
  top1 = (int(pred) == int(label)) * 1.0
  top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
  return [pred, top1, top5, int(label)]
# --- END: 重新添加的函数 ---