import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE

# Import base classes with an alias to avoid name conflicts
from data.ava import AVAVideoDataset as AVABaseDataset, KineticsDataset, AKDataset
from data.transforms import TransformsCfg
import alphaction.config.paths_catalog as paths_catalog
from alphaction.dataset.collate_batch import batch_different_videos

# Imports for frame reading and robustness
from PIL import Image
import glob
import numpy as np
import torch
import inspect


class DataAugmentationForVideoMAE(object):
  def __init__(self, args):
    self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
    self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
    normalize = GroupNormalize(self.input_mean, self.input_std)
    self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
    self.transform = transforms.Compose([
      self.train_augmentation,
      Stack(roll=False),
      ToTorchFormatTensor(div=True),
      normalize,
    ])
    if getattr(args, "mask_type", None) == 'tube':
      self.masked_position_generator = TubeMaskingGenerator(
        args.window_size, args.mask_ratio
      )
    else:
      self.masked_position_generator = None

  def __call__(self, images):
    process_data, _ = self.transform(images)
    if self.masked_position_generator is None:
      return process_data, None
    return process_data, self.masked_position_generator()

  def __repr__(self):
    repr = "(DataAugmentationForVideoMAE,\n"
    repr += "  transform = %s,\n" % str(self.transform)
    repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
    repr += ")"
    return repr


def build_pretraining_dataset(args):
  transform = DataAugmentationForVideoMAE(args)
  dataset = VideoMAE(
    root=None,
    setting=args.data_path,
    video_ext='mp4',
    is_color=True,
    modality='rgb',
    new_length=args.num_frames,
    new_step=args.sampling_rate,
    transform=transform,
    temporal_jitter=False,
    video_loader=True,
    use_decord=True,
    lazy_init=False)
  print("Data Aug = %s" % str(transform))
  return dataset


def build_dataset(is_train, test_mode, args):
  if args.data_set == 'Kinetics-400':
    mode = None
    anno_path = None
    if is_train is True:
      mode = 'train'
      anno_path = os.path.join(args.data_path, 'train.csv')
    elif test_mode is True:
      mode = 'test'
      anno_path = os.path.join(args.data_path, 'val.csv')
    else:
      mode = 'validation'
      anno_path = os.path.join(args.data_path, 'test.csv')

    dataset = VideoClsDataset(
      anno_path=anno_path,
      data_path='/',
      mode=mode,
      clip_len=args.num_frames,
      frame_sample_rate=args.sampling_rate,
      num_segment=1,
      test_num_segment=args.test_num_segment,
      test_num_crop=args.test_num_crop,
      num_crop=1 if not test_mode else 3,
      keep_aspect_ratio=True,
      crop_size=args.input_size,
      short_side_size=args.short_side_size,
      new_height=256,
      new_width=320,
      args=args)
    nb_classes = 400

  elif args.data_set == 'SSV2':
    mode = None
    anno_path = None
    if is_train is True:
      mode = 'train'
      anno_path = os.path.join(args.data_path, 'train.csv')
    elif test_mode is True:
      mode = 'test'
      anno_path = os.path.join(args.data_path, 'val.csv')
    else:
      mode = 'validation'
      anno_path = os.path.join(args.data_path, 'test.csv')

    dataset = VideoClsDataset(
      anno_path=anno_path,
      data_path='/',
      mode=mode,
      clip_len=args.num_frames,
      frame_sample_rate=args.sampling_rate,
      num_segment=1,
      test_num_segment=args.test_num_segment,
      test_num_crop=args.test_num_crop,
      num_crop=1 if not test_mode else 3,
      keep_aspect_ratio=True,
      crop_size=args.input_size,
      short_side_size=args.short_side_size,
      new_height=256,
      new_width=320,
      args=args)
    nb_classes = 174

  elif args.data_set == 'UCF101':
    mode = None
    anno_path = None
    if is_train is True:
      mode = 'train'
      anno_path = os.path.join(args.data_path, 'train.csv')
    elif test_mode is True:
      mode = 'test'
      anno_path = os.path.join(args.data_path, 'val.csv')
    else:
      mode = 'validation'
      anno_path = os.path.join(args.data_path, 'test.csv')

    dataset = VideoClsDataset(
      anno_path=anno_path,
      data_path='/',
      mode=mode,
      clip_len=args.num_frames,
      frame_sample_rate=args.sampling_rate,
      num_segment=1,
      test_num_segment=args.test_num_segment,
      test_num_crop=args.test_num_crop,
      num_crop=1 if not test_mode else 3,
      keep_aspect_ratio=True,
      crop_size=args.input_size,
      short_side_size=args.short_side_size,
      new_height=256,
      new_width=320,
      args=args)
    nb_classes = 101

  elif args.data_set == 'HMDB51':
    mode = None
    anno_path = None
    if is_train is True:
      mode = 'train'
      anno_path = os.path.join(args.data_path, 'train.csv')
    elif test_mode is True:
      mode = 'test'
      anno_path = os.path.join(args.data_path, 'val.csv')
    else:
      mode = 'validation'
      anno_path = os.path.join(args.data_path, 'test.csv')

    dataset = VideoClsDataset(
      anno_path=anno_path,
      data_path='/',
      mode=mode,
      clip_len=args.num_frames,
      frame_sample_rate=args.sampling_rate,
      num_segment=1,
      test_num_segment=args.test_num_segment,
      test_num_crop=args.test_num_crop,
      num_crop=1 if not test_mode else 3,
      keep_aspect_ratio=True,
      crop_size=args.input_size,
      short_side_size=args.short_side_size,
      new_height=256,
      new_width=320,
      args=args)
    nb_classes = 51
  else:
    raise NotImplementedError()
  assert nb_classes == args.nb_classes
  print("Number of the class = %d" % args.nb_classes)

  return dataset, nb_classes


class BatchCollator(object):
  """
    From a list of samples from the dataset,
    returns the batched objectimages and targets.
    This should be passed to the DataLoader
    """

  def __init__(self, size_divisible=0):
    self.divisible = size_divisible
    self.size_divisible = self.divisible

  def __call__(self, batch):
    transposed_batch = list(zip(*batch))
    video_data = batch_different_videos(transposed_batch[0], self.size_divisible)
    boxes = transposed_batch[1]
    video_ids = transposed_batch[2]
    return video_data, boxes, video_ids


# =============================================================================
# --- START OF MODIFIED SECTION ---
# The primary changes are in this AVAVideoDataset class and the
# build_ava_dataset function that calls it.
# =============================================================================
class AVAVideoDataset(AVABaseDataset):
  """
    Adapter for SAV dataset (AVA format, 3s clips, 15 classes, multi-label).
    - Reads from pre-extracted frames when image_mode=True.
    - Truncates box labels to the specified num_classes (e.g., 15 for SAV).
    - Safely handles different __init__ signatures from the parent class.
    """

  def __init__(
          self,
          video_root,
          ann_file,
          remove_clips_without_annotations,
          frame_span,
          box_file=None,
          eval_file_paths={},
          box_thresh=0.0,
          action_thresh=0.0,
          transforms=None,
          object_file=None,
          object_transforms=None,
          image_mode=False,
          num_classes=15,
          img_tmpl="img_{:05d}.jpg",
          **kwargs
  ):
    # Safely pass only the arguments supported by the parent constructor
    base_sig = inspect.signature(AVABaseDataset.__init__)
    allowed = set(base_sig.parameters.keys())
    base_kwargs = {
      "video_root": video_root,
      "ann_file": ann_file,
      "remove_clips_without_annotations": remove_clips_without_annotations,
      "frame_span": frame_span,
      "box_file": box_file,
      "eval_file_paths": eval_file_paths,
      "box_thresh": box_thresh,
      "action_thresh": action_thresh,
      "transforms": transforms,
      **({"object_file": object_file} if "object_file" in allowed else {}),
      **({"object_transforms": object_transforms} if "object_transforms" in allowed else {}),
    }
    super().__init__(**base_kwargs)

    # Store custom settings
    self._image_mode = image_mode
    self._num_classes_override = num_classes
    self._img_tmpl = img_tmpl

  def _decode_video_data(self, dirname, timestamp):
    # If in image mode, read from frame directories; otherwise, use parent's video decoder
    if self._image_mode:
      return self._decode_from_frames(dirname)
    return super()._decode_video_data(dirname, timestamp)

  def _decode_from_frames(self, dirname):
    folder = os.path.join(self.video_root, dirname)
    # More robust glob to find common image extensions
    frame_paths = sorted(glob.glob(os.path.join(folder, "*.[jp][pn]g")) + glob.glob(os.path.join(folder, "*.bmp")))

    if len(frame_paths) == 0:
      raise RuntimeError(f"No frames found in {folder}")

    num_total = len(frame_paths)
    num_need = self.frame_span if (self.frame_span is not None and self.frame_span > 0) else num_total

    idxs = np.linspace(0, num_total - 1, num=num_need, dtype=np.int64)
    frames = [np.array(Image.open(frame_paths[i]).convert("RGB")) for i in idxs]
    return np.stack(frames, axis=0)  # [T, H, W, 3]

  def _truncate_boxes_labels(self, boxes):
    # Truncates the one-hot label tensor to the required number of classes
    if boxes is not None and hasattr(boxes, "has_field") and boxes.has_field("labels"):
      lbl = boxes.get_field("labels")
      if hasattr(lbl, "dim") and lbl.dim() == 2 and lbl.size(1) > self._num_classes_override:
        boxes.add_field("labels", lbl[:, :self._num_classes_override])
    return boxes

  # This is the new, more robust __getitem__ method.
  def __getitem__(self, idx):
    # Get the data tuple from the parent class
    ret_tuple = super().__getitem__(idx)

    # If it's not a tuple, we can't process it, so return it as is.
    if not isinstance(ret_tuple, tuple):
      return ret_tuple

    # Iterate through the items returned by the data loader
    new_ret_list = []
    for item in ret_tuple:
      # Check if the item is a BoxList object that has a "labels" field
      # This is a safe way to identify the object we need to modify
      if hasattr(item, "has_field") and callable(getattr(item, "has_field")) and item.has_field("labels"):
        # If it is, truncate the labels and add it to our new list
        truncated_item = self._truncate_boxes_labels(item)
        new_ret_list.append(truncated_item)
      else:
        # Otherwise, add the item to the list without modification
        new_ret_list.append(item)

    # Return the data as a tuple with the corrected labels
    return tuple(new_ret_list)


def build_ava_dataset(is_train, transforms, args=None):
  """
    Builds the dataset for SAV (AVA format).
    - Uses paths_catalog for configuration.
    - Sets defaults for image_mode, num_classes, and img_tmpl for robustness.
    """
  input_filename = 'ava_video_train_v2.2' if is_train else 'ava_video_val_v2.2'
  assert input_filename
  dataset_catalog = paths_catalog.DatasetCatalog
  data = dataset_catalog.get(input_filename)
  ava_args = data["args"]

  ava_args["remove_clips_without_annotations"] = is_train

  # If using frame reading, frame_span is the number of frames to sample
  if ava_args.get("image_mode", True):
    # In image_mode, we read from pre-extracted frames.
    # Baseline uses FRAME_NUM (e.g., 16). Two-stream uses a longer aligned window (e.g., 64).
    if args is not None and getattr(args, "dual_stream", False):
      slow_frames = int(getattr(args, "slow_frames", 16))
      slow_stride = int(getattr(args, "slow_stride", 4))
      ava_args["frame_span"] = slow_frames * slow_stride  # default 64
    else:
      ava_args["frame_span"] = getattr(TransformsCfg, "FRAME_NUM", 16)
  else:
    # For video decode mode, frame_span already covers a longer window.
    ava_args["frame_span"] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE

  if not is_train:
    ava_args["box_thresh"] = 0.8
    ava_args["action_thresh"] = 0.0
  else:
    ava_args["box_file"] = None  # Use ground-truth boxes for training

  # Set defaults to ensure the dataset class gets the required parameters
  ava_args.setdefault("image_mode", True)
  ava_args.setdefault("num_classes", 15)  # Critical for SAV dataset
  ava_args.setdefault("img_tmpl", "img_{:05d}.jpg")
  ava_args["transforms"] = transforms

  # Use our robust adapter class
  dataset = AVAVideoDataset(**ava_args)
  return dataset


# =============================================================================
# --- END OF MODIFIED SECTION ---
# =============================================================================


def build_kinetics_dataset(is_train, transforms):
  input_filename = 'ava_video_train_v2.2' if is_train else 'ava_video_val_v2.2'
  kinetics_args = {}
  if is_train:
    kinetics_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_train_v1.0.json"
    kinetics_args['box_file'] = None
    kinetics_args['transforms'] = transforms
    kinetics_args['remove_clips_without_annotations'] = True
    kinetics_args['frame_span'] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE  # 64
  else:
    kinetics_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_val_v1.0.json"
    kinetics_args['box_file'] = '/mnt/cache/xingsen/xingsen2/kinetics_person_box.json'
    kinetics_args['box_thresh'] = 0.8
    kinetics_args['action_thresh'] = 0.
    kinetics_args['transforms'] = transforms
    kinetics_args['remove_clips_without_annotations'] = True
    kinetics_args['frame_span'] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE  # 64
    kinetics_args['eval_file_paths'] = {
      "csv_gt_file": '/mnt/cache/xingsen/xingsen2/kinetics_val_gt_v1.0.csv',
      "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
      "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
    }
  dataset = KineticsDataset(**kinetics_args)
  return dataset


def build_ak_dataset(is_train, transforms):
  input_filename = "ak_train" if is_train else "ak_val"
  assert input_filename
  dataset_catalog = paths_catalog.DatasetCatalog
  data_args = dataset_catalog.get(input_filename)
  if is_train:
    data_args['video_root'] = '/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval/'
    data_args['ava_annfile'] = "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2_min.json"
    data_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_train_v1.0.json"
    data_args['transforms'] = transforms
    data_args['remove_clips_without_annotations'] = True
    data_args['frame_span'] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE  # 64
    data_args["ava_eval_file_path"] = {
      "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_v2.2.csv",
      "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
      "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_train_excluded_timestamps_v2.2.csv",
    }
  else:
    data_args['video_root'] = '/mnt/cache/xingsen/ava_dataset/AVA/clips/trainval/'
    data_args['ava_annfile'] = "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2_min.json"
    data_args['kinetics_annfile'] = "/mnt/cache/xingsen/xingsen2/kinetics_val_v1.0.json"
    data_args['box_thresh'] = 0.8
    data_args['action_thresh'] = 0.
    data_args['transforms'] = transforms
    data_args['remove_clips_without_annotations'] = True
    data_args['kinetics_box_file'] = '/mnt/cache/xingsen/xingsen2/kinetics_person_box.json'
    data_args['ava_box_file'] = '/mnt/cache/xingsen/ava_dataset/AVA/boxes/ava_val_det_person_bbox.json'
    data_args['frame_span'] = TransformsCfg.FRAME_NUM * TransformsCfg.FRAME_SAMPLE_RATE  # 64
    data_args['eval_file_paths'] = {
      "csv_gt_file": '/mnt/cache/xingsen/xingsen2/ak_val_gt.csv',
      "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
      "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
    }
    data_args["ava_eval_file_path"] = {
      "csv_gt_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_v2.2.csv",
      "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
      "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
    }
    data_args['kinetics_eval_file_path'] = {
      "csv_gt_file": '/mnt/cache/xingsen/xingsen2/kinetics_val_gt_v1.0.csv',
      "labelmap_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt",
      "exclusion_file": "/mnt/cache/xingsen/ava_dataset/AVA/annotations/ava_val_excluded_timestamps_v2.2.csv",
    }

  dataset = AKDataset(**data_args)
  return dataset