import argparse
import os
from dataclasses import dataclass
from pathlib import Path


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in ("yes", "true", "t", "1"):
        return True
    if normalized in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected")


@dataclass(frozen=True)
class PolicyConfig:
    horizontal_step: float = 2.0
    vertical_step: float = 1.0
    rotate_angle: float = 15.0
    is_fixed: bool = True
    safety_margin: float = 0.5
    collision_percentile: float = 5.0
    approach_score: float = 0.225
    stop_score: float = 0.255
    min_specificity: float = -0.005
    stop_specificity: float = 0.005
    stop_category_score: float = 0.28
    stop_detail_score: float = 0.245
    direction_margin: float = 0.006
    quick_stop_score: float = 0.285
    quick_stop_specificity: float = 0.045
    quick_stop_min_depth: float = 3.0
    quick_stop_depth: float = 10.0
    quick_stop_enabled: bool = False
    min_stop_depth: float = 3.0
    stop_depth: float = 10.0
    spatial_stop_enabled: bool = False
    min_stop_depth_support: float = 0.20
    stop_confirm_frames: int = 1
    approach_confirm_frames: int = 2
    lost_target_frames: int = 3
    scan_turns: int = 0
    periodic_scan_turns: int = 1
    search_moves_per_scan: int = 12
    search_translation_budget: int = 32
    recovery_rotation_limit: int = 6
    visit_cell_size: float = 5.0
    search_radius: float = 50.0
    image_batch_size: int = 64
    world_model_enabled: bool = True
    voxel_xy_cells: int = 24
    voxel_z_cells: int = 8
    voxel_size: float = 1.0
    voxel_max_depth: float = 20.0
    voxel_samples_per_axis: int = 12
    jepa_latent_dim: int = 48
    jepa_hidden_dim: int = 96
    jepa_learning_rate: float = 0.0003
    jepa_regularization: float = 0.05
    jepa_online_training: bool = True
    jepa_replay_capacity: int = 512
    jepa_batch_size: int = 8
    jepa_planning_horizon: int = 6
    jepa_beam_width: int = 8
    jepa_trust_transitions: int = 32
    jepa_collision_weight: float = 6.0
    jepa_revisit_weight: float = 0.8
    jepa_semantic_weight: float = 3.0
    jepa_uncertainty_weight: float = 0.05
    jepa_latent_novelty_weight: float = 0.1
    jepa_override_margin: float = 2.0
    jepa_override_min_risk_reduction: float = 0.0
    jepa_hard_collision_risk: float = 0.20
    jepa_hard_collision_enabled: bool = False
    jepa_checkpoint_path: str = ""
    jepa_semantic_memory_weight: float = 2.0
    jepa_goal_latent_weight: float = 0.0
    jepa_goal_min_score: float = 0.23


def build_parser():
    package_root = Path(__file__).resolve().parent
    workspace_root = package_root.parent
    parser = argparse.ArgumentParser(
        description="Collision-aware, self-contained CLIP-H evaluator"
    )
    parser.add_argument("--name", default="Aerial-CLIP-H")
    parser.add_argument("--maxActions", type=int, default=150)
    parser.add_argument("--xOy_step_size", type=float, default=2.0)
    parser.add_argument("--z_step_size", type=float, default=1.0)
    parser.add_argument("--rotateAngle", type=float, default=15.0)
    parser.add_argument("--batchSize", type=int, default=1)
    parser.add_argument("--simulator_tool_port", type=int, default=30011)
    parser.add_argument(
        "--dataset_path",
        default=str(workspace_root / "DATASETS" / "valset" / "DownTown.json"),
    )
    parser.add_argument("--is_fixed", type=str2bool, default=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--eval_save_path",
        default=str(package_root / "logs"),
    )
    parser.add_argument("--image_save_path", default=None)
    parser.add_argument("--clip_model_path", default="openai/clip-vit-base-patch16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--allow_cpu",
        type=str2bool,
        default=False,
        help="allow slow CPU fallback when a requested CUDA device is unavailable",
    )
    parser.add_argument(
        "--resume",
        type=str2bool,
        default=False,
        help="skip tasks that already have a complete result.json in eval_save_path",
    )

    safety = parser.add_argument_group("depth safety")
    safety.add_argument("--safety_margin", type=float, default=0.5)
    safety.add_argument("--collision_percentile", type=float, default=5.0)

    detection = parser.add_argument_group("target detection")
    detection.add_argument("--approach_score", type=float, default=0.225)
    detection.add_argument("--stop_score", type=float, default=0.255)
    detection.add_argument("--min_specificity", type=float, default=-0.005)
    detection.add_argument("--stop_specificity", type=float, default=0.005)
    detection.add_argument("--stop_category_score", type=float, default=0.28)
    detection.add_argument("--stop_detail_score", type=float, default=0.245)
    detection.add_argument("--direction_margin", type=float, default=0.006)
    detection.add_argument("--quick_stop_score", type=float, default=0.285)
    detection.add_argument("--quick_stop_specificity", type=float, default=0.045)
    detection.add_argument("--quick_stop_min_depth", type=float, default=3.0)
    detection.add_argument("--quick_stop_depth", type=float, default=10.0)
    detection.add_argument("--quick_stop_enabled", type=str2bool, default=False)
    detection.add_argument("--min_stop_depth", type=float, default=3.0)
    detection.add_argument("--stop_depth", type=float, default=10.0)
    detection.add_argument("--spatial_stop_enabled", type=str2bool, default=False)
    detection.add_argument("--min_stop_depth_support", type=float, default=0.20)
    detection.add_argument("--stop_confirm_frames", type=int, default=1)
    detection.add_argument("--approach_confirm_frames", type=int, default=2)
    detection.add_argument("--lost_target_frames", type=int, default=3)
    detection.add_argument("--clip_image_batch_size", type=int, default=64)

    exploration = parser.add_argument_group("exploration")
    exploration.add_argument("--scan_turns", type=int, default=0)
    exploration.add_argument("--periodic_scan_turns", type=int, default=1)
    exploration.add_argument("--search_moves_per_scan", type=int, default=12)
    exploration.add_argument(
        "--search_translation_budget",
        type=int,
        default=32,
        help="maximum translations before policy stop; 0 disables the limit",
    )
    exploration.add_argument(
        "--recovery_rotation_limit",
        type=int,
        default=6,
        help="consecutive rotations before policy stop; 0 disables the limit",
    )
    exploration.add_argument("--visit_cell_size", type=float, default=5.0)
    exploration.add_argument("--search_radius", type=float, default=50.0)
    world = parser.add_argument_group("voxel JEPA world model")
    world.add_argument("--world_model_enabled", type=str2bool, default=True)
    world.add_argument("--voxel_xy_cells", type=int, default=24)
    world.add_argument("--voxel_z_cells", type=int, default=8)
    world.add_argument("--voxel_size", type=float, default=1.0)
    world.add_argument("--voxel_max_depth", type=float, default=20.0)
    world.add_argument("--voxel_samples_per_axis", type=int, default=12)
    world.add_argument("--jepa_latent_dim", type=int, default=48)
    world.add_argument("--jepa_hidden_dim", type=int, default=96)
    world.add_argument("--jepa_learning_rate", type=float, default=0.0003)
    world.add_argument("--jepa_regularization", type=float, default=0.05)
    world.add_argument("--jepa_online_training", type=str2bool, default=True)
    world.add_argument("--jepa_replay_capacity", type=int, default=512)
    world.add_argument("--jepa_batch_size", type=int, default=8)
    world.add_argument("--jepa_planning_horizon", type=int, default=6)
    world.add_argument("--jepa_beam_width", type=int, default=8)
    world.add_argument("--jepa_trust_transitions", type=int, default=32)
    world.add_argument("--jepa_collision_weight", type=float, default=6.0)
    world.add_argument("--jepa_revisit_weight", type=float, default=0.8)
    world.add_argument("--jepa_semantic_weight", type=float, default=3.0)
    world.add_argument("--jepa_uncertainty_weight", type=float, default=0.05)
    world.add_argument("--jepa_latent_novelty_weight", type=float, default=0.1)
    world.add_argument("--jepa_override_margin", type=float, default=2.0)
    world.add_argument(
        "--jepa_override_min_risk_reduction", type=float, default=0.0
    )
    world.add_argument("--jepa_hard_collision_risk", type=float, default=0.20)
    world.add_argument("--jepa_hard_collision_enabled", type=str2bool, default=False)
    world.add_argument("--jepa_checkpoint_path", default="")
    world.add_argument("--jepa_semantic_memory_weight", type=float, default=2.0)
    world.add_argument("--jepa_goal_latent_weight", type=float, default=0.0)
    world.add_argument("--jepa_goal_min_score", type=float, default=0.23)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batchSize < 1 or args.batchSize > 16:
        parser.error("--batchSize must be between 1 and 16")
    if args.xOy_step_size <= 0 or args.z_step_size <= 0 or args.rotateAngle <= 0:
        parser.error("movement and rotation step sizes must be positive")
    if args.safety_margin < 0:
        parser.error("--safety_margin must be non-negative")
    if (
        args.stop_confirm_frames < 1
        or args.approach_confirm_frames < 1
        or args.lost_target_frames < 1
    ):
        parser.error("temporal confirmation counts must be positive")
    if args.min_stop_depth < 0 or args.min_stop_depth >= args.stop_depth:
        parser.error("--min_stop_depth must be non-negative and below --stop_depth")
    if not 0.0 <= args.min_stop_depth_support <= 1.0:
        parser.error("--min_stop_depth_support must be between 0 and 1")
    if (
        args.quick_stop_min_depth < 0
        or args.quick_stop_min_depth >= args.quick_stop_depth
    ):
        parser.error("quick-stop depth range is invalid")
    if args.clip_image_batch_size < 1:
        parser.error("--clip_image_batch_size must be positive")
    if (
        args.scan_turns < 0
        or args.periodic_scan_turns < 0
        or args.search_moves_per_scan < 1
        or args.search_translation_budget < 0
        or args.recovery_rotation_limit < 0
    ):
        parser.error(
            "scan turns and strategy limits must be non-negative, and search "
            "moves must be positive"
        )
    if (
        args.voxel_xy_cells < 8
        or args.voxel_z_cells < 4
        or args.voxel_size <= 0
        or args.voxel_max_depth <= 0
        or args.voxel_samples_per_axis < 4
        or args.jepa_latent_dim < 8
        or args.jepa_hidden_dim < 8
        or args.jepa_learning_rate <= 0
        or args.jepa_regularization < 0
        or args.jepa_replay_capacity < 1
        or args.jepa_batch_size < 2
        or args.jepa_batch_size > args.jepa_replay_capacity
        or args.jepa_planning_horizon < 1
        or args.jepa_beam_width < 1
        or args.jepa_trust_transitions < 1
        or args.jepa_override_margin < 0
        or not 0.0 <= args.jepa_hard_collision_risk <= 1.0
    ):
        parser.error("invalid voxel JEPA configuration")
    if args.image_save_path in ("", "None", "none", "null"):
        args.image_save_path = None
    args.dataset_path = os.path.abspath(args.dataset_path)
    args.eval_save_path = os.path.abspath(args.eval_save_path)
    if args.image_save_path:
        args.image_save_path = os.path.abspath(args.image_save_path)
    return args


def policy_config_from_args(args):
    return PolicyConfig(
        horizontal_step=args.xOy_step_size,
        vertical_step=args.z_step_size,
        rotate_angle=args.rotateAngle,
        is_fixed=args.is_fixed,
        safety_margin=args.safety_margin,
        collision_percentile=args.collision_percentile,
        approach_score=args.approach_score,
        stop_score=args.stop_score,
        min_specificity=args.min_specificity,
        stop_specificity=args.stop_specificity,
        stop_category_score=args.stop_category_score,
        stop_detail_score=args.stop_detail_score,
        direction_margin=args.direction_margin,
        quick_stop_score=args.quick_stop_score,
        quick_stop_specificity=args.quick_stop_specificity,
        quick_stop_min_depth=args.quick_stop_min_depth,
        quick_stop_depth=args.quick_stop_depth,
        quick_stop_enabled=args.quick_stop_enabled,
        min_stop_depth=args.min_stop_depth,
        stop_depth=args.stop_depth,
        spatial_stop_enabled=args.spatial_stop_enabled,
        min_stop_depth_support=args.min_stop_depth_support,
        stop_confirm_frames=args.stop_confirm_frames,
        approach_confirm_frames=args.approach_confirm_frames,
        lost_target_frames=args.lost_target_frames,
        scan_turns=args.scan_turns,
        periodic_scan_turns=args.periodic_scan_turns,
        search_moves_per_scan=args.search_moves_per_scan,
        search_translation_budget=args.search_translation_budget,
        recovery_rotation_limit=args.recovery_rotation_limit,
        visit_cell_size=args.visit_cell_size,
        search_radius=args.search_radius,
        image_batch_size=args.clip_image_batch_size,
        world_model_enabled=args.world_model_enabled,
        voxel_xy_cells=args.voxel_xy_cells,
        voxel_z_cells=args.voxel_z_cells,
        voxel_size=args.voxel_size,
        voxel_max_depth=args.voxel_max_depth,
        voxel_samples_per_axis=args.voxel_samples_per_axis,
        jepa_latent_dim=args.jepa_latent_dim,
        jepa_hidden_dim=args.jepa_hidden_dim,
        jepa_learning_rate=args.jepa_learning_rate,
        jepa_regularization=args.jepa_regularization,
        jepa_online_training=args.jepa_online_training,
        jepa_replay_capacity=args.jepa_replay_capacity,
        jepa_batch_size=args.jepa_batch_size,
        jepa_planning_horizon=args.jepa_planning_horizon,
        jepa_beam_width=args.jepa_beam_width,
        jepa_trust_transitions=args.jepa_trust_transitions,
        jepa_collision_weight=args.jepa_collision_weight,
        jepa_revisit_weight=args.jepa_revisit_weight,
        jepa_semantic_weight=args.jepa_semantic_weight,
        jepa_uncertainty_weight=args.jepa_uncertainty_weight,
        jepa_latent_novelty_weight=args.jepa_latent_novelty_weight,
        jepa_override_margin=args.jepa_override_margin,
        jepa_override_min_risk_reduction=args.jepa_override_min_risk_reduction,
        jepa_hard_collision_risk=args.jepa_hard_collision_risk,
        jepa_hard_collision_enabled=args.jepa_hard_collision_enabled,
        jepa_checkpoint_path=args.jepa_checkpoint_path,
        jepa_semantic_memory_weight=args.jepa_semantic_memory_weight,
        jepa_goal_latent_weight=args.jepa_goal_latent_weight,
        jepa_goal_min_score=args.jepa_goal_min_score,
    )
