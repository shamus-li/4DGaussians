def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)

    return args


def load_config(config_path):
    """Load a config file compatible with both mmcv and mmengine."""

    ConfigCls = None

    try:
        import mmcv  # type: ignore
    except ImportError:
        mmcv = None  # type: ignore

    if mmcv is not None:
        ConfigCls = getattr(mmcv, "Config", None)
        if ConfigCls is None:
            try:
                from mmcv.utils import Config as ConfigCls  # type: ignore
            except ImportError:
                ConfigCls = None

    if ConfigCls is None:
        try:
            from mmengine.config import Config as ConfigCls  # type: ignore
        except ImportError as exc:  # pragma: no cover - defensive
            raise ImportError(
                "Unable to import Config from mmcv or mmengine. Please install a "
                "compatible version of either package."
            ) from exc

    return ConfigCls.fromfile(config_path)
