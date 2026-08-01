import logging
import warnings
from pathlib import Path

import transformers.utils.logging as transformers_logging

WARNING_LOG_FILENAME = "pdf-ingestion.log"


def configure_warnings(output_dir: Path, save_log: bool) -> None:
    # "Loading weights" (HF model loading inside docling_ibm_models's layout
    # model, via transformers.AutoModelForObjectDetection.from_pretrained) is
    # a transient progress bar, not a diagnostic message - there's no
    # meaningful text to redirect into a log file, so it's always disabled
    # outright rather than routed anywhere.
    transformers_logging.disable_progress_bar()

    if not save_log:
        # Discard entirely: no console output, no file. ERROR+ still surfaces
        # in case something more serious than the known noisy warnings occurs.
        warnings.simplefilter("ignore")
        for logger_name in ("docling", "RapidOCR"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        # rapidocr's own logger setup (rapidocr/utils/log.py) runs lazily on
        # first OCR use, i.e. after this function returns, and unconditionally
        # resets this logger's level back to INFO - but only attaches its own
        # console handler `if not self.logger.handlers`. Pre-attaching a
        # no-op handler here defeats that guard, so its console spam never
        # appears no matter when that lazy init actually runs.
        logging.getLogger("RapidOCR").addHandler(logging.NullHandler())
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(output_dir / WARNING_LOG_FILENAME, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    # warnings.warn() (bbox clamping) does not use `logging` by default.
    # captureWarnings() routes it through the "py.warnings" logger instead;
    # simplefilter("always") disables Python's default per-location dedup,
    # so every occurrence reaches the file, not just the first at each line.
    warnings.simplefilter("always")
    logging.captureWarnings(True)

    for logger_name in ("docling", "py.warnings"):
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        logger.propagate = False

    # RapidOCR's own logger does not propagate and ships with its own
    # console StreamHandler attached at import time - must be swapped out
    # explicitly, attaching to "docling" alone has no effect on it.
    rapidocr_logger = logging.getLogger("RapidOCR")
    for existing in list(rapidocr_logger.handlers):
        rapidocr_logger.removeHandler(existing)
    rapidocr_logger.addHandler(handler)
    # INFO here (not WARNING like the other two loggers above): this also
    # captures RapidOCR's own engine-init lines ("Using engine_name: torch",
    # "Using CPU device", model file paths), which are only ever emitted at
    # INFO level and would otherwise be dropped even with the handler above.
    rapidocr_logger.setLevel(logging.INFO)
