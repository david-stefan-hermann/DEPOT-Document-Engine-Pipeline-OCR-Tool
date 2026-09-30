from __future__ import annotations

import logging
import signal
import sys

from depot.config import Config
from depot.pipeline import Pipeline
from depot.watcher import ScanWatcher
from depot.workqueue import WorkQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
)
log = logging.getLogger("depot")


def main() -> None:
    config = Config.from_env()
    pipeline = Pipeline(config)

    try:
        log.info("Checking WebDAV connectivity to %s ...", config.nextcloud_webdav_url)
        pipeline.webdav.check_connection()
        log.info("WebDAV connection OK.")

        work_queue = WorkQueue()
        watcher = ScanWatcher(
            local_path=config.scan_eingang_local_path,
            supported_extensions=config.supported_extensions,
            log_file_prefix=config.log_file_prefix,
            config_file_name=config.config_file_name,
            out_queue=work_queue,
        )

        log.info("Running startup sweep of %s ...", config.scan_eingang_local_path)
        watcher.startup_sweep()

        workers = pipeline.run_workers(work_queue)
        watcher.start()
        watcher.start_periodic_sweep(config.sweep_interval_seconds)
    except Exception:
        log.error("Startup failed, exiting.", exc_info=True)
        pipeline.close()
        sys.exit(1)

    def _handle_signal(signum, frame):
        log.info("Received signal %s, shutting down ...", signum)
        watcher.stop()
        pipeline.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    for w in workers:
        w.join()


if __name__ == "__main__":
    main()
