import threading
import time


class Spinner:
    def __init__(self, label: str) -> None:
        self._label = label
        self._stop_event = threading.Event()
        self._start = time.monotonic()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        # Length of the last line actually printed, so a shorter next line
        # (e.g. label switching from a long "captioning tables (9/12)" to a
        # short "done in Xs") pads with exactly enough spaces to erase the
        # leftover tail instead of a fixed guess that can fall short.
        self._last_len = 0

    def _print(self, text: str, end: str = "") -> None:
        pad = max(0, self._last_len - len(text))
        print(f"\r{text}" + " " * pad, end=end, flush=True)
        self._last_len = len(text)

    def _spin(self) -> None:
        while not self._stop_event.is_set():
            self._print(f"{self._label}... {self.elapsed:.0f}s")
            self._stop_event.wait(1.0)

    def start(self) -> None:
        self._thread.start()

    def update(self, label: str) -> None:
        self._label = label

    def stop(self, final_message: str) -> None:
        self._stop_event.set()
        self._thread.join()
        self._print(final_message, end="\n")

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start
