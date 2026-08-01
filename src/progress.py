import threading
import time


class Spinner:
    def __init__(self, label: str) -> None:
        self._label = label
        self._stop_event = threading.Event()
        self._start = time.monotonic()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self) -> None:
        while not self._stop_event.is_set():
            print(f"\r{self._label}... {self.elapsed:.0f}s", end="", flush=True)
            self._stop_event.wait(1.0)

    def start(self) -> None:
        self._thread.start()

    def stop(self, final_message: str) -> None:
        self._stop_event.set()
        self._thread.join()
        # Trailing spaces overwrite any leftover characters from the last
        # spinner frame if it was longer than the final message.
        print(f"\r{final_message}" + " " * 10)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start
