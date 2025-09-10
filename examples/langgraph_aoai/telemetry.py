import time
from contextlib import contextmanager

@contextmanager
def timed(span: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000
        print(f"[telemetry] {span} took {dt:.1f} ms")

# Example:
# with timed("llm.call"): out = call_llm(prompt)