"""Send (optionally perturbed) tiles to the serving API — the live demo of the loop.

The drift experiment (drift_experiment.py) runs offline and in process, because
20k HTTP round trips would add hours and change no number. This script is the
complement: it exercises the REAL path — HTTP, the shared preprocessing inside
the API, the prediction log, the Prometheus counters — so that the end-to-end
chain can be demonstrated rather than asserted:

    drift_traffic.py --kind cloud --intensity 0.6
        -> API /predict logs N rows tagged source=sim:cloud:0.6
    drift_report.py --source sim:cloud:0.6
        -> Evidently report against the frozen training reference
    drift_monitor.py --source sim:cloud:0.6
        -> the CT decision (persistence, cooldown, dispatch)

Every request carries an X-TerraOps-Source header naming the perturbation and its
intensity. That tag is what keeps synthetic traffic out of a report about real
traffic — mixing them would corrupt the very reference the CT loop reacts to,
which is exactly the kind of self-inflicted data quality problem monitoring is
supposed to catch, not cause.
"""

import argparse
import io
import time
import urllib.request
from typing import List

from torchvision import datasets

import drift_sim
from utils import load_params


def build_payload(image, name: str) -> tuple:
    """PIL image -> multipart body. PNG, so the perturbation is not re-quantized
    by JPEG on the way out (that would add an uncontrolled second perturbation)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return name, buffer.getvalue()


def post_image(url: str, payload: tuple, source: str, timeout: float = 30.0) -> dict:
    """One multipart POST, hand-rolled to keep this script dependency-free."""
    import json
    import uuid

    name, blob = payload
    boundary = uuid.uuid4().hex
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{name}"\r\n'.encode(),
        b"Content-Type: image/png\r\n\r\n",
        blob,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "X-TerraOps-Source": source})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--kind", default="none",
                        choices=["none", *drift_sim.perturbation_names()])
    parser.add_argument("--intensity", type=float, default=0.0)
    parser.add_argument("--count", type=int, default=250)
    parser.add_argument("--offset", type=int, default=0,
                        help="skip the first N images (send different tiles per run)")
    parser.add_argument("--source", default=None,
                        help="override the X-TerraOps-Source tag")
    args = parser.parse_args()

    params = load_params()
    source = args.source or (
        "sim:baseline" if args.kind == "none"
        else f"sim:{args.kind}:{args.intensity:g}")

    raw = datasets.EuroSAT(root=params["prepare"]["data_dir"],
                           download=False, transform=None)
    # A stride over the dataset rather than the first N images: consecutive
    # EuroSAT indices are the same class, and a single-class window would trip
    # the class-collapse detector for a reason that has nothing to do with drift.
    step = max(1, len(raw) // (args.count + args.offset))
    picked = list(range(0, len(raw), step))[args.offset:args.offset + args.count]

    sent, failed = 0, 0
    latencies: List[float] = []
    started = time.perf_counter()

    for i in picked:
        image = raw[i][0]
        if args.kind != "none" and args.intensity > 0:
            image = drift_sim.apply_named(image, args.kind, args.intensity,
                                          params, index=i)
        try:
            call_started = time.perf_counter()
            post_image(f"{args.api}/predict", build_payload(image, f"tile_{i}.png"),
                       source)
            latencies.append((time.perf_counter() - call_started) * 1000)
            sent += 1
        except Exception as exc:
            failed += 1
            if failed <= 3:
                print(f"  request failed ({type(exc).__name__}: {exc})")

    elapsed = time.perf_counter() - started
    latencies.sort()
    p95 = latencies[int(0.95 * len(latencies))] if latencies else float("nan")
    print(f"sent={sent} failed={failed} source={source} "
          f"in {elapsed:.1f}s (client-side p95 {p95:.0f} ms)")
    if sent < params["monitor"]["min_current_rows"]:
        # Warn rather than let the next step return a confusing non-verdict.
        print(f"note: {sent} rows is below monitor.min_current_rows="
              f"{params['monitor']['min_current_rows']}, so a drift report over "
              f"this source alone will be INCONCLUSIVE, not 'no drift'.")


if __name__ == "__main__":
    main()
