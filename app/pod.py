"""Runpod pod lifecycle: reuse if possible, create if not, always in region.

Why this is not a fixed pod id: Runpod reassigns hardware. A pod stopped
overnight can come back with "Your Pod's GPUs are no longer available",
which happened mid-project. A pinned id is a single point of failure.

It also matters WHERE the pod lands. The same code measured 276 ms
round-trip on a European pod and 132 ms on a North American one -- a 144 ms
difference that no amount of buffer tuning recovers. So region is a
constraint, not a preference.

Strategy, cheapest first:
  1. Reuse a running pod from this image.
  2. Start an existing stopped one -- the image layers are already cached on
     that host, so it is up in ~30 s rather than several minutes.
  3. Create a new one, pinned to the configured datacenters.

The proxy hostname derives from the pod id, so the client URL is discovered
rather than configured.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request

from app.region import describe, home_region, ranked_datacenters

log = logging.getLogger("transend.pod")

# Only cards at or above the A40's compute, at comparable cost.
#
# This is a correctness constraint, not a preference. Inference measured
# ~85 ms against a 160 ms block on an A40 -- roughly 2x headroom. A card
# half as fast lands at ~170 ms, PAST the block duration, and once inference
# is slower than real time the backlog grows without bound: the buffer
# deepens, drift correction starts discarding speech, and the session
# degrades until it is unusable. That is the 2754 ms cold-start behaviour,
# except permanent.
#
# So the list is restricted to GA102 silicon or newer:
#   A40        GA102, 48 GB, ~$0.49/hr  -- the measured baseline
#   RTX A6000  GA102, 48 GB, ~$0.49-0.76/hr
#   RTX 3090   GA102, 24 GB, ~$0.22-0.43/hr  -- same silicon, cheaper
#   RTX 4090   AD102, 24 GB, ~$0.34-0.74/hr  -- faster than the baseline
#
# Deliberately excluded despite being valid enum values: L4 (72 W, heavily
# cut down), RTX 4000 Ada and A5000 (materially slower), L40S and H100
# (fast but several times the price for headroom we do not need).
#
# VRAM is irrelevant here -- the model is 142 MB.
# Locked to the A40 for now: it is the only card the pipeline has actually
# been measured on (~85 ms inference, ~500-630 ms end to end, zero data
# loss). The others above are plausible on paper but unverified, and the
# failure mode of a too-slow card is not "slightly worse" -- it is audio
# falling behind without bound.
#
# Widen via RUNPOD_GPUS once another card has been measured, e.g.
#   RUNPOD_GPUS=NVIDIA A40,NVIDIA RTX A6000
DEFAULT_GPUS = ["NVIDIA A40"]


class PodController:
    API_ROOT = "https://rest.runpod.io/v1"

    def __init__(self, api_key=None, pod_id=None, image=None, port=8000,
                 datacenters=None, gpu_types=None, name="transend",
                 region=None):
        self.api_key = api_key or os.getenv("RUNPOD_API_KEY") or ""
        # A configured id is a hint, not a requirement.
        self.pod_id = pod_id or os.getenv("RUNPOD_POD_ID") or ""
        self.image = image or os.getenv(
            "RUNPOD_IMAGE", "aboro049/seedvc-server:latest")
        self.port = port
        self.name = name
        self.region = region
        # Distance dominates latency, so the datacenter list follows the
        # user's region rather than a fixed default.
        self.datacenters = datacenters or ranked_datacenters()
        self.gpu_types = gpu_types or [
            g.strip() for g in os.getenv(
                "RUNPOD_GPUS", ",".join(DEFAULT_GPUS)).split(",") if g.strip()]
        self.created_here = False
        self.last_error = ""
        # What we ended up with, for the UI and the history log. The API
        # does not report datacenter on /v1/pods, so this may stay blank
        # when reusing an existing pod.
        self.actual_datacenter = ""
        self.actual_gpu = ""
        self.actual_cost = 0.0
        # Retry locally before considering another continent.
        self.local_attempts = int(os.getenv("RUNPOD_LOCAL_ATTEMPTS", "3"))
        self.local_retry_s = float(os.getenv("RUNPOD_LOCAL_RETRY_S", "20"))
        self.allow_far = os.getenv("RUNPOD_ALLOW_FAR", "1") not in ("0", "false")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def url_for(self, pod_id: str) -> str:
        return f"wss://{pod_id}-{self.port}.proxy.runpod.net"

    # --- transport --------------------------------------------------------

    def _req(self, path, method="GET", body=None, timeout=60):
        data = json.dumps(body).encode() if body is not None else None
        # Runpod sits behind Cloudflare, which 403s the default
        # "Python-urllib/3.x" user agent with error code 1010. curl works,
        # urllib does not, purely because of this header.
        req = urllib.request.Request(
            f"{self.API_ROOT}{path}", method=method, data=data,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json",
                     "Accept": "application/json",
                     "User-Agent": "transend/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode()
                return True, (json.loads(raw) if raw.strip() else {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            self.last_error = f"HTTP {e.code}: {detail}"
            return False, self.last_error
        except Exception as e:
            self.last_error = str(e)
            return False, self.last_error

    # --- queries ----------------------------------------------------------

    def list_pods(self) -> list[dict]:
        ok, data = self._req("/pods")
        if not ok:
            # Returning [] here would look like "no pods exist" and send us
            # straight to creating one, hiding the real error.
            log.warning("list_pods failed: %s", data)
            return []
        if isinstance(data, dict):
            data = data.get("pods") or data.get("data") or []
        return data if isinstance(data, list) else []

    def get_pod(self, pod_id: str) -> dict | None:
        ok, data = self._req(f"/pods/{pod_id}")
        if ok and isinstance(data, dict):
            self._remember(data)
            return data
        return None

    def _remember(self, pod: dict) -> None:
        """Record datacenter / GPU / cost wherever this API version puts them."""
        dc = self._datacenter(pod)
        if dc:
            self.actual_datacenter = dc
        for key in ("gpuTypeId", "gpuType", "gpuDisplayName"):
            v = pod.get(key)
            if isinstance(v, str) and v:
                self.actual_gpu = v
                break
            if isinstance(v, dict) and v.get("displayName"):
                self.actual_gpu = v["displayName"]
                break
        mach = pod.get("machine")
        if isinstance(mach, dict) and not self.actual_gpu:
            for key in ("gpuTypeId", "gpuDisplayName"):
                if isinstance(mach.get(key), str) and mach[key]:
                    self.actual_gpu = mach[key]
                    break
        if isinstance(pod.get("costPerHr"), (int, float)):
            self.actual_cost = float(pod["costPerHr"])

    @staticmethod
    def _datacenter(pod: dict) -> str:
        """Datacenter id, however this API version spells it."""
        for key in ("dataCenterId", "datacenterId", "dataCenter", "locationId"):
            v = pod.get(key)
            if isinstance(v, str) and v:
                return v
            if isinstance(v, dict) and v.get("id"):
                return v["id"]
        mach = pod.get("machine")
        if isinstance(mach, dict):
            for key in ("dataCenterId", "datacenterId", "location"):
                if isinstance(mach.get(key), str) and mach[key]:
                    return mach[key]
        return ""

    @staticmethod
    def _status(pod: dict) -> str:
        return str(pod.get("desiredStatus") or pod.get("status") or "").upper()

    def _is_ours(self, pod: dict) -> bool:
        img = str(pod.get("imageName") or pod.get("image") or "")
        return self.image.split(":")[0] in img

    # --- actions ----------------------------------------------------------

    def _probe_ms(self, pod_id: str, timeout=6.0) -> float | None:
        """TCP handshake time to a pod's proxy, as a stand-in for distance."""
        import socket
        import ssl
        host = f"{pod_id}-{self.port}.proxy.runpod.net"
        try:
            t0 = time.perf_counter()
            with socket.create_connection((host, 443), timeout=timeout) as sock:
                ctx = ssl.create_default_context()
                with ctx.wrap_socket(sock, server_hostname=host):
                    return (time.perf_counter() - t0) * 1000.0
        except Exception:
            return None

    def start(self, pod_id: str):
        return self._req(f"/pods/{pod_id}/start", method="POST", body={})

    def stop(self, pod_id: str | None = None):
        pid = pod_id or self.pod_id
        if not pid:
            return False, "no pod id"
        return self._req(f"/pods/{pid}/stop", method="POST", body={})

    def terminate(self, pod_id: str | None = None):
        pid = pod_id or self.pod_id
        if not pid:
            return False, "no pod id"
        return self._req(f"/pods/{pid}", method="DELETE")

    def create(self, datacenters=None):
        """Create a pod in the given datacenters (default: the full list).

        Both the datacenter and GPU lists are preferences, so Runpod falls
        back within our constraints rather than failing on one sold-out
        combination.
        """
        body = {
            "name": self.name,
            "imageName": self.image,
            "gpuTypeIds": self.gpu_types,
            "gpuCount": 1,
            "dataCenterIds": datacenters or self.datacenters,
            "containerDiskInGb": 30,
            "volumeInGb": 0,
            "ports": [f"{self.port}/http"],
            "cloudType": "SECURE",
        }
        ok, data = self._req("/pods", method="POST", body=body, timeout=120)
        if not ok:
            return None, data
        pid = (data.get("id") if isinstance(data, dict) else None)
        if not pid:
            return None, f"no pod id in response: {str(data)[:200]}"
        self.created_here = True
        return pid, "created"

    # --- the one the app calls -------------------------------------------

    def ensure_pod(self, progress=lambda m: None) -> tuple[str | None, str]:
        """Return a pod id that is RUNNING, reusing or creating as needed."""
        if not self.enabled:
            return None, "no RUNPOD_API_KEY configured"

        # 1. Already-running pod from this image, NEAREST first.
        #
        # Taking whichever the API happens to list first can hand back a
        # pod on another continent while a local one sits idle -- measured
        # 363 ms vs 132 ms round trip, which is the difference between a
        # POOR and a GOOD session.
        running = [p for p in self.list_pods()
                   if self._is_ours(p) and self._status(p) == "RUNNING"]
        if running:
            order = {dc: i for i, dc in enumerate(self.datacenters)}
            known = [p for p in running if self._datacenter(p) in order]
            if known:
                known.sort(key=lambda p: order[self._datacenter(p)])
                pod = known[0]
                pid = pod.get("id")
                progress(f"reusing pod {pid} in {self._datacenter(pod)}")
                self.pod_id = pid
                return pid, "reused"

            # /v1/pods returns no datacenter for a pod (machine is {}), so
            # location cannot be read from the API. Measure instead: a TCP
            # handshake to each pod's proxy is a direct proxy for distance,
            # and picking wrong means 363 ms instead of 132 ms.
            if len(running) == 1:
                pid = running[0].get("id")
                progress(f"reusing running pod {pid}")
                self.pod_id = pid
                return pid, "reused"

            progress(f"{len(running)} pods running -- measuring which is "
                     f"closest...")
            best, best_ms = None, None
            for pod in running:
                pid = pod.get("id")
                ms = self._probe_ms(pid)
                progress(f"  {pid}: "
                         + (f"{ms:.0f} ms" if ms is not None else "unreachable"))
                if ms is not None and (best_ms is None or ms < best_ms):
                    best, best_ms = pid, ms
            if best:
                progress(f"reusing pod {best} ({best_ms:.0f} ms)")
                self.pod_id = best
                return best, "reused"

        # 2. A stopped one -- its image layers are cached on that host.
        candidates = [p for p in self.list_pods()
                      if self._is_ours(p) and self._status(p) in
                      ("EXITED", "STOPPED", "TERMINATED")]
        if self.pod_id:
            candidates.sort(key=lambda p: p.get("id") != self.pod_id)
        for pod in candidates:
            pid = pod.get("id")
            progress(f"starting existing pod {pid}...")
            ok, _ = self.start(pid)
            if ok:
                self.pod_id = pid
                return pid, "started"
            # Usually "GPUs no longer available" -- fall through and create.
            progress(f"could not start {pid}, creating a new pod...")

        # 3. Fresh pod -- HOME REGION ONLY first.
        #
        # Falling straight through to another continent is worse than
        # waiting: a European pod measured 363 ms round trip against 132 ms
        # nearby, which preflight then correctly grades POOR. Better to
        # retry locally for a while, and only offer the distant option as an
        # explicit choice.
        near = home_region()
        for attempt in range(1, self.local_attempts + 1):
            progress(f"creating a pod nearby ({near[0]} +{len(near) - 1} "
                     f"more, attempt {attempt}/{self.local_attempts})...")
            pid, detail = self.create(near)
            if pid:
                self.pod_id = pid
                return pid, f"created in {near[0]} region"
            if attempt < self.local_attempts:
                time.sleep(self.local_retry_s)

        if not self.allow_far:
            return None, (f"no GPUs free in your region right now "
                          f"({len(near)} datacenters, {len(self.gpu_types)} "
                          f"GPU types tried). Try again in a few minutes.")

        far = [d for d in self.datacenters if d not in near]
        progress("nothing free nearby -- trying further afield "
                 "(expect higher latency)...")
        pid, detail = self.create(far)
        if not pid:
            return None, f"could not create a pod anywhere: {detail}"
        self.pod_id = pid
        return pid, "created outside your region -- latency will be higher"

    def wait_running(self, pod_id: str, timeout_s=420,
                     progress=lambda m: None) -> bool:
        """Poll until the pod reports RUNNING.

        A new pod on a fresh host pulls 12-15 GB first, so this is minutes,
        not seconds. A restarted pod has the layers cached and is quick.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            pod = self.get_pod(pod_id)
            if pod and self._status(pod) == "RUNNING":
                self._remember(pod)
                return True
            left = int(deadline - time.time())
            waited = int(timeout_s - left)
            status = self._status(pod) if pod else "?"
            log.info("pod %s: %s (%ds)", pod_id, status or "pending", waited)
            progress(f"pod {status.lower() or 'pending'}...  {waited}s elapsed")
            time.sleep(5)
        return False

    def shutdown(self) -> tuple[bool, str]:
        """Stop pods we reused; terminate ones we created.

        A created pod has no cached value to preserve and would keep billing
        for its disk, so it goes away entirely.
        """
        if not self.pod_id:
            return True, "nothing to stop"
        if self.created_here:
            ok, d = self.terminate(self.pod_id)
            return ok, ("terminated" if ok else str(d))
        ok, d = self.stop(self.pod_id)
        return ok, ("stopped" if ok else str(d))
