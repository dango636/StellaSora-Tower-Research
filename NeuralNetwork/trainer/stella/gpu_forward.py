"""Opt-in, fixed-shape FP32 graph replay for rollout inference only.

The final short chunk always follows eager inference. No padding, AMP, TF32,
compilation or arithmetic fusion is introduced here. This is experimental until
measured on the deployment GPU; STELLA_GPU_CUDA_GRAPH defaults to disabled.
"""
from __future__ import annotations

import torch


class FixedGraphForward:
    def __init__(self, rows=16384):
        self.rows = int(rows)
        self.model = None
        self.graph = None
        self.inputs = None
        self.output = None
        self.event = None
        self._capture_stream = None
        self._failed_replay_stream = None
        self.verified = False
        self.disabled_reason = None
        self.replays = 0
        self.captures = 0
        self.verifications = 0

    def set_model(self, model):
        if model is not self.model:
            self.close()
            self.model = model
        # In-place teacher copies retain addresses but still require a new exact
        # eager/replay comparison before accepting this teacher's graph output.
        self.verified = False

    def close(self):
        # A failed device synchronization must remain retryable: callers may
        # explicitly close after the original exception. Never fall back to an
        # older completed event while a newer stream is still unaccounted for.
        if self._failed_replay_stream is not None:
            self._failed_replay_stream.synchronize()
            self._failed_replay_stream = None
        if self._capture_stream is not None:
            self._capture_stream.synchronize()
            self._capture_stream = None
        if self.event is not None:
            self.event.synchronize()
        self.event = None
        self._capture_stream = None
        self.graph = None
        self.inputs = None
        self.output = None
        self.model = None
        self.verified = False

    def _capture(self, features, scale):
        self.inputs = features.clone()
        current = torch.cuda.current_stream(features.device)
        side = torch.cuda.Stream(device=features.device)
        side.wait_stream(current)
        self._capture_stream = side
        self.event = torch.cuda.Event()
        try:
            with torch.cuda.stream(side):
                # Warm up cuBLAS handles and workspaces outside capture, on a
                # side stream as required by PyTorch's CUDA graph workflow.
                for _ in range(3):
                    warmup = self.model(self.inputs, amp=False) * scale
            current.wait_stream(side)
            del warmup
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=side):
                self.output = self.model(self.inputs, amp=False) * scale
        finally:
            # Also protect the static input if warmup/capture raises. close()
            # waits for this event before releasing its cross-stream storage.
            self.event.record(side)
            current.wait_event(self.event)
        self._capture_stream = None
        self.captures += 1

    @torch.inference_mode()
    def __call__(self, model, features, scale):
        if (len(features) != self.rows or features.device.type != "cuda"
                or features.dtype != torch.float32 or self.disabled_reason is not None):
            return model(features, amp=False) * scale
        if model is not self.model:
            self.set_model(model)
        eager = None
        replay_stream = None
        replay_submitted = False
        try:
            if not self.verified:
                eager = model(features, amp=False) * scale
            if self.graph is None:
                self._capture(features, scale)
            current = torch.cuda.current_stream(features.device)
            if self.replays:
                current.wait_event(self.event)
            # From this point, an exception can occur after work was queued
            # on a stream different from the static input's allocation stream.
            # Mark before copy/replay: either operation may enqueue then raise.
            replay_stream = current
            replay_submitted = True
            self.inputs.copy_(features)
            self.graph.replay()
            # Return independent storage: callers may retain logits while this
            # instance replays the next chunk or another root batch.
            result = self.output.clone()
            self.event.record(current)
            self.replays += 1
            if not self.verified:
                self.verifications += 1
                if not torch.equal(eager.view(torch.int32), result.view(torch.int32)):
                    self.disabled_reason = "first_teacher_chunk_bitwise_mismatch"
                    self.close()
                    return eager
                self.verified = True
            return result
        except RuntimeError as error:
            # Unsupported capture and insufficient private-pool memory are
            # recoverable; actual eager failures still propagate to the caller.
            self.disabled_reason = f"{type(error).__name__}: {error}"
            if self._capture_stream is not None:
                # Also cover failure while recording the capture-side event.
                # Failed stream synchronization propagates without freeing buffers.
                self._capture_stream.synchronize()
                self._capture_stream = None
            if replay_submitted:
                # output.clone() may OOM before event.record(current), leaving
                # self.event referring to the previous replay. Wait THIS stream
                # before dropping static storage. No success-path sync is added.
                # If synchronization itself fails, propagate the device error;
                # storage stays referenced rather than pretending safe fallback.
                self._failed_replay_stream = replay_stream
                replay_stream.synchronize()
                self._failed_replay_stream = None
            self.close()
            return eager if eager is not None else model(features, amp=False) * scale

    def snapshot(self):
        return dict(requested=True, rows=self.rows, captures=self.captures,
                    replays=self.replays, verifications=self.verifications,
                    verified=self.verified, disabled_reason=self.disabled_reason)
