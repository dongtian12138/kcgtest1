"""Persistent compiled MPFR point-to-plane evaluator.

The exact root isolator asks the same scalar question many times: for one
binary64 path phase, on which side of an object triangle plane is one fixed
finger witness?  Calling thousands of individual ``mpmath`` operations from
Python dominates the wall time.  This optional backend keeps one complete
URDF chain in MPFR storage and answers that whole question with one C call.

It does not replace interval derivatives, interval-wide motion bounds, final
triangle contact predicates, or collision acceptance.  Those remain on the
existing Python/mpmath proof path.  Loading is fail-closed: architecture,
compiler, ABI and a directed-rounding numerical smoke test must all pass.
"""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import threading
import weakref

import numpy as np


MPFR_POINT_PLANE_ABI_VERSION = 4
MPFR_POINT_PLANE_METHOD_ID = (
    "REBINDABLE_PERSISTENT_MPFR_DIRECTED_POINT_PLANE_URDF_CHAIN_V2"
)
_SOURCE_NAME = "_mpfr_point_plane_kernel.c"
_LIBRARY_LOCK = threading.Lock()
_LIBRARY: ctypes.CDLL | None = None
_LIBRARY_FAILURE: str | None = None

_INT_POINTER = ctypes.POINTER(ctypes.c_int)
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


class MpfrPointPlaneBackendUnavailable(RuntimeError):
    """The optional compiled backend cannot be safely used on this host."""


class MpfrPointPlaneBackendError(RuntimeError):
    """The compiled backend rejected a valid-looking evaluation request."""


def _configure_library(library: ctypes.CDLL) -> None:
    library.carts_mpfr_point_plane_abi_version.argtypes = []
    library.carts_mpfr_point_plane_abi_version.restype = ctypes.c_int
    library.carts_mpfr_point_plane_create.argtypes = [
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_int,
        _INT_POINTER,
        _INT_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
    ]
    library.carts_mpfr_point_plane_create.restype = ctypes.c_void_p
    library.carts_mpfr_point_plane_set_triangle.argtypes = [
        ctypes.c_void_p,
        _DOUBLE_POINTER,
    ]
    library.carts_mpfr_point_plane_set_triangle.restype = ctypes.c_int
    library.carts_mpfr_point_plane_evaluate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
    ]
    library.carts_mpfr_point_plane_evaluate.restype = ctypes.c_int
    library.carts_mpfr_point_plane_evaluate_interval.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
    ]
    library.carts_mpfr_point_plane_evaluate_interval.restype = ctypes.c_int
    library.carts_mpfr_point_plane_isolate_monotone_root.argtypes = [
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        _DOUBLE_POINTER,
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.carts_mpfr_point_plane_isolate_monotone_root.restype = ctypes.c_int
    library.carts_mpfr_point_plane_destroy.argtypes = [ctypes.c_void_p]
    library.carts_mpfr_point_plane_destroy.restype = None


def _pointer(array: np.ndarray, c_type: type[ctypes._SimpleCData]):
    return array.ctypes.data_as(ctypes.POINTER(c_type))


def _self_test(library: ctypes.CDLL) -> None:
    if library.carts_mpfr_point_plane_abi_version() != MPFR_POINT_PLANE_ABI_VERSION:
        raise MpfrPointPlaneBackendUnavailable(
            "compiled MPFR point-plane ABI version mismatch"
        )

    joint_types = np.asarray((2,), dtype=np.int32)
    source_indices = np.asarray((0,), dtype=np.int32)
    origin = np.zeros((1, 3), dtype=np.float64)
    axes = np.asarray(((1.0, 0.0, 0.0),), dtype=np.float64)
    unit = np.asarray((1.0,), dtype=np.float64)
    zero = np.asarray((0.0,), dtype=np.float64)
    base = np.asarray(
        ((1.0, 0.0, 0.0, 0.0),
         (0.0, 1.0, 0.0, 0.0),
         (0.0, 0.0, 1.0, 0.0)),
        dtype=np.float64,
    )
    witness = np.zeros(3, dtype=np.float64)
    triangle = np.asarray(
        ((0.25, 0.0, 0.0),
         (0.25, 1.0, 0.0),
         (0.25, 0.0, 1.0)),
        dtype=np.float64,
    )
    handle = library.carts_mpfr_point_plane_create(
        192,
        1,
        1,
        _pointer(joint_types, ctypes.c_int),
        _pointer(source_indices, ctypes.c_int),
        _pointer(origin, ctypes.c_double),
        _pointer(origin, ctypes.c_double),
        _pointer(axes, ctypes.c_double),
        _pointer(unit, ctypes.c_double),
        _pointer(zero, ctypes.c_double),
        _pointer(zero, ctypes.c_double),
        _pointer(unit, ctypes.c_double),
        _pointer(base, ctypes.c_double),
        _pointer(witness, ctypes.c_double),
        _pointer(triangle, ctypes.c_double),
    )
    if not handle:
        raise MpfrPointPlaneBackendUnavailable(
            "compiled MPFR point-plane self-test could not create evaluator"
        )
    try:
        for phase, expected_sign in ((0.125, -1), (0.5, 1)):
            lower = ctypes.c_double()
            upper = ctypes.c_double()
            status = library.carts_mpfr_point_plane_evaluate(
                handle,
                phase,
                ctypes.byref(lower),
                ctypes.byref(upper),
            )
            if (
                status != 0
                or not math.isfinite(lower.value)
                or not math.isfinite(upper.value)
                or lower.value > upper.value
                or not lower.value <= phase - 0.25 <= upper.value
                or (lower.value > 0.0) - (upper.value < 0.0)
                != expected_sign
            ):
                raise MpfrPointPlaneBackendUnavailable(
                    "compiled MPFR point-plane directed-rounding self-test failed"
                )
        plane_lower = ctypes.c_double()
        plane_upper = ctypes.c_double()
        position_lower = np.empty(3, dtype=np.float64)
        position_upper = np.empty(3, dtype=np.float64)
        status = library.carts_mpfr_point_plane_evaluate_interval(
            handle,
            0.125,
            0.5,
            ctypes.byref(plane_lower),
            ctypes.byref(plane_upper),
            _pointer(position_lower, ctypes.c_double),
            _pointer(position_upper, ctypes.c_double),
        )
        if (
            status != 0
            or plane_lower.value > -0.125
            or plane_upper.value < 0.25
            or position_lower[0] > 0.125
            or position_upper[0] < 0.5
            or np.any(position_lower > position_upper)
        ):
            raise MpfrPointPlaneBackendUnavailable(
                "compiled MPFR interval-position self-test failed"
            )
        root_lower = ctypes.c_double()
        root_upper = ctypes.c_double()
        lower_value_lower = ctypes.c_double()
        lower_value_upper = ctypes.c_double()
        upper_value_lower = ctypes.c_double()
        upper_value_upper = ctypes.c_double()
        interpolation_iterations = ctypes.c_int()
        newton_iterations = ctypes.c_int()
        bisection_iterations = ctypes.c_int()
        status = library.carts_mpfr_point_plane_isolate_monotone_root(
            handle,
            0.0,
            0.5,
            1.0,
            1.0,
            -1,
            1,
            256,
            ctypes.byref(root_lower),
            ctypes.byref(root_upper),
            ctypes.byref(lower_value_lower),
            ctypes.byref(lower_value_upper),
            ctypes.byref(upper_value_lower),
            ctypes.byref(upper_value_upper),
            ctypes.byref(interpolation_iterations),
            ctypes.byref(newton_iterations),
            ctypes.byref(bisection_iterations),
        )
        if (
            status != 0
            or not root_lower.value < 0.25 < root_upper.value
            or lower_value_upper.value >= 0.0
            or upper_value_lower.value <= 0.0
            or interpolation_iterations.value + newton_iterations.value <= 0
            or bisection_iterations.value > 2
        ):
            raise MpfrPointPlaneBackendUnavailable(
                "compiled MPFR monotone-root self-test failed"
            )
        rebound_triangle = np.asarray(
            ((0.75, 0.0, 0.0),
             (0.75, 1.0, 0.0),
             (0.75, 0.0, 1.0)),
            dtype=np.float64,
        )
        status = library.carts_mpfr_point_plane_set_triangle(
            handle,
            _pointer(rebound_triangle, ctypes.c_double),
        )
        rebound_lower = ctypes.c_double()
        rebound_upper = ctypes.c_double()
        rebound_evaluate_status = library.carts_mpfr_point_plane_evaluate(
            handle,
            0.5,
            ctypes.byref(rebound_lower),
            ctypes.byref(rebound_upper),
        )
        if (
            status != 0
            or rebound_evaluate_status != 0
            or rebound_upper.value >= 0.0
            or not rebound_lower.value <= -0.25 <= rebound_upper.value
        ):
            raise MpfrPointPlaneBackendUnavailable(
                "compiled MPFR plane-rebind self-test failed"
            )
    finally:
        library.carts_mpfr_point_plane_destroy(handle)


def _compile_and_load_library() -> ctypes.CDLL:
    if platform.system() != "Linux" or platform.machine().lower() not in {
        "x86_64",
        "amd64",
    }:
        raise MpfrPointPlaneBackendUnavailable(
            "manual MPFR runtime ABI is enabled only for Linux x86-64"
        )
    compiler = shutil.which("gcc")
    if compiler is None:
        raise MpfrPointPlaneBackendUnavailable("gcc is unavailable")
    source = Path(__file__).with_name(_SOURCE_NAME)
    try:
        source_bytes = source.read_bytes()
    except OSError as error:
        raise MpfrPointPlaneBackendUnavailable(
            f"MPFR point-plane source is unavailable: {error}"
        ) from error
    digest = hashlib.sha256(
        source_bytes
        + platform.machine().encode("ascii")
        + str(MPFR_POINT_PLANE_ABI_VERSION).encode("ascii")
    ).hexdigest()[:20]
    target = Path(tempfile.gettempdir()) / (
        f"kcg_connector_mpfr_point_plane_{digest}.so"
    )
    lock_path = target.with_suffix(".lock")
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if not target.is_file():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            try:
                completed = subprocess.run(
                    (
                        compiler,
                        "-O3",
                        "-std=c11",
                        "-fPIC",
                        "-shared",
                        str(source),
                        "-o",
                        str(temporary),
                        "-Wl,-z,defs",
                        "-l:libmpfr.so.6",
                        "-l:libgmp.so.10",
                        "-lm",
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60.0,
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout).strip()
                    raise MpfrPointPlaneBackendUnavailable(
                        "compiled MPFR point-plane build failed: "
                        + detail[-3000:]
                    )
                os.replace(temporary, target)
            except (OSError, subprocess.SubprocessError) as error:
                raise MpfrPointPlaneBackendUnavailable(
                    f"compiled MPFR point-plane build could not run: {error}"
                ) from error
            finally:
                temporary.unlink(missing_ok=True)
    try:
        library = ctypes.CDLL(str(target))
        _configure_library(library)
        _self_test(library)
    except (OSError, AttributeError) as error:
        raise MpfrPointPlaneBackendUnavailable(
            f"compiled MPFR point-plane library could not load: {error}"
        ) from error
    return library


def _library() -> ctypes.CDLL:
    global _LIBRARY, _LIBRARY_FAILURE
    with _LIBRARY_LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        if _LIBRARY_FAILURE is not None:
            raise MpfrPointPlaneBackendUnavailable(_LIBRARY_FAILURE)
        try:
            _LIBRARY = _compile_and_load_library()
        except MpfrPointPlaneBackendUnavailable as error:
            _LIBRARY_FAILURE = str(error)
            raise
        return _LIBRARY


def _finite_array(
    value: object,
    *,
    dtype: np.dtype,
    shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise MpfrPointPlaneBackendError(
            f"{label} must be finite with shape {shape}"
        )
    return result


def _destroy_handle(library: ctypes.CDLL, handle_value: int) -> None:
    if handle_value:
        library.carts_mpfr_point_plane_destroy(
            ctypes.c_void_p(handle_value)
        )


class MpfrPointPlaneEvaluator:
    """One persistent exact-point evaluator bound to one contact equation."""

    __slots__ = (
        "_library",
        "_handle",
        "_finalizer",
        "evaluation_count",
        "interval_evaluation_count",
        "root_transaction_count",
        "triangle_rebind_count",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        precision_bits: int,
        joint_types: object,
        source_indices: object,
        origins_xyz_m: object,
        origins_rpy_rad: object,
        axes: object,
        multipliers: object,
        offsets: object,
        q_start: object,
        direction: object,
        base_transform_3x4: object,
        witness_point_local_m: object,
        object_triangle_m: object,
    ) -> None:
        if (
            not isinstance(precision_bits, int)
            or isinstance(precision_bits, bool)
            or precision_bits < 64
        ):
            raise MpfrPointPlaneBackendError(
                "precision_bits must be an integer of at least 64"
            )
        joint_type_array = np.ascontiguousarray(joint_types, dtype=np.int32)
        if joint_type_array.ndim != 1 or joint_type_array.size == 0:
            raise MpfrPointPlaneBackendError(
                "joint_types must be a non-empty vector"
            )
        joint_count = int(joint_type_array.size)
        if np.any((joint_type_array < 0) | (joint_type_array > 2)):
            raise MpfrPointPlaneBackendError("joint type code is unsupported")
        source_array = np.ascontiguousarray(source_indices, dtype=np.int32)
        if source_array.shape != (joint_count,):
            raise MpfrPointPlaneBackendError(
                "source_indices must match joint_types"
            )
        start_array = np.ascontiguousarray(q_start, dtype=np.float64)
        rate_array = np.ascontiguousarray(direction, dtype=np.float64)
        if (
            start_array.ndim != 1
            or start_array.size == 0
            or rate_array.shape != start_array.shape
            or not np.all(np.isfinite(start_array))
            or not np.all(np.isfinite(rate_array))
        ):
            raise MpfrPointPlaneBackendError(
                "q_start and direction must be matching finite vectors"
            )
        independent_count = int(start_array.size)
        movable = joint_type_array != 0
        if np.any(
            movable
            & ((source_array < 0) | (source_array >= independent_count))
        ):
            raise MpfrPointPlaneBackendError(
                "movable joint source index is outside q_start"
            )
        origin_xyz_array = _finite_array(
            origins_xyz_m,
            dtype=np.dtype(np.float64),
            shape=(joint_count, 3),
            label="origins_xyz_m",
        )
        origin_rpy_array = _finite_array(
            origins_rpy_rad,
            dtype=np.dtype(np.float64),
            shape=(joint_count, 3),
            label="origins_rpy_rad",
        )
        axes_array = _finite_array(
            axes,
            dtype=np.dtype(np.float64),
            shape=(joint_count, 3),
            label="axes",
        )
        multiplier_array = _finite_array(
            multipliers,
            dtype=np.dtype(np.float64),
            shape=(joint_count,),
            label="multipliers",
        )
        offset_array = _finite_array(
            offsets,
            dtype=np.dtype(np.float64),
            shape=(joint_count,),
            label="offsets",
        )
        base_array = _finite_array(
            base_transform_3x4,
            dtype=np.dtype(np.float64),
            shape=(3, 4),
            label="base_transform_3x4",
        )
        witness_array = _finite_array(
            witness_point_local_m,
            dtype=np.dtype(np.float64),
            shape=(3,),
            label="witness_point_local_m",
        )
        triangle_array = _finite_array(
            object_triangle_m,
            dtype=np.dtype(np.float64),
            shape=(3, 3),
            label="object_triangle_m",
        )

        library = _library()
        handle = library.carts_mpfr_point_plane_create(
            precision_bits,
            joint_count,
            independent_count,
            _pointer(joint_type_array, ctypes.c_int),
            _pointer(source_array, ctypes.c_int),
            _pointer(origin_xyz_array, ctypes.c_double),
            _pointer(origin_rpy_array, ctypes.c_double),
            _pointer(axes_array, ctypes.c_double),
            _pointer(multiplier_array, ctypes.c_double),
            _pointer(offset_array, ctypes.c_double),
            _pointer(start_array, ctypes.c_double),
            _pointer(rate_array, ctypes.c_double),
            _pointer(base_array, ctypes.c_double),
            _pointer(witness_array, ctypes.c_double),
            _pointer(triangle_array, ctypes.c_double),
        )
        if not handle:
            raise MpfrPointPlaneBackendError(
                "compiled MPFR point-plane evaluator creation failed"
            )
        self._library = library
        self._handle = int(handle)
        self._finalizer = weakref.finalize(
            self, _destroy_handle, library, self._handle
        )
        self.evaluation_count = 0
        self.interval_evaluation_count = 0
        self.root_transaction_count = 0
        self.triangle_rebind_count = 0

    @property
    def closed(self) -> bool:
        return not self._finalizer.alive

    def rebind_object_triangle(self, object_triangle_m: object) -> None:
        """Replace only the plane while preserving the exact path evaluator."""

        if self.closed:
            raise MpfrPointPlaneBackendError("evaluator is closed")
        triangle = _finite_array(
            object_triangle_m,
            dtype=np.dtype(np.float64),
            shape=(3, 3),
            label="object_triangle_m",
        )
        status = self._library.carts_mpfr_point_plane_set_triangle(
            ctypes.c_void_p(self._handle),
            _pointer(triangle, ctypes.c_double),
        )
        if status != 0:
            raise MpfrPointPlaneBackendError(
                "compiled MPFR point-plane triangle rebind failed with "
                f"status {status}"
            )
        self.triangle_rebind_count += 1

    def evaluate(self, phase: float) -> tuple[float, float]:
        phase_value = float(phase)
        if self.closed:
            raise MpfrPointPlaneBackendError("evaluator is closed")
        if not math.isfinite(phase_value):
            raise MpfrPointPlaneBackendError("phase must be finite")
        lower = ctypes.c_double()
        upper = ctypes.c_double()
        status = self._library.carts_mpfr_point_plane_evaluate(
            ctypes.c_void_p(self._handle),
            phase_value,
            ctypes.byref(lower),
            ctypes.byref(upper),
        )
        if status != 0:
            raise MpfrPointPlaneBackendError(
                f"compiled MPFR point-plane evaluation failed with status {status}"
            )
        self.evaluation_count += 1
        # The C result is already directed.  One binary64 step protects the
        # MPFR-to-double boundary and makes containment tests architecture
        # independent without introducing a physical tolerance.
        return (
            float(np.nextafter(lower.value, -math.inf)),
            float(np.nextafter(upper.value, math.inf)),
        )

    def evaluate_interval(
        self,
        phase_lower: float,
        phase_upper: float,
    ) -> tuple[tuple[float, float], tuple[tuple[float, float], ...]]:
        lower_phase = float(phase_lower)
        upper_phase = float(phase_upper)
        if self.closed:
            raise MpfrPointPlaneBackendError("evaluator is closed")
        if (
            not math.isfinite(lower_phase)
            or not math.isfinite(upper_phase)
            or lower_phase > upper_phase
        ):
            raise MpfrPointPlaneBackendError(
                "phase interval must be finite and ordered"
            )
        plane_lower = ctypes.c_double()
        plane_upper = ctypes.c_double()
        position_lower = np.empty(3, dtype=np.float64)
        position_upper = np.empty(3, dtype=np.float64)
        status = self._library.carts_mpfr_point_plane_evaluate_interval(
            ctypes.c_void_p(self._handle),
            lower_phase,
            upper_phase,
            ctypes.byref(plane_lower),
            ctypes.byref(plane_upper),
            _pointer(position_lower, ctypes.c_double),
            _pointer(position_upper, ctypes.c_double),
        )
        if status != 0:
            raise MpfrPointPlaneBackendError(
                "compiled MPFR interval evaluation failed with status "
                f"{status}"
            )
        self.interval_evaluation_count += 1
        plane = (
            float(np.nextafter(plane_lower.value, -math.inf)),
            float(np.nextafter(plane_upper.value, math.inf)),
        )
        positions = tuple(
            (
                float(np.nextafter(position_lower[index], -math.inf)),
                float(np.nextafter(position_upper[index], math.inf)),
            )
            for index in range(3)
        )
        return plane, positions

    def isolate_monotone_root(
        self,
        *,
        phase_lower: float,
        phase_upper: float,
        derivative_lower: float,
        derivative_upper: float,
        lower_sign: int,
        upper_sign: int,
        maximum_iterations: int,
    ) -> tuple[
        float,
        float,
        tuple[float, float],
        tuple[float, float],
        int,
        int,
        int,
    ] | None:
        if self.closed:
            raise MpfrPointPlaneBackendError("evaluator is closed")
        root_lower = ctypes.c_double()
        root_upper = ctypes.c_double()
        lower_value_lower = ctypes.c_double()
        lower_value_upper = ctypes.c_double()
        upper_value_lower = ctypes.c_double()
        upper_value_upper = ctypes.c_double()
        interpolation_iterations = ctypes.c_int()
        newton_iterations = ctypes.c_int()
        bisection_iterations = ctypes.c_int()
        status = (
            self._library.carts_mpfr_point_plane_isolate_monotone_root(
                ctypes.c_void_p(self._handle),
                float(phase_lower),
                float(phase_upper),
                float(derivative_lower),
                float(derivative_upper),
                int(lower_sign),
                int(upper_sign),
                int(maximum_iterations),
                ctypes.byref(root_lower),
                ctypes.byref(root_upper),
                ctypes.byref(lower_value_lower),
                ctypes.byref(lower_value_upper),
                ctypes.byref(upper_value_lower),
                ctypes.byref(upper_value_upper),
                ctypes.byref(interpolation_iterations),
                ctypes.byref(newton_iterations),
                ctypes.byref(bisection_iterations),
            )
        )
        if status in (3, 4, 5):
            return None
        if status != 0:
            raise MpfrPointPlaneBackendError(
                "compiled MPFR monotone-root transaction failed with status "
                f"{status}"
            )
        self.root_transaction_count += 1
        return (
            root_lower.value,
            root_upper.value,
            (
                float(np.nextafter(lower_value_lower.value, -math.inf)),
                float(np.nextafter(lower_value_upper.value, math.inf)),
            ),
            (
                float(np.nextafter(upper_value_lower.value, -math.inf)),
                float(np.nextafter(upper_value_upper.value, math.inf)),
            ),
            interpolation_iterations.value,
            newton_iterations.value,
            bisection_iterations.value,
        )

    def close(self) -> None:
        self._finalizer()

    def __enter__(self) -> "MpfrPointPlaneEvaluator":
        return self

    def __exit__(self, *_arguments: object) -> None:
        self.close()


__all__ = [
    "MPFR_POINT_PLANE_ABI_VERSION",
    "MPFR_POINT_PLANE_METHOD_ID",
    "MpfrPointPlaneBackendError",
    "MpfrPointPlaneBackendUnavailable",
    "MpfrPointPlaneEvaluator",
]
