from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _mil_header() -> str:
    return (
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}})]\n"
    )


def _mil_common_conv_consts() -> list[str]:
    return [
        "        string pt = const()[name = string(\"pt\"), val = string(\"valid\")];\n",
        "        tensor<int32, [2]> st = const()[name = string(\"st\"), val = tensor<int32, [2]>([1, 1])];\n",
        "        tensor<int32, [4]> pd = const()[name = string(\"pd\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n",
        "        tensor<int32, [2]> dl = const()[name = string(\"dl\"), val = tensor<int32, [2]>([1, 1])];\n",
        "        int32 gr = const()[name = string(\"gr\"), val = int32(1)];\n",
    ]


def _mil_linear_conv(in_features: int, out_features: int, spatial: int = 1) -> str:
    lines = [
        _mil_header(),
        "{\n",
        f"    func main<ios18>(tensor<fp16, [1, {in_features}, 1, {spatial}]> x) {{\n",
    ]
    lines.extend(_mil_common_conv_consts())
    lines.extend(
        [
            f"        tensor<fp16, [{out_features}, {in_features}, 1, 1]> W = const()[name = string(\"W\"), ",
            f"val = tensor<fp16, [{out_features}, {in_features}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n",
            f"        tensor<fp16, [1, {out_features}, 1, {spatial}]> out = conv(dilations = dl, groups = gr, ",
            "pad = pd, pad_type = pt, strides = st, weight = W, x = x)[name = string(\"conv\")];\n",
            "    } -> (out);\n",
            "}\n",
        ]
    )
    return "".join(lines)


def _mil_swiglu_ffn_conv(dim: int, hidden: int, spatial: int = 1) -> str:
    lines = [
        _mil_header(),
        "{\n",
        f"    func main<ios18>(tensor<fp16, [1, {dim}, 1, {spatial}]> x) {{\n",
    ]
    lines.extend(_mil_common_conv_consts())
    lines.extend(
        [
            f"        tensor<fp16, [{hidden}, {dim}, 1, 1]> W1 = const()[name = string(\"W1\"), ",
            f"val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w1.bin\"), offset = uint64(64)))];\n",
            f"        tensor<fp16, [{hidden}, {dim}, 1, 1]> W3 = const()[name = string(\"W3\"), ",
            f"val = tensor<fp16, [{hidden}, {dim}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w3.bin\"), offset = uint64(64)))];\n",
            f"        tensor<fp16, [{dim}, {hidden}, 1, 1]> W2 = const()[name = string(\"W2\"), ",
            f"val = tensor<fp16, [{dim}, {hidden}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/w2.bin\"), offset = uint64(64)))];\n",
            f"        tensor<fp16, [1, {hidden}, 1, {spatial}]> h1 = conv(dilations = dl, groups = gr, ",
            "pad = pd, pad_type = pt, strides = st, weight = W1, x = x)[name = string(\"c1\")];\n",
            f"        tensor<fp16, [1, {hidden}, 1, {spatial}]> h3 = conv(dilations = dl, groups = gr, ",
            "pad = pd, pad_type = pt, strides = st, weight = W3, x = x)[name = string(\"c3\")];\n",
            f"        tensor<fp16, [1, {hidden}, 1, {spatial}]> sig = sigmoid(x = h1)[name = string(\"sg\")];\n",
            f"        tensor<fp16, [1, {hidden}, 1, {spatial}]> silu = mul(x = h1, y = sig)[name = string(\"si\")];\n",
            f"        tensor<fp16, [1, {hidden}, 1, {spatial}]> gate = mul(x = silu, y = h3)[name = string(\"gt\")];\n",
            f"        tensor<fp16, [1, {dim}, 1, {spatial}]> out = conv(dilations = dl, groups = gr, ",
            "pad = pd, pad_type = pt, strides = st, weight = W2, x = gate)[name = string(\"c2\")];\n",
            "    } -> (out);\n",
            "}\n",
        ]
    )
    return "".join(lines)


def _mil_swiglu_ffn_tiled_conv(dim: int, hidden_tiles: list[int], spatial: int = 1) -> str:
    if not hidden_tiles:
        raise ValueError("hidden_tiles cannot be empty")

    lines: list[str] = [
        _mil_header(),
        "{\n",
        f"    func main<ios18>(tensor<fp16, [1, {dim}, 1, {spatial}]> x) {{\n",
    ]
    lines.extend(_mil_common_conv_consts())

    acc_name = ""
    for i, tile in enumerate(hidden_tiles):
        lines.append(
            f"        tensor<fp16, [{tile}, {dim}, 1, 1]> W1_{i} = "
            f"const()[name = string(\"W1_{i}\"), val = tensor<fp16, [{tile}, {dim}, 1, 1]>"
            f"(BLOBFILE(path = string(\"@model_path/weights/w1_{i}.bin\"), offset = uint64(64)))];\n"
        )
        lines.append(
            f"        tensor<fp16, [{tile}, {dim}, 1, 1]> W3_{i} = "
            f"const()[name = string(\"W3_{i}\"), val = tensor<fp16, [{tile}, {dim}, 1, 1]>"
            f"(BLOBFILE(path = string(\"@model_path/weights/w3_{i}.bin\"), offset = uint64(64)))];\n"
        )
        lines.append(
            f"        tensor<fp16, [{dim}, {tile}, 1, 1]> W2_{i} = "
            f"const()[name = string(\"W2_{i}\"), val = tensor<fp16, [{dim}, {tile}, 1, 1]>"
            f"(BLOBFILE(path = string(\"@model_path/weights/w2_{i}.bin\"), offset = uint64(64)))];\n"
        )
        lines.append(
            f"        tensor<fp16, [1, {tile}, 1, {spatial}]> h1_{i} = "
            "conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, "
            f"weight = W1_{i}, x = x)[name = string(\"c1_{i}\")];\n"
        )
        lines.append(
            f"        tensor<fp16, [1, {tile}, 1, {spatial}]> h3_{i} = "
            "conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, "
            f"weight = W3_{i}, x = x)[name = string(\"c3_{i}\")];\n"
        )
        lines.append(
            f"        tensor<fp16, [1, {tile}, 1, {spatial}]> sig_{i} = "
            f"sigmoid(x = h1_{i})[name = string(\"sg_{i}\")];\n"
        )
        lines.append(
            f"        tensor<fp16, [1, {tile}, 1, {spatial}]> silu_{i} = "
            f"mul(x = h1_{i}, y = sig_{i})[name = string(\"si_{i}\")];\n"
        )
        lines.append(
            f"        tensor<fp16, [1, {tile}, 1, {spatial}]> gate_{i} = "
            f"mul(x = silu_{i}, y = h3_{i})[name = string(\"gt_{i}\")];\n"
        )
        lines.append(
            f"        tensor<fp16, [1, {dim}, 1, {spatial}]> out_{i} = "
            "conv(dilations = dl, groups = gr, pad = pd, pad_type = pt, strides = st, "
            f"weight = W2_{i}, x = gate_{i})[name = string(\"c2_{i}\")];\n"
        )

        if i == 0:
            acc_name = f"out_{i}"
            continue

        next_acc = f"sum_{i}"
        lines.append(
            f"        tensor<fp16, [1, {dim}, 1, {spatial}]> {next_acc} = "
            f"add(x = {acc_name}, y = out_{i})[name = string(\"add_{i}\")];\n"
        )
        acc_name = next_acc

    lines.append(f"    }} -> ({acc_name});\n")
    lines.append("}\n")
    return "".join(lines)


class _ANEKernelHandle(ctypes.Structure):
    pass


_ANEKernelPtr = ctypes.POINTER(_ANEKernelHandle)


@dataclass
class ANEKernel:
    lib: ctypes.CDLL
    ptr: _ANEKernelPtr
    input_channels: int
    output_channels: int
    spatial: int

    def run_fp16(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float16:
            raise TypeError("ANEKernel expects np.float16 input")
        expected = self.input_channels * self.spatial
        if x.size != expected:
            raise ValueError(f"expected {expected} elems, got {x.size}")

        in_arr = np.ascontiguousarray(x)
        out_arr = np.empty(self.output_channels * self.spatial, dtype=np.float16)

        self.lib.ane_bridge_write_input(
            self.ptr,
            0,
            in_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(in_arr.nbytes),
        )

        ok = self.lib.ane_bridge_eval(self.ptr)
        if not ok:
            raise RuntimeError("ane_bridge_eval failed")

        self.lib.ane_bridge_read_output(
            self.ptr,
            0,
            out_arr.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(out_arr.nbytes),
        )
        return out_arr

    def run_vec_fp16(self, x: np.ndarray) -> np.ndarray:
        if x.dtype != np.float16:
            raise TypeError("ANEKernel expects np.float16 input")
        if x.size != self.input_channels:
            raise ValueError(f"expected vector of {self.input_channels} elems, got {x.size}")

        if self.spatial == 1:
            return self.run_fp16(x)

        buf = np.zeros(self.input_channels * self.spatial, dtype=np.float16)
        buf[0::self.spatial] = np.ascontiguousarray(x)
        out_full = self.run_fp16(buf)
        return out_full[0::self.spatial]

    def close(self) -> None:
        if self.ptr:
            self.lib.ane_bridge_free(self.ptr)
            self.ptr = _ANEKernelPtr()


class ANEBridge:
    def __init__(self, lib_path: str | Path):
        self.lib_path = str(lib_path)
        self.lib = ctypes.CDLL(self.lib_path)
        self._bind()
        rc = self.lib.ane_bridge_init()
        if rc != 0:
            raise RuntimeError("ane_bridge_init failed")

    def _bind(self) -> None:
        lib = self.lib
        self.libc = ctypes.CDLL(None)
        self.libc.free.argtypes = [ctypes.c_void_p]
        self.libc.free.restype = None

        lib.ane_bridge_init.argtypes = []
        lib.ane_bridge_init.restype = ctypes.c_int

        lib.ane_bridge_compile.argtypes = [
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.ane_bridge_compile.restype = _ANEKernelPtr

        self.has_compile_multi = hasattr(lib, "ane_bridge_compile_multi_weights")
        if self.has_compile_multi:
            lib.ane_bridge_compile_multi_weights.argtypes = [
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            lib.ane_bridge_compile_multi_weights.restype = _ANEKernelPtr

        lib.ane_bridge_eval.argtypes = [_ANEKernelPtr]
        lib.ane_bridge_eval.restype = ctypes.c_bool

        lib.ane_bridge_write_input.argtypes = [
            _ANEKernelPtr,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.ane_bridge_write_input.restype = None

        lib.ane_bridge_read_output.argtypes = [
            _ANEKernelPtr,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.ane_bridge_read_output.restype = None

        lib.ane_bridge_free.argtypes = [_ANEKernelPtr]
        lib.ane_bridge_free.restype = None

        lib.ane_bridge_get_compile_count.argtypes = []
        lib.ane_bridge_get_compile_count.restype = ctypes.c_int

        self.has_build_blob = hasattr(lib, "ane_bridge_build_weight_blob")
        if self.has_build_blob:
            lib.ane_bridge_build_weight_blob.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_size_t),
            ]
            lib.ane_bridge_build_weight_blob.restype = ctypes.POINTER(ctypes.c_uint8)

    def _build_weight_blob(self, weights_out_in_f32: np.ndarray) -> bytes:
        if weights_out_in_f32.dtype != np.float32:
            raise TypeError("weights must be float32")
        if weights_out_in_f32.ndim != 2:
            raise ValueError("weights must be rank-2 [out, in]")

        out_features, in_features = weights_out_in_f32.shape
        contiguous = np.ascontiguousarray(weights_out_in_f32)

        if self.has_build_blob:
            out_len = ctypes.c_size_t(0)
            ptr = self.lib.ane_bridge_build_weight_blob(
                contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(out_features),
                ctypes.c_int(in_features),
                ctypes.byref(out_len),
            )
            if not ptr:
                raise RuntimeError("ane_bridge_build_weight_blob failed")
            try:
                return ctypes.string_at(ptr, out_len.value)
            finally:
                self.libc.free(ctypes.cast(ptr, ctypes.c_void_p))

        fp16 = np.ascontiguousarray(contiguous.astype(np.float16, copy=False))
        payload = fp16.tobytes(order="C")
        total = 128 + len(payload)
        blob = bytearray(total)
        blob[0] = 0x01
        blob[4] = 0x02
        blob[64] = 0xEF
        blob[65] = 0xBE
        blob[66] = 0xAD
        blob[67] = 0xDE
        blob[68] = 0x01
        data_size = len(payload)
        blob[72:76] = int(data_size).to_bytes(4, "little", signed=False)
        blob[80:84] = int(128).to_bytes(4, "little", signed=False)
        blob[128:128 + data_size] = payload
        return bytes(blob)

    def compile_linear(self, weight_out_in_f32: np.ndarray, spatial: int = 32) -> ANEKernel:
        out_features, in_features = weight_out_in_f32.shape
        mil = _mil_linear_conv(in_features, out_features, spatial=spatial)
        weight_blob = self._build_weight_blob(weight_out_in_f32)

        return self._compile_single_io(
            mil=mil,
            weights={"@model_path/weights/weight.bin": weight_blob},
            input_channels=in_features,
            output_channels=out_features,
            spatial=spatial,
        )

    def compile_fused_swiglu_ffn(
        self,
        w1_hidden_dim: np.ndarray,
        w3_hidden_dim: np.ndarray,
        w2_dim_hidden: np.ndarray,
        spatial: int = 32,
    ) -> ANEKernel:
        dim, hidden = self._validate_swiglu_shapes(w1_hidden_dim, w3_hidden_dim, w2_dim_hidden)
        mil = _mil_swiglu_ffn_conv(dim=dim, hidden=hidden, spatial=spatial)
        return self._compile_single_io(
            mil=mil,
            weights={
                "@model_path/weights/w1.bin": self._build_weight_blob(w1_hidden_dim),
                "@model_path/weights/w3.bin": self._build_weight_blob(w3_hidden_dim),
                "@model_path/weights/w2.bin": self._build_weight_blob(w2_dim_hidden),
            },
            input_channels=dim,
            output_channels=dim,
            spatial=spatial,
        )

    def compile_tiled_fused_swiglu_ffn(
        self,
        w1_hidden_dim: np.ndarray,
        w3_hidden_dim: np.ndarray,
        w2_dim_hidden: np.ndarray,
        hidden_tile: int = 2048,
        spatial: int = 32,
    ) -> ANEKernel:
        dim, hidden = self._validate_swiglu_shapes(w1_hidden_dim, w3_hidden_dim, w2_dim_hidden)
        if hidden_tile < 1:
            raise ValueError("hidden_tile must be >= 1")

        hidden_tiles: list[int] = []
        start = 0
        while start < hidden:
            tile = min(hidden_tile, hidden - start)
            hidden_tiles.append(tile)
            start += tile

        mil = _mil_swiglu_ffn_tiled_conv(dim=dim, hidden_tiles=hidden_tiles, spatial=spatial)

        weights: dict[str, bytes] = {}
        offset = 0
        for i, tile in enumerate(hidden_tiles):
            chunk = slice(offset, offset + tile)
            w1_chunk = np.ascontiguousarray(w1_hidden_dim[chunk, :], dtype=np.float32)
            w3_chunk = np.ascontiguousarray(w3_hidden_dim[chunk, :], dtype=np.float32)
            w2_chunk = np.ascontiguousarray(w2_dim_hidden[:, chunk], dtype=np.float32)
            weights[f"@model_path/weights/w1_{i}.bin"] = self._build_weight_blob(w1_chunk)
            weights[f"@model_path/weights/w3_{i}.bin"] = self._build_weight_blob(w3_chunk)
            weights[f"@model_path/weights/w2_{i}.bin"] = self._build_weight_blob(w2_chunk)
            offset += tile

        return self._compile_single_io(
            mil=mil,
            weights=weights,
            input_channels=dim,
            output_channels=dim,
            spatial=spatial,
        )

    def _validate_swiglu_shapes(
        self,
        w1_hidden_dim: np.ndarray,
        w3_hidden_dim: np.ndarray,
        w2_dim_hidden: np.ndarray,
    ) -> tuple[int, int]:
        if w1_hidden_dim.ndim != 2 or w3_hidden_dim.ndim != 2 or w2_dim_hidden.ndim != 2:
            raise ValueError("all weights must be rank-2")
        if (
            w1_hidden_dim.dtype != np.float32
            or w3_hidden_dim.dtype != np.float32
            or w2_dim_hidden.dtype != np.float32
        ):
            raise TypeError("all weights must be float32")

        hidden_1, dim_1 = w1_hidden_dim.shape
        hidden_3, dim_3 = w3_hidden_dim.shape
        dim_2, hidden_2 = w2_dim_hidden.shape
        if hidden_1 != hidden_3 or dim_1 != dim_3:
            raise ValueError("w1 and w3 must have identical [hidden, dim] shape")
        if dim_2 != dim_1 or hidden_2 != hidden_1:
            raise ValueError("w2 must have shape [dim, hidden] matching w1/w3")
        return dim_1, hidden_1

    def _compile_single_io(
        self,
        mil: str,
        weights: dict[str, bytes],
        input_channels: int,
        output_channels: int,
        spatial: int,
    ) -> ANEKernel:
        mil_bytes = mil.encode("utf-8")
        in_bytes = (ctypes.c_size_t * 1)(ctypes.c_size_t(input_channels * spatial * 2))
        out_bytes = (ctypes.c_size_t * 1)(ctypes.c_size_t(output_channels * spatial * 2))

        if len(weights) <= 1:
            if weights:
                (_, blob), = weights.items()
                weight_arr = np.frombuffer(blob, dtype=np.uint8)
                weight_ptr = weight_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
                weight_len = ctypes.c_size_t(weight_arr.nbytes)
            else:
                weight_arr = None
                weight_ptr = ctypes.POINTER(ctypes.c_uint8)()
                weight_len = ctypes.c_size_t(0)

            ptr = self.lib.ane_bridge_compile(
                ctypes.c_char_p(mil_bytes),
                ctypes.c_size_t(len(mil_bytes)),
                weight_ptr,
                weight_len,
                ctypes.c_int(1),
                in_bytes,
                ctypes.c_int(1),
                out_bytes,
            )
        else:
            if not self.has_compile_multi:
                raise RuntimeError(
                    "Bridge dylib is missing ane_bridge_compile_multi_weights; "
                    "rebuild bridge/libane_bridge.dylib from source."
                )

            names = list(weights.keys())
            blobs = [np.frombuffer(weights[n], dtype=np.uint8) for n in names]
            n_weights = len(names)
            c_names = (ctypes.c_char_p * n_weights)(*[n.encode("utf-8") for n in names])
            c_datas = (ctypes.POINTER(ctypes.c_uint8) * n_weights)(
                *[b.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)) for b in blobs]
            )
            c_lens = (ctypes.c_size_t * n_weights)(*[ctypes.c_size_t(b.nbytes) for b in blobs])

            ptr = self.lib.ane_bridge_compile_multi_weights(
                ctypes.c_char_p(mil_bytes),
                ctypes.c_size_t(len(mil_bytes)),
                c_names,
                c_datas,
                c_lens,
                ctypes.c_int(n_weights),
                ctypes.c_int(1),
                in_bytes,
                ctypes.c_int(1),
                out_bytes,
            )

        if not ptr:
            raise RuntimeError(
                f"ANE compile failed [in={input_channels}, out={output_channels}, spatial={spatial}, n_weights={len(weights)}]"
            )
        return ANEKernel(
            lib=self.lib,
            ptr=ptr,
            input_channels=input_channels,
            output_channels=output_channels,
            spatial=spatial,
        )

    def compile_count(self) -> int:
        return int(self.lib.ane_bridge_get_compile_count())
