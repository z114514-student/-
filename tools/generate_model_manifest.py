#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from typing import Any

MAX_ONNX_BYTES = 50_000_000
EXPECTED_INPUT = ("obs", "float32", [1, 1097])
EXPECTED_OUTPUT = ("target_pose", "float32", [1, 225])


def _shape_from_value_info(value_info: Any) -> list[Any]:
    dims = []
    shape = value_info.type.tensor_type.shape.dim
    for d in shape:
        if d.dim_value:
            dims.append(int(d.dim_value))
        elif d.dim_param:
            dims.append(d.dim_param)
        else:
            dims.append(None)
    return dims


def _dtype_str(elem_type: int) -> str:
    try:
        import onnx
        raw = onnx.TensorProto.DataType.Name(elem_type).lower()
        alias = {
            "float": "float32",
            "double": "float64",
            "int64": "int64",
            "int32": "int32",
            "bool": "bool",
        }
        return alias.get(raw, raw)
    except Exception:
        return f"unknown({elem_type})"


def _collect_io(model: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inputs = []
    outputs = []
    for v in model.graph.input:
        if not v.type.HasField("tensor_type"):
            continue
        inputs.append({"name": v.name, "dtype": _dtype_str(v.type.tensor_type.elem_type), "shape": _shape_from_value_info(v)})
    for v in model.graph.output:
        if not v.type.HasField("tensor_type"):
            continue
        outputs.append({"name": v.name, "dtype": _dtype_str(v.type.tensor_type.elem_type), "shape": _shape_from_value_info(v)})
    return inputs, outputs


def _has_external_data(model: Any) -> bool:
    from onnx.external_data_helper import uses_external_data

    for t in list(model.graph.initializer):
        if uses_external_data(t):
            return True
    return False


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_expected_interface(inputs: list[dict[str, Any]], outputs: list[dict[str, Any]]) -> bool:
    if len(inputs) != 1 or len(outputs) != 1:
        return False
    i = inputs[0]
    o = outputs[0]
    return (
        i["name"] == EXPECTED_INPUT[0]
        and i["dtype"] == EXPECTED_INPUT[1]
        and i["shape"] == EXPECTED_INPUT[2]
        and o["name"] == EXPECTED_OUTPUT[0]
        and o["dtype"] == EXPECTED_OUTPUT[1]
        and o["shape"] == EXPECTED_OUTPUT[2]
    )


def _load_official_validation(path: str | None) -> dict[str, Any]:
    if not path:
        return {"status": "not_provided"}
    with open(path, "r", encoding="utf-8") as f:
        return {"status": "provided", "details": json.load(f)}


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 ONNX 模型交付清单")
    parser.add_argument("--onnx", required=True, help="ONNX 文件路径")
    parser.add_argument("--source-checkpoint", required=True, help="来源 checkpoint 描述")
    parser.add_argument("--training-stage", required=True, help="训练阶段描述")
    parser.add_argument("--official-validation-json", default=None, help="官方运行验证结果 JSON（可选）")
    parser.add_argument("--output", default="-", help="输出 JSON 文件，默认 stdout")
    args = parser.parse_args()

    onnx_path = os.path.abspath(args.onnx)
    if not os.path.isfile(onnx_path):
        print(f"[ERROR] ONNX 文件不存在: {onnx_path}", file=sys.stderr)
        return 2

    try:
        import onnx
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] 缺少 onnx 依赖: {exc}", file=sys.stderr)
        return 2

    size_bytes = os.path.getsize(onnx_path)
    model = onnx.load(onnx_path, load_external_data=False)
    inputs, outputs = _collect_io(model)
    has_external = _has_external_data(model)

    result = {
        "file_name": os.path.basename(onnx_path),
        "absolute_path": onnx_path,
        "bytes": size_bytes,
        "sha256": _sha256(onnx_path),
        "declared_source_checkpoint": args.source_checkpoint,
        "declared_training_stage": args.training_stage,
        "interface": {
            "inputs": inputs,
            "outputs": outputs,
        },
        "static_checks": {
            "size_limit_bytes": MAX_ONNX_BYTES,
            "size_limit_pass": size_bytes < MAX_ONNX_BYTES,
            "expected_interface": {
                "input": {"name": EXPECTED_INPUT[0], "dtype": EXPECTED_INPUT[1], "shape": EXPECTED_INPUT[2]},
                "output": {"name": EXPECTED_OUTPUT[0], "dtype": EXPECTED_OUTPUT[1], "shape": EXPECTED_OUTPUT[2]},
            },
            "expected_interface_pass": _check_expected_interface(inputs, outputs),
            "has_external_data": has_external,
            "external_data_pass": not has_external,
        },
        "official_validation": _load_official_validation(args.official_validation_json),
        "notes": "静态检查通过不代表官方运行验收通过。",
    }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output == "-":
        print(text)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
