import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ManifestToolTest(unittest.TestCase):
    @unittest.skipUnless(__import__("importlib").util.find_spec("onnx") is not None, "onnx not installed")
    def test_manifest_expected_interface(self):
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model_path = td_path / "ok.onnx"
            input_tensor = helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 1097])
            output_tensor = helper.make_tensor_value_info("target_pose", TensorProto.FLOAT, [1, 225])
            node = helper.make_node("Identity", inputs=["obs"], outputs=["target_pose"])
            graph = helper.make_graph([node], "g", [input_tensor], [output_tensor])
            model = helper.make_model(graph)
            onnx.save(model, str(model_path))

            cmd = [
                sys.executable,
                "tools/generate_model_manifest.py",
                "--onnx",
                str(model_path),
                "--source-checkpoint",
                "ckpt.pt",
                "--training-stage",
                "sl-20",
            ]
            out = subprocess.check_output(cmd, text=True)
            data = json.loads(out)
            self.assertTrue(data["static_checks"]["size_limit_pass"])
            self.assertTrue(data["static_checks"]["expected_interface_pass"])
            self.assertTrue(data["static_checks"]["external_data_pass"])

    @unittest.skipUnless(__import__("importlib").util.find_spec("onnx") is not None, "onnx not installed")
    def test_manifest_detects_interface_mismatch(self):
        import onnx
        from onnx import TensorProto, helper

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model_path = td_path / "bad.onnx"
            input_tensor = helper.make_tensor_value_info("wrong_obs", TensorProto.FLOAT, [1, 100])
            output_tensor = helper.make_tensor_value_info("wrong_out", TensorProto.FLOAT, [1, 10])
            node = helper.make_node("Identity", inputs=["wrong_obs"], outputs=["wrong_out"])
            graph = helper.make_graph([node], "g", [input_tensor], [output_tensor])
            model = helper.make_model(graph)
            onnx.save(model, str(model_path))

            cmd = [
                sys.executable,
                "tools/generate_model_manifest.py",
                "--onnx",
                str(model_path),
                "--source-checkpoint",
                "ckpt.pt",
                "--training-stage",
                "sl-20",
            ]
            out = subprocess.check_output(cmd, text=True)
            data = json.loads(out)
            self.assertFalse(data["static_checks"]["expected_interface_pass"])
            self.assertIn("sha256", data)


if __name__ == "__main__":
    unittest.main()
