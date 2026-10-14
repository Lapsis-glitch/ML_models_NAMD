"""
Tests for the one-call builder (``src.build``) and the shared helpers of the
optimised builds (``src.fast``).

The compile steps and the native libraries are mocked: these tests check how
``src.build`` chains compile -> native libraries -> wrap (which arguments reach
``src.cli``), not the models themselves.  Weight-free, CPU only.
"""

import pytest
import torch

import src.build as build
import src.cli
import src.fast.native as native
from src.fast import water_box


class _Tiny(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


@pytest.fixture
def scripted(tmp_path):
    p = tmp_path / "scripted.pt"
    torch.jit.script(_Tiny()).save(str(p))
    return str(p)


@pytest.fixture
def calls(monkeypatch):
    """Record what src.build hands to each step instead of running it."""
    rec = {"compile": [], "wrap": [], "subprocess": [], "ensure": []}

    def fake_compile(a, inner):
        rec["compile"].append((a.model_type, a.source, a.fast, inner))
        libset = {"mace": "cueq", "nequip": "oeq", "sevennet": "oeq",
                  "torchani": "cuaev"}.get(a.model_type, "") if a.fast else ""
        extra = ["--elements", "1,6,7,8"] if a.model_type == "torchani" else []
        if a.model_type == "torchani" and a.fast:
            extra.append("--fast")
        return libset, extra

    def fake_ensure(name, libtorch=None, rebuild=False):
        rec["ensure"].append((name, libtorch))
        return [f"/libs/{name}_a.so", f"/libs/{name}_b.so"]

    monkeypatch.setattr(build, "compile_inner", fake_compile)
    monkeypatch.setattr(src.cli, "main", lambda argv: rec["wrap"].append(argv))
    monkeypatch.setattr(native, "ensure", fake_ensure)
    monkeypatch.setattr(native, "ensure_cuda_lib_path", lambda module: None)
    monkeypatch.setattr(build.subprocess, "run",
                        lambda cmd, **kw: rec["subprocess"].append(cmd))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return rec


def _opt(argv, name):
    return argv[argv.index(name) + 1]


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def test_is_torchscript(tmp_path, scripted):
    saved = tmp_path / "state.pt"
    torch.save({"w": torch.zeros(3)}, str(saved))
    assert build.is_torchscript(scripted)
    assert not build.is_torchscript(str(saved))
    assert not build.is_torchscript(str(tmp_path / "missing.pt"))
    assert not build.is_torchscript("7net-0")


def test_inner_path():
    assert build.inner_path("/m/mlff.pt", "mace") == "/m/mlff_inner.pt"
    assert build.inner_path("/m/mlff.pt", "nequip") == "/m/mlff_inner.nequip.pth"
    assert build.inner_path("/m/mlff", "allegro") == "/m/mlff_inner.nequip.pth"


def test_water_box():
    xyz = water_box(300)
    assert xyz.shape == (300, 3) and xyz.dtype == torch.float64
    d = torch.cdist(xyz, xyz) + 1e3 * torch.eye(300, dtype=torch.float64)
    assert d.min() > 0.8  # no overlapping atoms
    assert torch.equal(water_box(30), water_box(30))  # deterministic


def test_native_lib_sets():
    cueq = native.lib_paths("cueq")  # [nvrtc,] libcue_ops, uniform_1d op
    assert cueq[-2].endswith("libcue_ops.so") and cueq[-1].endswith("libcueq_uniform1d_native.so")
    assert native.lib_paths("oeq")[0].endswith("liboeq_native.so")
    assert native.lib_paths("cuaev")[0].endswith("libcuaev_native_precise.so")


# --------------------------------------------------------------------------- #
#  src.build: compile -> libraries -> wrap
# --------------------------------------------------------------------------- #
def test_reference_build_wraps_in_process(tmp_path, calls):
    out = str(tmp_path / "mlff.pt")
    build.main(["mace", "MACE-OFF23_medium.model", "--out", out])
    assert calls["compile"] == [("mace", "MACE-OFF23_medium.model", False,
                                 str(tmp_path / "mlff_inner.pt"))]
    assert calls["ensure"] == [] and calls["subprocess"] == []
    (argv,) = calls["wrap"]
    assert _opt(argv, "--model-type") == "mace"
    assert _opt(argv, "--compiled") == str(tmp_path / "mlff_inner.pt")
    assert _opt(argv, "--out") == out
    assert "--extra-libs" not in argv


@pytest.mark.parametrize("model_type,libset", [("mace", "cueq"), ("nequip", "oeq"),
                                               ("sevennet", "oeq"), ("torchani", "cuaev")])
def test_fast_build_loads_native_libs_in_fresh_process(tmp_path, calls, capsys,
                                                       model_type, libset):
    out = str(tmp_path / "mlff.pt")
    build.main([model_type, "src", "--fast", "--libtorch", "/opt/libtorch", "--out", out])
    assert calls["ensure"] == [(libset, "/opt/libtorch")]
    assert calls["wrap"] == []  # wrapped in a subprocess, as NAMD loads it
    (cmd,) = calls["subprocess"]
    assert cmd[1:3] == ["-m", "src.cli"]
    libs = f"/libs/{libset}_a.so:/libs/{libset}_b.so"
    assert _opt(cmd, "--extra-libs") == libs
    assert f"export NAMD_MLFF_EXTRA_LIBS={libs}" in capsys.readouterr().out


def test_torchani_elements_and_lean(tmp_path, calls):
    build.main(["torchani", "ani1x", "--fast", "--out", str(tmp_path / "m.pt")])
    (cmd,) = calls["subprocess"]
    assert _opt(cmd, "--elements") == "1,6,7,8" and "--fast" in cmd


def test_schnet_fast_is_a_wrap_flag(tmp_path, calls, scripted):
    build.main(["schnet", scripted, "--r-max", "5.0", "--fast", "--out", str(tmp_path / "m.pt")])
    assert calls["compile"] == [] and calls["ensure"] == []
    (argv,) = calls["wrap"]
    assert _opt(argv, "--compiled") == scripted
    assert _opt(argv, "--r-max") == "5.0" and "--fast" in argv


def test_prebuilt_source_skips_compile(tmp_path, calls, scripted):
    build.main(["torchani", scripted, "--elements", "1,6,7,8", "--out", str(tmp_path / "m.pt")])
    assert calls["compile"] == []
    (argv,) = calls["wrap"]
    assert _opt(argv, "--compiled") == scripted and _opt(argv, "--elements") == "1,6,7,8"


def test_xmace_state_and_d3(tmp_path, calls):
    build.main(["xmace", "fulvene.model", "--state", "2", "--d3", "--out", str(tmp_path / "m.pt")])
    (argv,) = calls["wrap"]
    assert _opt(argv, "--state") == "2"
    assert "--d3" in argv and _opt(argv, "--d3-functional") == "pbe"


@pytest.mark.parametrize("model_type", ["allegro", "xmace"])
def test_fast_rejected_without_optimised_build(tmp_path, calls, model_type):
    with pytest.raises(SystemExit):
        build.main([model_type, "src", "--fast", "--out", str(tmp_path / "m.pt")])
    assert calls["compile"] == []


def test_fast_rejects_compiled_source(tmp_path, calls, scripted):
    with pytest.raises(SystemExit):
        build.main(["mace", scripted, "--fast", "--out", str(tmp_path / "m.pt")])
    assert calls["compile"] == []


@pytest.mark.parametrize("model_type", ["mace", "sevennet", "torchani"])
def test_fast_needs_gpu(tmp_path, calls, monkeypatch, model_type):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit):
        build.main([model_type, "src", "--fast", "--out", str(tmp_path / "m.pt")])
    assert calls["compile"] == [] and calls["ensure"] == []


def test_schnet_fast_runs_without_gpu(tmp_path, calls, monkeypatch, scripted):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    build.main(["schnet", scripted, "--r-max", "5.0", "--fast", "--out", str(tmp_path / "m.pt")])
    assert "--fast" in calls["wrap"][0]


@pytest.mark.parametrize("gpu,device", [(True, "cuda"), (False, "cpu")])
def test_compile_device_default(tmp_path, calls, monkeypatch, gpu, device):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu)
    seen = {}
    monkeypatch.setattr(build, "compile_inner",
                        lambda a, inner: seen.setdefault("dev", a.compile_device) and ("", []))
    build.main(["nequip", "pkg.nequip.zip", "--out", str(tmp_path / "m.pt")])
    assert seen["dev"] == device
