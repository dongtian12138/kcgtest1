from pathlib import Path
import runpy


PACKAGE_ROOT = Path(__file__).parents[1]


def test_setup_installs_public_d38999_sources_and_design_docs(monkeypatch):
    captured = {}

    def capture_setup(**kwargs):
        captured.update(kwargs)

    monkeypatch.chdir(PACKAGE_ROOT)
    monkeypatch.setattr("setuptools.setup", capture_setup)
    runpy.run_path(str(PACKAGE_ROOT / "setup.py"), run_name="__main__")
    installed = {
        Path(source).as_posix()
        for _, sources in captured["data_files"]
        for source in sources
    }
    assert {
        "assets/public_specs/mil_dtl_38999/SOURCE.md",
        "assets/public_specs/mil_dtl_38999/dtl38999ss20.pdf",
        "assets/public_specs/mil_dtl_38999/dtl38999ss26.pdf",
        "docs/wrist_ft_v1_design.md",
    } <= installed
