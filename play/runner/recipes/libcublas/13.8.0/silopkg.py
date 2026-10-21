import shutil

from silo import Component, Package, SharedLib, SiloPackage

DLLS = ("cublas64_13.dll", "cublasLt64_13.dll")


class LibCublas(SiloPackage):
    """NVIDIA cuBLAS for CUDA 13, the redistributable zip."""

    name = "libcublas"
    version = "13.8.0"
    ignore_settings = ("compiler", "build_type", "cppstd", "libcxx")

    def source(self, src):
        src.fetch(
            "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-13.8.0.4-archive.zip",
            sha256="0974318e9861a61cb9091cf7de8d4c2880e9787fe311cb4bcbd08865663c5f43",
        )

    def build(self):
        bin = self.install_dir / "bin"
        bin.mkdir(parents=True)
        for dll in DLLS:
            shutil.copy2(next(self.source_dir.rglob(dll)), bin)

    def package(self):
        runtime = Component("cublas", shared_libs=[SharedLib(f"bin/{dll}") for dll in DLLS])
        return Package(components=[runtime], licenses=["LICENSE"])
