import shutil

from silo import Component, Package, SharedLib, SiloPackage

DLLS = (
    "cudnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
)


class Cudnn(SiloPackage):
    """NVIDIA cuDNN 9 for CUDA 13, the redistributable zip, with only the DLLs ONNX Runtime needs for convolutions."""

    name = "cudnn"
    version = "9.27.0"
    ignore_settings = ("compiler", "build_type", "cppstd", "libcxx")

    def source(self, src):
        src.fetch(
            "https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.27.0.42_cuda13-archive.zip",
            sha256="15595c7b52896262a0e590f02b3b3851c6a2c9046a72047526deaf59a9cbf7d5",
        )

    def build(self):
        bin = self.install_dir / "bin"
        bin.mkdir(parents=True)
        for dll in DLLS:
            shutil.copy2(next(self.source_dir.rglob(dll)), bin)

    def package(self):
        runtime = Component("cudnn", shared_libs=[SharedLib(f"bin/{dll}") for dll in DLLS])
        return Package(components=[runtime], licenses=["LICENSE"])
