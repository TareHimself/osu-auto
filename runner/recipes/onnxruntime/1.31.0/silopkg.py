import shutil

from silo import Component, Package, SiloPackage

PROVIDER_DLLS = ("onnxruntime_providers_shared.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_tensorrt.dll")


class OnnxRuntime(SiloPackage):
    """Microsoft's prebuilt Windows x64 CUDA 13 release."""

    name = "onnxruntime"
    version = "1.31.0"
    ignore_settings = ("compiler", "build_type", "cppstd", "libcxx")

    def source(self, src):
        src.fetch(
            "https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-x64-gpu_cuda13-1.31.0.zip",
            sha256="d332ccf84acb769a11364cddd4e8d2992a168d16be472d5ba70b279b644552f5",
        )

    def build(self):
        shutil.copytree(self.source_dir / "include", self.install_dir / "include")
        lib, bin = self.install_dir / "lib", self.install_dir / "bin"
        lib.mkdir(parents=True)
        bin.mkdir()
        shutil.copy2(self.source_dir / "lib" / "onnxruntime.lib", lib)
        for dll in ("onnxruntime.dll", *PROVIDER_DLLS):
            shutil.copy2(self.source_dir / "lib" / dll, bin)

    def package(self):
        runtime = Component("onnxruntime", headers=["include"], shared_libs=["onnxruntime"], content=["bin"])
        return Package(components=[runtime], licenses=["LICENSE"])
