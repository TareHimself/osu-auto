from silo import SiloPackage
from silo.tools.cmake import CMakeBuild, CMakeDeps, CMakeGenerator, CMakePresets


class Runner(SiloPackage):
    name = "osu-auto-runner"
    version = "0.1.0"

    def source(self, src):
        src.local(".")

    def deps(self):
        self.dep("onnxruntime@1.31.0")
        self.dep("cudnn@9.27.0")
        self.dep("libcublas@13.8.0")
        self.tool("cmake@4.3")

    def generate(self):
        CMakeDeps(self).generate()
        toolchain = CMakeGenerator(self)
        toolchain.generate()
        name = "debug" if self.is_debug else "release"
        presets = CMakePresets(self, toolchain)
        presets.add(f"runner-{name}", build_dir=f"build/{name}")
        presets.generate()

    def build(self):
        cmake = CMakeBuild(self)
        cmake.configure()
        cmake.build()
        cmake.install()
