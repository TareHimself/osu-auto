from silo import SiloPackage


class RunnerShell(SiloPackage):
    """The project you run the runner from: its build environment has runner.exe on the PATH for silo run."""

    name = "runner"
    version = "0.1.0"

    def source(self, src):
        src.empty()

    def deps(self):
        self.tool("../runner")
