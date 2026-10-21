from silo import SiloPackage


class RunnerShell(SiloPackage):
    """A project whose build environment has the runner on its PATH, for silo run."""

    name = "runner-shell"
    version = "0.1.0"

    def source(self, src):
        src.empty()

    def deps(self):
        self.tool("../runner")
