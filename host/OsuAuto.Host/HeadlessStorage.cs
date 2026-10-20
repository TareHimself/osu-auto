namespace OsuAuto.Host;

public static class HeadlessStorage
{
    public static string DirectoryFor(string gameName) => Path.Combine(Directory.GetCurrentDirectory(), "headless", gameName);

    public static void EnsureSilentAudioDevice(string gameName)
    {
        writeNoSound(DirectoryFor(gameName));
        writeNoSound(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), gameName));
    }

    private static void writeNoSound(string dir)
    {
        Directory.CreateDirectory(dir);
        string ini = Path.Combine(dir, "framework.ini");

        var lines = File.Exists(ini)
            ? File.ReadAllLines(ini).Where(l => !l.TrimStart().StartsWith("AudioDevice", StringComparison.Ordinal)).ToList()
            : new List<string>();

        lines.Add("AudioDevice = No sound");
        File.WriteAllLines(ini, lines);
    }
}
