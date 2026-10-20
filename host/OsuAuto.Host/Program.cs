using System.Drawing;
using osu.Framework;
using osu.Framework.Platform;
using OsuAuto.Host;
using OsuAuto.Host.Commands;

if (args.Length == 0)
{
    Console.Error.WriteLine("usage: OsuAuto.Host <run-auto|export-labels|serve> [--window WxH] [--skin file.osk] [--instance N] [options]");
    return 2;
}

var options = CliOptions.Parse(args.Skip(1).ToArray());
bool realtime = options.Flag("realtime");

Func<AutoHostGame, Task> job = args[0] switch
{
    "run-auto" => game => RunAuto.Execute(game, options),
    "export-labels" => game => ExportLabels.Execute(game, options),
    "serve" => game => Serve.Execute(game, options),
    _ => throw new ArgumentException($"unknown command {args[0]}"),
};

Size? windowSize = null;

if (options.Get("window") is { } w)
{
    string[] parts = w.Split('x');
    windowSize = new Size(int.Parse(parts[0]), int.Parse(parts[1]));
}

int instance = int.Parse(options.Get("instance") ?? "0");
string name = instance == 0 ? "osu-auto" : $"osu-auto-{instance}";

HeadlessStorage.EnsureSilentAudioDevice(name);

using GameHost host = windowSize == null
    ? new HeadlessGameHost(name, new HostOptions(), realtime: realtime)
    : Host.GetSuitableDesktopHost(name, new HostOptions());

int updateHz = int.Parse(options.Get("update-hz") ?? "1000");
int drawHz = int.Parse(options.Get("draw-hz") ?? "1000");

using var game = new AutoHostGame(job, name, options.Get("skin"), windowSize, updateHz, drawHz);
host.Run(game);
return game.ExitCode;
