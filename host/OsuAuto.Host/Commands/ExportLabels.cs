using System.Security.Cryptography;
using System.Text.Json;
using osu.Game.Beatmaps;
using osu.Game.Replays;
using osu.Game.Rulesets.Mods;
using osu.Game.Rulesets.Objects;
using osu.Game.Rulesets.Osu;
using osu.Game.Rulesets.Osu.Objects;
using osu.Game.Rulesets.Osu.Replays;
using osu.Game.Scoring.Legacy;
using OsuAuto.Host.Export;

namespace OsuAuto.Host.Commands;

public static class ExportLabels
{
    private const double step_ms = 10;
    private const double lead_in_ms = 1500;

    public static async Task Execute(AutoHostGame game, CliOptions options)
    {
        string path = options.Positional.FirstOrDefault() ?? throw new ArgumentException("beatmap path required");
        string outRoot = options.Require("out");
        string? osrPath = options.Get("osr");
        var modSpecs = ModSpec.Split(options.Get("mods"));

        var ruleset = new OsuRuleset();
        var info = await game.ImportBeatmap(path, options.Get("diff")).ConfigureAwait(false);

        IReadOnlyList<Mod> mods;
        Replay replay;
        string source;

        if (osrPath != null)
        {
            if (modSpecs.Count > 0)
                throw new ArgumentException("--mods cannot be combined with --osr, the replay carries its own mods");

            var decoder = new DatabasedLegacyScoreDecoder(game.RulesetStore, game.BeatmapManager);
            using var stream = File.OpenRead(osrPath);
            var score = decoder.Parse(stream);
            mods = score.ScoreInfo.Mods;
            replay = score.Replay;
            source = "osr-" + Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(osrPath)))[..8].ToLowerInvariant();
        }
        else
        {
            mods = ModSpec.Parse(ruleset, modSpecs);
            replay = null!;
            source = "autoplay" + ModSpec.Tag(modSpecs);
        }

        var working = await game.SelectBeatmap(info, ruleset, mods).ConfigureAwait(false);
        var playable = working.GetPlayableBeatmap(ruleset.RulesetInfo, mods);

        replay ??= new OsuAutoGenerator(playable, mods).Generate();

        var objects = playable.HitObjects.OfType<OsuHitObject>().ToList();
        double t0 = objects[0].StartTime - lead_in_ms;
        double t1 = Math.Max(objects[^1].GetEndTime(), replay.Frames.Max(f => f.Time));

        var samples = ReplaySampler.Sample(replay, t0, t1, step_ms);

        string outDir = Path.Combine(outRoot, $"{info.MD5Hash}-{source}");
        Directory.CreateDirectory(outDir);

        writeLabels(Path.Combine(outDir, "labels.npz"), samples);
        writeObjects(Path.Combine(outDir, "objects.npz"), objects);

        var meta = new
        {
            beatmap_md5 = info.MD5Hash,
            title = info.GetDisplayTitle(),
            difficulty = info.DifficultyName,
            mods = mods.Select(m => m.Acronym).ToArray(),
            mod_specs = modSpecs,
            source,
            osr = osrPath == null ? null : Path.GetFileName(osrPath),
            t0_ms = t0,
            t1_ms = t1,
            step_ms,
            circle_size = playable.Difficulty.CircleSize,
            approach_rate = playable.Difficulty.ApproachRate,
            overall_difficulty = playable.Difficulty.OverallDifficulty,
            playfield = new { width = 512, height = 384 },
            samples = samples.Count,
            objects = objects.Count,
            replay_frames = replay.Frames.Count,
        };

        await File.WriteAllTextAsync(Path.Combine(outDir, "meta.json"), JsonSerializer.Serialize(meta, new JsonSerializerOptions { WriteIndented = true })).ConfigureAwait(false);

        Console.WriteLine($"exported {samples.Count} samples, {objects.Count} objects -> {outDir}");
    }

    private static void writeLabels(string path, List<LabelSample> samples)
    {
        int n = samples.Count;
        var t = new double[n];
        var x = new float[n];
        var y = new float[n];
        var k1 = new byte[n];
        var k2 = new byte[n];
        var press1 = new byte[n];
        var press2 = new byte[n];

        for (int i = 0; i < n; i++)
        {
            var s = samples[i];
            t[i] = s.TimeMs;
            x[i] = s.X;
            y[i] = s.Y;
            k1[i] = (byte)(s.K1 ? 1 : 0);
            k2[i] = (byte)(s.K2 ? 1 : 0);
            press1[i] = (byte)(s.K1 && (i == 0 || !samples[i - 1].K1) ? 1 : 0);
            press2[i] = (byte)(s.K2 && (i == 0 || !samples[i - 1].K2) ? 1 : 0);
        }

        using var npz = new Npz(path);
        npz.Add("t_ms", t);
        npz.Add("x", x);
        npz.Add("y", y);
        npz.Add("k1", k1);
        npz.Add("k2", k2);
        npz.Add("press_k1", press1);
        npz.Add("press_k2", press2);
    }

    private static void writeObjects(string path, List<OsuHitObject> objects)
    {
        int n = objects.Count;
        var start = new double[n];
        var end = new double[n];
        var x = new float[n];
        var y = new float[n];
        var endX = new float[n];
        var endY = new float[n];
        var kind = new byte[n];
        var radius = new float[n];
        var combo = new int[n];

        for (int i = 0; i < n; i++)
        {
            var h = objects[i];
            start[i] = h.StartTime;
            end[i] = h.GetEndTime();
            x[i] = h.StackedPosition.X;
            y[i] = h.StackedPosition.Y;
            endX[i] = h.StackedEndPosition.X;
            endY[i] = h.StackedEndPosition.Y;
            kind[i] = h switch
            {
                Slider => 1,
                Spinner => 2,
                _ => 0,
            };
            radius[i] = (float)h.Radius;
            combo[i] = h.ComboIndex;
        }

        using var npz = new Npz(path);
        npz.Add("start_ms", start);
        npz.Add("end_ms", end);
        npz.Add("x", x);
        npz.Add("y", y);
        npz.Add("end_x", endX);
        npz.Add("end_y", endY);
        npz.Add("kind", kind);
        npz.Add("radius", radius);
        npz.Add("combo", combo);
    }
}
