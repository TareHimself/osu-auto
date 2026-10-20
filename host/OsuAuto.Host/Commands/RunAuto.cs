using System.Diagnostics;
using osu.Game.Beatmaps;
using osu.Game.Rulesets.Mods;
using osu.Game.Rulesets.Objects;
using osu.Game.Rulesets.Osu;

namespace OsuAuto.Host.Commands;

public static class RunAuto
{
    public static async Task Execute(AutoHostGame game, CliOptions options)
    {
        string path = options.Positional.FirstOrDefault() ?? throw new ArgumentException("beatmap path required");

        var ruleset = new OsuRuleset();
        var autoplay = ruleset.GetAutoplayMod() ?? throw new InvalidOperationException("no autoplay mod");
        var mods = new Mod[] { autoplay };

        var info = await game.ImportBeatmap(path, options.Get("diff")).ConfigureAwait(false);
        var working = await game.SelectBeatmap(info, ruleset, mods).ConfigureAwait(false);
        var playable = working.GetPlayableBeatmap(ruleset.RulesetInfo, mods);
        double mapLengthMs = playable.HitObjects[^1].GetEndTime() - playable.HitObjects[0].StartTime;

        Console.WriteLine($"beatmap: {info.GetDisplayTitle()} [{info.DifficultyName}] objects={playable.HitObjects.Count} length={mapLengthMs / 1000:F1}s");

        var completed = new TaskCompletionSource<bool>();
        var wall = Stopwatch.StartNew();
        AutoReplayPlayer player = null!;

        await game.OnUpdateThread(() =>
        {
            player = new AutoReplayPlayer((beatmap, m) => autoplay.CreateScoreFromReplayData(beatmap, m));
            player.OnLoadComplete += _ =>
            {
                player.ScoreProcessor.HasCompleted.BindValueChanged(e =>
                {
                    if (e.NewValue)
                        completed.TrySetResult(true);
                });
            };
        }).ConfigureAwait(false);

        await game.PushScreen(player).ConfigureAwait(false);

        var timeout = Task.Delay(TimeSpan.FromMilliseconds(mapLengthMs * 3 + 60_000));
        if (await Task.WhenAny(completed.Task, timeout).ConfigureAwait(false) == timeout)
            throw new TimeoutException("gameplay did not complete");

        wall.Stop();

        string summary = await game.OnUpdateThread(() =>
        {
            var score = player.Score.ScoreInfo;
            player.ScoreProcessor.PopulateScore(score);
            string stats = string.Join(" ", score.Statistics.Where(s => s.Value > 0).OrderByDescending(s => s.Key).Select(s => $"{s.Key}={s.Value}"));
            return $"score={score.TotalScore} accuracy={score.Accuracy:P2} maxCombo={score.MaxCombo} stats=[{stats}] gameplayClock={player.GameplayClockContainer.CurrentTime:F0}ms";
        }).ConfigureAwait(false);

        Console.WriteLine(summary);
        Console.WriteLine($"wall={wall.Elapsed.TotalSeconds:F1}s map={mapLengthMs / 1000:F1}s speedup={mapLengthMs / wall.Elapsed.TotalMilliseconds:F2}x");
    }
}
