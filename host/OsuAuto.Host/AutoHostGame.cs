using System.Drawing;
using System.Reflection;
using osu.Framework.Allocation;
using osu.Framework.Configuration;
using osu.Framework.Graphics;
using osu.Framework.Input;
using osu.Framework.Platform;
using osu.Framework.Logging;
using osu.Game;
using osu.Game.Beatmaps;
using osu.Game.Configuration;
using osu.Game.Database;
using osu.Game.Graphics.Cursor;
using osu.Game.Online.API;
using osu.Game.Rulesets;
using osu.Game.Rulesets.Mods;
using osu.Game.Rulesets.Osu;
using osu.Game.Rulesets.Osu.Configuration;
using osu.Game.Scoring;
using osu.Game.Screens;
using osu.Game.Skinning;

namespace OsuAuto.Host;

public partial class AutoHostGame : OsuGameBase
{
    private readonly Func<AutoHostGame, Task> job;
    private readonly DummyAPIAccess dummyApi = new DummyAPIAccess();
    private OsuScreenStack screenStack = null!;

    public int ExitCode { get; private set; }

    public override bool UseDevelopmentServer => false;

    public new BeatmapManager BeatmapManager => base.BeatmapManager;
    public new ScoreManager ScoreManager => base.ScoreManager;
    public new RealmRulesetStore RulesetStore => base.RulesetStore;

    private readonly string name;
    private readonly string? skinPath;
    private readonly Size? windowSize;

    public new SkinManager SkinManager => base.SkinManager;

    private readonly int updateHz;
    private readonly int drawHz;

    public AutoHostGame(Func<AutoHostGame, Task> job, string name, string? skinPath = null, Size? windowSize = null, int updateHz = 1000, int drawHz = 1000)
    {
        this.job = job;
        this.name = name;
        this.skinPath = skinPath;
        this.windowSize = windowSize;
        this.updateHz = updateHz;
        this.drawHz = drawHz;
        API = dummyApi;
        Storage = new NativeStorage(HeadlessStorage.DirectoryFor(name));
    }

    [BackgroundDependencyLoader]
    private void load(FrameworkConfigManager frameworkConfig)
    {
        frameworkConfig.SetValue(FrameworkSetting.AudioDevice, "No sound");
        Audio.Volume.Value = 0;
        Audio.VolumeTrack.Value = 0;
        Audio.VolumeSample.Value = 0;
        Console.WriteLine($"audio device: '{Audio.AudioDevice.Value}' volume={Audio.Volume.Value}");

        LocalConfig.SetValue(OsuSetting.BeatmapSkins, false);
        LocalConfig.SetValue(OsuSetting.BeatmapColours, false);
        LocalConfig.SetValue(OsuSetting.BeatmapHitsounds, false);
        LocalConfig.SetValue(OsuSetting.DimLevel, 1.0);
        LocalConfig.SetValue(OsuSetting.BlurLevel, 0.0);
        LocalConfig.SetValue(OsuSetting.LightenDuringBreaks, false);
        LocalConfig.SetValue(OsuSetting.ShowStoryboard, false);
        LocalConfig.SetValue(OsuSetting.KeyOverlay, false);
        LocalConfig.SetValue(OsuSetting.HUDVisibilityMode, HUDVisibilityMode.Never);
        LocalConfig.SetValue(OsuSetting.ShowHealthDisplayWhenCantFail, false);
        LocalConfig.SetValue(OsuSetting.FadePlayfieldWhenHealthLow, false);
        LocalConfig.SetValue(OsuSetting.HitLighting, false);
        LocalConfig.SetValue(OsuSetting.ReplaySettingsOverlay, false);
        LocalConfig.SetValue(OsuSetting.ReplayPlaybackControlsExpanded, false);
        LocalConfig.SetValue(OsuSetting.Scaling, ScalingMode.Off);
        LocalConfig.SetValue(OsuSetting.ShowFpsDisplay, false);
        LocalConfig.SetValue(OsuSetting.CursorRotation, false);
        LocalConfig.SetValue(OsuSetting.AutoCursorSize, false);
        LocalConfig.SetValue(OsuSetting.GameplayCursorSize, 1.0f);
        LocalConfig.SetValue(OsuSetting.GameplayDisableWinKey, false);

        frameworkConfig.SetValue(FrameworkSetting.ConfineMouseMode, ConfineMouseMode.Never);

        if (windowSize is { } size)
        {
            frameworkConfig.SetValue(FrameworkSetting.WindowMode, WindowMode.Windowed);
            frameworkConfig.SetValue(FrameworkSetting.WindowedSize, size);
            frameworkConfig.SetValue(FrameworkSetting.FrameSync, FrameSync.Unlimited);

            Host.MaximumDrawHz = drawHz;
            Host.MaximumUpdateHz = updateHz;
            Host.MaximumInactiveHz = updateHz;

            if (Host.Window != null)
                Host.Window.Title = name;
        }

        hideMenuCursor();

        Add(dummyApi);
        Add(screenStack = new OsuScreenStack { RelativeSizeAxes = Axes.Both });
    }

    protected override void LoadComplete()
    {
        base.LoadComplete();

        Task.Run(async () =>
        {
            try
            {
                await applyOsuRulesetSettings().ConfigureAwait(false);

                if (skinPath != null)
                    await ApplySkin(skinPath).ConfigureAwait(false);

                await job(this).ConfigureAwait(false);
            }
            catch (Exception e)
            {
                Logger.Error(e, "job failed");
                Console.Error.WriteLine(e);
                ExitCode = 1;
            }

            Schedule(Exit);
        });
    }

    private async Task applyOsuRulesetSettings()
    {
        for (int attempt = 0; attempt < 200; attempt++)
        {
            try
            {
                await OnUpdateThread(() =>
                {
                    if (Dependencies.Get<IRulesetConfigCache>().GetConfigFor(new OsuRuleset()) is OsuRulesetConfigManager osuConfig)
                    {
                        osuConfig.SetValue(OsuRulesetSetting.ShowCursorTrail, false);
                        osuConfig.SetValue(OsuRulesetSetting.ShowCursorRipples, false);
                    }
                }).ConfigureAwait(false);

                return;
            }
            catch (InvalidOperationException)
            {
                await Task.Delay(50).ConfigureAwait(false);
            }
        }

        throw new TimeoutException("ruleset config cache never became available");
    }

    public Task<T> OnUpdateThread<T>(Func<T> func)
    {
        var tcs = new TaskCompletionSource<T>();

        Schedule(() =>
        {
            try
            {
                tcs.SetResult(func());
            }
            catch (Exception e)
            {
                tcs.SetException(e);
            }
        });

        return tcs.Task;
    }

    public Task OnUpdateThread(Action action) => OnUpdateThread(() =>
    {
        action();
        return true;
    });

    private string tempImportCopy(string path)
    {
        string tempDir = Path.Combine(HeadlessStorage.DirectoryFor(name), "import");
        Directory.CreateDirectory(tempDir);
        string tempCopy = Path.Combine(tempDir, $"{Guid.NewGuid():N}{Path.GetExtension(path)}");
        File.Copy(path, tempCopy, true);
        return tempCopy;
    }

    public async Task ApplySkin(string path)
    {
        string tempCopy = tempImportCopy(path);
        var imported = await SkinManager.Import(new ImportTask(tempCopy)).ConfigureAwait(false)
                       ?? throw new InvalidOperationException($"skin import failed for {path}");

        if (File.Exists(tempCopy))
            File.Delete(tempCopy);

        await OnUpdateThread(() => SkinManager.CurrentSkinInfo.Value = imported).ConfigureAwait(false);
        Console.WriteLine($"skin: {imported.PerformRead(s => s.Name)}");
    }

    public async Task<BeatmapInfo> ImportBeatmap(string path, string? difficulty)
    {
        string tempCopy = tempImportCopy(path);

        var imported = await BeatmapManager.Import(new ImportTask(tempCopy)).ConfigureAwait(false)
                       ?? throw new InvalidOperationException($"import failed for {path}");

        if (File.Exists(tempCopy))
            File.Delete(tempCopy);

        return imported.PerformRead(set =>
        {
            var candidates = set.Beatmaps.Where(b => b.Ruleset.ShortName == "osu").ToList();

            if (candidates.Count == 0)
                throw new InvalidOperationException("no osu! difficulties in beatmap set");

            var chosen = difficulty == null
                ? candidates.OrderBy(b => b.StarRating).First()
                : candidates.FirstOrDefault(b => string.Equals(b.DifficultyName, difficulty, StringComparison.OrdinalIgnoreCase))
                  ?? throw new InvalidOperationException($"difficulty '{difficulty}' not found; have: {string.Join(", ", candidates.Select(b => b.DifficultyName))}");

            return chosen.Detach();
        });
    }

    public Task<WorkingBeatmap> SelectBeatmap(BeatmapInfo info, Ruleset ruleset, IReadOnlyList<Mod> mods) => OnUpdateThread(() =>
    {
        var working = BeatmapManager.GetWorkingBeatmap(info);
        Ruleset.Value = ruleset.RulesetInfo;
        Beatmap.Value = working;
        SelectedMods.Value = mods;
        return working;
    });

    public Task PushScreen(OsuScreen screen) => OnUpdateThread(() => screenStack.Push(screen));

    private void hideMenuCursor()
    {
        var field = typeof(GlobalCursorDisplay).GetField("ShowCursor", BindingFlags.Instance | BindingFlags.NonPublic)
                    ?? throw new InvalidOperationException("GlobalCursorDisplay.ShowCursor not found; lazer changed");
        field.SetValue(GlobalCursorDisplay, false);
    }

    public new GameHost Host => base.Host;

    public SyntheticInputManager SyntheticInput { get; private set; } = null!;

    protected override UserInputManager CreateUserInputManager() => SyntheticInput = new SyntheticInputManager();

    public Task<T> OnDrawThread<T>(Func<T> func)
    {
        var tcs = new TaskCompletionSource<T>();

        Host.DrawThread.Scheduler.Add(() =>
        {
            try
            {
                tcs.SetResult(func());
            }
            catch (Exception e)
            {
                tcs.SetException(e);
            }
        });

        return tcs.Task;
    }

    /// <summary>
    /// Waits until the renderer has drawn <paramref name="frames"/> frames that can include the current update state.
    /// The draw thread skips rendering when no new draw tree is ready, so scheduler ticks alone do not count.
    /// </summary>
    public async Task WaitForDraws(int frames)
    {
        await OnUpdateThread(() => true).ConfigureAwait(false);
        ulong start = await OnDrawThread(() => Host.Renderer.FrameIndex).ConfigureAwait(false);
        int polls = 0;

        while (await OnDrawThread(() => Host.Renderer.FrameIndex).ConfigureAwait(false) < start + (ulong)frames)
        {
            if (++polls > 1_000_000)
                throw new TimeoutException("draw thread stopped rendering");
        }
    }
}
