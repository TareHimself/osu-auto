using osu.Framework.Screens;
using osu.Game.Beatmaps;
using osu.Game.Rulesets.Mods;
using osu.Game.Rulesets.Scoring;
using osu.Game.Rulesets.UI;
using osu.Game.Scoring;
using osu.Game.Screens.Play;

namespace OsuAuto.Host;

public partial class AutoReplayPlayer : ReplayPlayer, IHostPlayer
{
    public Player Player => this;

    protected override bool PauseOnFocusLost => false;

    protected override bool CheckModsAllowFailure() => !lockstep && base.CheckModsAllowFailure();

    public new ScoreProcessor ScoreProcessor => base.ScoreProcessor;
    public new GameplayClockContainer GameplayClockContainer => base.GameplayClockContainer;
    public new DrawableRuleset DrawableRuleset => base.DrawableRuleset;
    public new Score Score => base.Score;

    protected override void LoadComplete()
    {
        base.LoadComplete();
        ReplayOverlay.Hide();
    }

    private readonly TaskCompletionSource<bool> entered = new TaskCompletionSource<bool>();

    public Task Entered => entered.Task;

    public override void OnEntering(ScreenTransitionEvent e)
    {
        base.OnEntering(e);
        entered.TrySetResult(true);
    }

    public void FinishEntryTransforms()
    {
        FinishTransforms(false);
        ApplyToBackground(b => b.FinishTransforms(true));
    }

    public void FinishBackgroundTransforms() => ApplyToBackground(b => b.FinishTransforms(true));

    private readonly bool lockstep;

    protected override GameplayClockContainer CreateGameplayClockContainer(WorkingBeatmap beatmap, double gameplayStart)
        => lockstep ? new LockstepGameplayClockContainer(beatmap, gameplayStart) : base.CreateGameplayClockContainer(beatmap, gameplayStart);

    public AutoReplayPlayer(Func<IBeatmap, IReadOnlyList<Mod>, Score> createScore, bool lockstep = false)
        : base(createScore, new PlayerConfiguration
        {
            ShowResults = false,
            AllowPause = false,
            AllowRestart = false,
            AllowSkipping = false,
        })
    {
        this.lockstep = lockstep;
    }
}
