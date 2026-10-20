using osu.Framework.Screens;
using osu.Game.Beatmaps;
using osu.Game.Rulesets.Scoring;
using osu.Game.Rulesets.UI;
using osu.Game.Scoring;
using osu.Game.Screens.Play;
using osu.Game.Screens.Ranking;

namespace OsuAuto.Host;

/// <summary>
/// A regular (non-replay) player driven by <see cref="SyntheticInputManager"/>, so gameplay mods such as Relax and Autopilot behave as in real play.
/// </summary>
public partial class AutoPlayer : Player, IHostPlayer
{
    protected override bool PauseOnFocusLost => false;

    protected override bool CheckModsAllowFailure() => false;

    protected override ResultsScreen CreateResults(ScoreInfo score) => new SoloResultsScreen(score);

    protected override GameplayClockContainer CreateGameplayClockContainer(WorkingBeatmap beatmap, double gameplayStart)
        => new LockstepGameplayClockContainer(beatmap, gameplayStart);

    public Player Player => this;
    public new ScoreProcessor ScoreProcessor => base.ScoreProcessor;
    public new GameplayClockContainer GameplayClockContainer => base.GameplayClockContainer;
    public new DrawableRuleset DrawableRuleset => base.DrawableRuleset;
    public new Score Score => base.Score;

    public AutoPlayer()
        : base(new PlayerConfiguration
        {
            ShowResults = false,
            AllowPause = false,
            AllowRestart = false,
            AllowSkipping = false,
        })
    {
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
}
