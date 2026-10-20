using osu.Game.Rulesets.Scoring;
using osu.Game.Rulesets.UI;
using osu.Game.Scoring;
using osu.Game.Screens.Play;

namespace OsuAuto.Host;

public interface IHostPlayer
{
    Player Player { get; }
    ScoreProcessor ScoreProcessor { get; }
    GameplayClockContainer GameplayClockContainer { get; }
    DrawableRuleset DrawableRuleset { get; }
    Score Score { get; }
    Task Entered { get; }

    void FinishEntryTransforms();
    void FinishBackgroundTransforms();
}
