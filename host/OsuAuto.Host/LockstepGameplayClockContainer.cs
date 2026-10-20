using osu.Game.Beatmaps;
using osu.Game.Screens.Play;

namespace OsuAuto.Host;

public partial class LockstepGameplayClockContainer : MasterGameplayClockContainer
{
    public LockstepGameplayClockContainer(WorkingBeatmap beatmap, double gameplayStartTime)
        : base(beatmap, gameplayStartTime)
    {
    }

    protected override void StartGameplayClock()
    {
    }
}
