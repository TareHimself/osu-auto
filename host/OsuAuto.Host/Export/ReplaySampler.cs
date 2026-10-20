using osu.Framework.Utils;
using osu.Game.Replays;
using osu.Game.Rulesets.Osu;
using osu.Game.Rulesets.Osu.Replays;
using osuTK;

namespace OsuAuto.Host.Export;

public readonly record struct LabelSample(double TimeMs, float X, float Y, bool K1, bool K2);

public static class ReplaySampler
{
    public static List<LabelSample> Sample(Replay replay, double fromMs, double toMs, double stepMs)
    {
        var frames = replay.Frames.OfType<OsuReplayFrame>().OrderBy(f => f.Time).ToList();

        if (frames.Count == 0)
            throw new InvalidOperationException("replay has no osu! frames");

        var samples = new List<LabelSample>((int)((toMs - fromMs) / stepMs) + 1);
        int index = 0;

        for (double t = fromMs; t <= toMs; t += stepMs)
        {
            while (index + 1 < frames.Count && frames[index + 1].Time <= t)
                index++;

            OsuReplayFrame current = frames[index];
            Vector2 position;

            if (t < current.Time)
                position = current.Position;
            else if (index + 1 < frames.Count)
            {
                OsuReplayFrame next = frames[index + 1];
                position = Interpolation.ValueAt(t, current.Position, next.Position, current.Time, next.Time);
            }
            else
                position = current.Position;

            bool k1 = t >= current.Time && current.Actions.Contains(OsuAction.LeftButton);
            bool k2 = t >= current.Time && current.Actions.Contains(OsuAction.RightButton);

            samples.Add(new LabelSample(t, position.X, position.Y, k1, k2));
        }

        return samples;
    }
}
