using osu.Game.Replays;
using osu.Game.Rulesets.Replays;
using osu.Game.Rulesets.Osu.Objects;
using osu.Game.Rulesets.Osu.Replays;
using osuTK;

namespace OsuAuto.Host.Export;

public readonly record struct SpinTrack(double StartMs, double EndMs, Vector2 Centre, float Radius, float Direction, float RadPerMs);

/// <summary>
/// Rewrites every spinner of an autoplay replay so it opens from a random state (stopped, crawling or reversing, at a random angle
/// and radius), then ramps to autoplay speed, sometimes faltering once more. Rendering this replay shows what a faltering spin
/// looks like; labels take the full-speed target from <see cref="Target"/>, never the faltering path.
/// </summary>
public static class SpinRecovery
{
    /// <summary>Every spin goes this way (decreasing atan2 angle in playfield coordinates). Autoplay picks a direction per spinner; we do not.</summary>
    public const float DIRECTION = -1;

    private const double frame_ms = 5;

    public static (Replay replay, List<SpinTrack> tracks) Apply(Replay auto, IEnumerable<OsuHitObject> objects, int seed)
    {
        var rng = new Random(seed);
        var frames = auto.Frames.OfType<OsuReplayFrame>().OrderBy(f => f.Time).ToList();
        var tracks = new List<SpinTrack>();

        foreach (var spinner in objects.OfType<Spinner>())
        {
            double s = spinner.StartTime, e = spinner.EndTime;
            Vector2 centre = spinner.StackedPosition;
            var inside = frames.Where(f => f.Time > s && f.Time < e).ToList();

            if (inside.Count < 4)
                continue;

            var (_, radPerMs, radius) = measure(inside, centre);
            float direction = DIRECTION;
            var actions = inside[inside.Count / 2].Actions.ToList();
            var entry = frames.LastOrDefault(f => f.Time <= s) ?? inside[0];

            frames.RemoveAll(f => f.Time > s && f.Time < e);

            double angle = rng.NextDouble() * Math.PI * 2;
            float r = (float)(15 + rng.NextDouble() * 55);
            double openUntil = s + rng.NextDouble() * 1500;
            double openSpeed = pick(rng, radPerMs);
            double falterAt = rng.NextDouble() < 0.5 ? s + (e - s) * (0.4 + 0.4 * rng.NextDouble()) : double.MaxValue;
            double falterUntil = falterAt + 150 + rng.NextDouble() * 600;
            double falterSpeed = pick(rng, radPerMs);
            double speed = openSpeed;

            var generated = new List<OsuReplayFrame> { new OsuReplayFrame(Math.Round(s + 1), entry.Position, actions.ToArray()) };

            for (double t = Math.Round(s + frame_ms); t < e; t += frame_ms)
            {
                double wanted = t < openUntil ? openSpeed : t >= falterAt && t < falterUntil ? falterSpeed : radPerMs;
                speed += (wanted - speed) * 0.08;
                angle += direction * speed * frame_ms;
                float rWanted = t < openUntil || (t >= falterAt && t < falterUntil) ? r : radius;
                r += (rWanted - r) * 0.05f;
                generated.Add(new OsuReplayFrame(t, centre + new Vector2(r * MathF.Cos((float)angle), r * MathF.Sin((float)angle)), actions.ToArray()));
            }

            frames.AddRange(generated);
            frames.Sort((a, b) => a.Time.CompareTo(b.Time));
            tracks.Add(new SpinTrack(s, e, centre, radius, direction, (float)radPerMs));
        }

        foreach (var f in frames)
            f.Time = Math.Round(f.Time);

        return (new Replay { Frames = frames.Cast<ReplayFrame>().ToList(), HasReceivedAllFrames = true }, tracks);
    }

    /// <summary>The aim target at a cursor position inside a spinner: one 10 ms full-speed step ahead, on autoplay's circle.</summary>
    public static Vector2 Target(SpinTrack track, Vector2 cursor, double stepMs)
    {
        Vector2 d = cursor - track.Centre;
        double angle = Math.Atan2(d.Y, d.X) + track.Direction * track.RadPerMs * stepMs;
        return track.Centre + new Vector2(track.Radius * (float)Math.Cos(angle), track.Radius * (float)Math.Sin(angle));
    }

    private static double pick(Random rng, double full)
    {
        double u = rng.NextDouble();
        return u < 0.4 ? 0 : u < 0.75 ? full * (0.1 + 0.5 * rng.NextDouble()) : -full * (0.1 + 0.3 * rng.NextDouble());
    }

    private static (float direction, double radPerMs, float radius) measure(List<OsuReplayFrame> inside, Vector2 centre)
    {
        var angles = inside.Select(f => Math.Atan2(f.Position.Y - centre.Y, f.Position.X - centre.X)).ToList();
        var rates = new List<double>();

        for (int i = 1; i < inside.Count; i++)
        {
            double dt = inside[i].Time - inside[i - 1].Time;
            if (dt <= 0)
                continue;

            double da = Math.IEEERemainder(angles[i] - angles[i - 1], Math.PI * 2);
            rates.Add(da / dt);
        }

        rates.Sort();
        double median = rates.Count > 0 ? rates[rates.Count / 2] : 0.05;
        var radii = inside.Select(f => (f.Position - centre).Length).OrderBy(v => v).ToList();
        return (median >= 0 ? 1f : -1f, Math.Abs(median), radii[radii.Count / 2]);
    }
}
