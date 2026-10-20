using MessagePack;
using MessagePack.Resolvers;
using NetMQ;
using NetMQ.Sockets;
using osu.Framework.Graphics;
using osu.Framework.Graphics.Shapes;
using osu.Framework.Screens;
using osu.Game.Beatmaps;
using osu.Game.Rulesets.Mods;
using osu.Game.Rulesets.Objects;
using osu.Game.Online.API.Requests.Responses;
using osu.Game.Replays;
using osu.Game.Input.Handlers;
using osu.Game.Rulesets.Osu;
using osu.Game.Rulesets.Osu.Objects;
using osu.Game.Rulesets.Osu.Replays;
using osu.Game.Rulesets.Osu.UI;
using osu.Game.Rulesets.UI;
using osu.Game.Scoring;
using osu.Game.Scoring.Legacy;
using osu.Game.Screens.Play;
using osuTK;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OsuAuto.Host.Commands;

public static class Serve
{
    private static readonly MessagePackSerializerOptions msgpack = ContractlessStandardResolver.Options;

    public static async Task Execute(AutoHostGame game, CliOptions options)
    {
        int port = int.Parse(options.Get("port") ?? (5555 + int.Parse(options.Get("instance") ?? "0")).ToString());
        using var socket = new ResponseSocket($"@tcp://127.0.0.1:{port}");
        Console.WriteLine($"serving on tcp://127.0.0.1:{port}");

        var session = new Session(game);

        while (true)
        {
            byte[] raw = socket.ReceiveFrameBytes();
            Dictionary<string, object> reply;

            try
            {
                var request = MessagePackSerializer.Deserialize<Dictionary<string, object>>(raw, msgpack);
                string cmd = (string)request["cmd"];

                if (cmd == "quit")
                {
                    socket.SendFrame(MessagePackSerializer.Serialize(new Dictionary<string, object> { ["ok"] = true }, msgpack));
                    return;
                }

                reply = await session.Handle(cmd, request).ConfigureAwait(false);
            }
            catch (Exception e)
            {
                reply = new Dictionary<string, object> { ["ok"] = false, ["error"] = e.ToString() };
            }

            socket.SendFrame(MessagePackSerializer.Serialize(reply, msgpack));
        }
    }

    private sealed class Session
    {
        private readonly AutoHostGame game;
        private readonly OsuRuleset ruleset = new OsuRuleset();
        private IHostPlayer? player;
        private bool userInput;
        private double t0, t1;

        public Session(AutoHostGame game)
        {
            this.game = game;
        }

        private bool agentDriven;
        private bool autoClicks;
        private bool relaxActive;
        private List<OsuAction> lastAppliedActions = new List<OsuAction>();
        private List<OsuReplayFrame> clickSource = new List<OsuReplayFrame>();
        private int clickIndex;
        private readonly Dictionary<string, int> hitsSinceStep = new Dictionary<string, int>();

        public Task<Dictionary<string, object>> Handle(string cmd, Dictionary<string, object> request) => cmd switch
        {
            "load" => load(request),
            "seek" => seekAndMaybeGrab(request),
            "grab" => grab(request.TryGetValue("pair", out object? pr) && Convert.ToBoolean(pr)),
            "step" => step(request),
            "cursor" => cursor(Convert.ToBoolean(request["visible"]), !request.TryGetValue("wait", out object? w) || Convert.ToBoolean(w)),
            "backdrop" => backdrop(Convert.ToBoolean(request["visible"])),
            "export_frames" => exportFrames(request),
            "status" => status(),
            _ => throw new ArgumentException($"unknown cmd {cmd}"),
        };

        private async Task<Dictionary<string, object>> seekAndMaybeGrab(Dictionary<string, object> request)
        {
            var reply = await seek(Convert.ToDouble(request["t"])).ConfigureAwait(false);

            if (request.TryGetValue("grab", out object? g) && Convert.ToBoolean(g))
            {
                foreach (var kv in await grab(request.TryGetValue("pair", out object? pr) && Convert.ToBoolean(pr)).ConfigureAwait(false))
                    reply[kv.Key] = kv.Value;
            }

            return reply;
        }

        private async Task<Dictionary<string, object>> grab(bool pair)
        {
            if (player == null)
                throw new InvalidOperationException("no beatmap loaded");

            var (pixels, width, height) = await readPlayBox().ConfigureAwait(false);
            var reply = new Dictionary<string, object> { ["ok"] = true, ["pixels"] = pixels, ["width"] = width, ["height"] = height };

            if (pair)
            {
                phaseWatch.Restart();
                await cursor(false, true).ConfigureAwait(false);
                reply["pixels_nocursor"] = (await readPlayBox().ConfigureAwait(false)).pixels;
                await cursor(true, false).ConfigureAwait(false);
                phase(3);
            }

            reportPhases();
            return reply;
        }

        private readonly System.Diagnostics.Stopwatch phaseWatch = new System.Diagnostics.Stopwatch();
        private readonly double[] phaseTotals = new double[4];
        private int phaseCount;

        private void phase(int index)
        {
            phaseTotals[index] += phaseWatch.Elapsed.TotalMilliseconds;
            phaseWatch.Restart();
        }

        private void reportPhases()
        {
            if (++phaseCount % 500 != 0)
                return;

            Console.WriteLine($"timing per seek (ms): seek+catchup={phaseTotals[0] / phaseCount:F2} draws={phaseTotals[1] / phaseCount:F2} screenshot={phaseTotals[2] / phaseCount:F2} post={phaseTotals[3] / phaseCount:F2}");
        }

        private async Task<(byte[] pixels, int width, int height)> readPlayBox()
        {
            phaseWatch.Restart();
            using Image<Rgba32> image = await game.Host.TakeScreenshotAsync().ConfigureAwait(false);
            phase(2);

            int boxW = Math.Min(image.Width, image.Height * 4 / 3);
            int boxH = boxW * 3 / 4;
            int x0 = (image.Width - boxW) / 2;
            int y0 = (image.Height - boxH) / 2;
            byte[] bytes = new byte[boxW * boxH * 4];

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < boxH; y++)
                {
                    var row = accessor.GetRowSpan(y0 + y);
                    int o = y * boxW * 4;

                    for (int x = 0; x < boxW; x++, o += 4)
                    {
                        Rgba32 px = row[x0 + x];
                        bytes[o] = px.B;
                        bytes[o + 1] = px.G;
                        bytes[o + 2] = px.R;
                        bytes[o + 3] = px.A;
                    }
                }
            });

            return (bytes, boxW, boxH);
        }

        private Box? backdropBox;

        private async Task<Dictionary<string, object>> backdrop(bool visible)
        {
            var p = player ?? throw new InvalidOperationException("no beatmap loaded");

            await game.OnUpdateThread(() =>
            {
                if (visible && backdropBox == null)
                    p.DrawableRuleset.PlayfieldAdjustmentContainer.Add(backdropBox = new Box
                    {
                        RelativeSizeAxes = Axes.Both,
                        Size = new Vector2(3),
                        Anchor = Anchor.Centre,
                        Origin = Anchor.Centre,
                        Colour = Colour4.White,
                        Depth = 1,
                    });
                else if (!visible && backdropBox != null)
                {
                    backdropBox.Expire();
                    backdropBox = null;
                }
            }).ConfigureAwait(false);

            await game.WaitForDraws(2).ConfigureAwait(false);
            return new Dictionary<string, object> { ["ok"] = true, ["visible"] = visible };
        }

        private async Task<Dictionary<string, object>> exportFrames(Dictionary<string, object> request)
        {
            if (player == null)
                throw new InvalidOperationException("no beatmap loaded");

            double[] times = ((object[])request["t"]).Select(Convert.ToDouble).ToArray();
            int w = Convert.ToInt32(request["width"]);
            int h = Convert.ToInt32(request["height"]);
            byte[] frames = new byte[times.Length * w * h];
            double[] clocks = new double[times.Length];

            for (int i = 0; i < times.Length; i++)
            {
                clocks[i] = (double)(await seek(times[i]).ConfigureAwait(false))["clock_ms"];
                await readGrayBox(frames.AsMemory(i * w * h, w * h), w, h).ConfigureAwait(false);
                reportPhases();
            }

            return new Dictionary<string, object> { ["ok"] = true, ["frames"] = frames, ["clocks"] = clocks, ["width"] = w, ["height"] = h };
        }

        private async Task readGrayBox(Memory<byte> dest, int w, int h)
        {
            phaseWatch.Restart();
            using Image<Rgba32> image = await game.Host.TakeScreenshotAsync().ConfigureAwait(false);
            phase(2);

            int boxW = Math.Min(image.Width, image.Height * 4 / 3);
            int boxH = boxW * 3 / 4;
            int x0 = (image.Width - boxW) / 2;
            int y0 = (image.Height - boxH) / 2;

            if (boxW % w != 0 || boxH % h != 0 || boxW / w != boxH / h)
                throw new ArgumentException($"play box {boxW}x{boxH} is not an integer multiple of {w}x{h}");

            int f = boxW / w;
            int area = f * f;
            int[] sums = new int[w * h];

            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < boxH; y++)
                {
                    var row = accessor.GetRowSpan(y0 + y);
                    int o = (y / f) * w;

                    for (int x = 0; x < boxW; x++)
                    {
                        Rgba32 px = row[x0 + x];
                        sums[o + x / f] += (px.R * 4899 + px.G * 9617 + px.B * 1868 + 8192) >> 14;
                    }
                }
            });

            var span = dest.Span;
            for (int i = 0; i < sums.Length; i++)
                span[i] = (byte)((sums[i] + area / 2) / area);
            phase(3);
        }

        private async Task<Dictionary<string, object>> cursor(bool visible, bool wait)
        {
            var p = player ?? throw new InvalidOperationException("no beatmap loaded");

            await game.OnUpdateThread(() =>
            {
                var c = p.DrawableRuleset.Cursor ?? throw new InvalidOperationException("ruleset has no cursor");
                c.Alpha = visible ? 1 : 0;
            }).ConfigureAwait(false);

            if (wait)
                await game.WaitForDraws(2).ConfigureAwait(false);

            return new Dictionary<string, object> { ["ok"] = true, ["visible"] = visible };
        }

        private async Task<Dictionary<string, object>> step(Dictionary<string, object> request)
        {
            var p = player ?? throw new InvalidOperationException("no beatmap loaded");

            if (!agentDriven)
                throw new InvalidOperationException("load with agent=true to drive frames");

            double t = Convert.ToDouble(request["t"]);
            float x = Convert.ToSingle(request["x"]);
            float y = Convert.ToSingle(request["y"]);

            var actions = new List<OsuAction>();

            if (autoClicks)
            {
                while (clickIndex + 1 < clickSource.Count && clickSource[clickIndex + 1].Time <= t)
                    clickIndex++;

                if (clickSource.Count > 0 && t >= clickSource[clickIndex].Time)
                    actions.AddRange(clickSource[clickIndex].Actions);
            }
            else
            {
                if (Convert.ToBoolean(request["k1"])) actions.Add(OsuAction.LeftButton);
                if (Convert.ToBoolean(request["k2"])) actions.Add(OsuAction.RightButton);
            }

            if (userInput)
            {
                await game.OnUpdateThread(() =>
                {
                    Vector2 screen = p.DrawableRuleset.Playfield.GamefieldToScreenSpace(new Vector2(x, y));
                    game.SyntheticInput.Set(screen, false, false);
                }).ConfigureAwait(false);
            }
            else
            {
                await game.OnUpdateThread(() =>
                {
                    var frames = p.Score.Replay.Frames;

                    if (frames.Count > 0 && frames[^1].Time > t)
                        throw new InvalidOperationException($"frame time {t} is before last frame {frames[^1].Time}");

                    frames.Add(new OsuReplayFrame(t, new Vector2(x, y), actions.ToArray()));
                }).ConfigureAwait(false);
            }

            var reply = await seekAndMaybeGrab(request).ConfigureAwait(false);

            if (request.TryGetValue("obs", out object? obsSize) && obsSize is object[] wh)
            {
                int w = Convert.ToInt32(wh[0]);
                int h = Convert.ToInt32(wh[1]);
                byte[] gray = new byte[w * h];
                await readGrayBox(gray, w, h).ConfigureAwait(false);
                reply["obs"] = gray;
            }

            if (userInput && !relaxActive && !actions.SequenceEqual(lastAppliedActions))
            {
                await game.OnUpdateThread(() =>
                {
                    var inputManager = ((DrawableOsuRuleset)p.DrawableRuleset).KeyBindingInputManager;
                    new ReplayInputHandler.ReplayState<OsuAction> { PressedActions = actions }.Apply(inputManager.CurrentState, inputManager);
                }).ConfigureAwait(false);
                lastAppliedActions = actions;
            }

            var snapshot = await status().ConfigureAwait(false);

            foreach (var kv in snapshot)
                reply[kv.Key] = kv.Value;

            reply["hits"] = new Dictionary<string, int>(hitsSinceStep);
            reply["actions"] = actions.Select(a => a.ToString()).ToArray();
            hitsSinceStep.Clear();
            return reply;
        }

        private async Task<Dictionary<string, object>> load(Dictionary<string, object> request)
        {
            string map = (string)request["map"];
            string? diff = request.TryGetValue("diff", out object? d) ? d as string : null;
            string? osr = request.TryGetValue("osr", out object? o) ? o as string : null;
            agentDriven = request.TryGetValue("agent", out object? a) && Convert.ToBoolean(a);
            autoClicks = request.TryGetValue("clicks", out object? c) && (c as string) == "auto";
            userInput = agentDriven && request.TryGetValue("input", out object? inp) && (inp as string) == "user";
            clickIndex = 0;
            hitsSinceStep.Clear();

            var info = await game.ImportBeatmap(map, diff).ConfigureAwait(false);

            IReadOnlyList<Mod> mods;
            Score? loadedScore = null;
            ModAutoplay? autoplay = null;
            var requested = ModSpec.Parse(ruleset, request.TryGetValue("mods", out object? m) && m is object[] list ? list.Select(o => (string)o) : Array.Empty<string>());

            if (osr != null)
            {
                using var stream = File.OpenRead(osr);
                loadedScore = new DatabasedLegacyScoreDecoder(game.RulesetStore, game.BeatmapManager).Parse(stream);
                mods = loadedScore.ScoreInfo.Mods;
            }
            else if (agentDriven)
            {
                mods = requested;
            }
            else
            {
                autoplay = ruleset.GetAutoplayMod() ?? throw new InvalidOperationException("no autoplay mod");
                mods = requested.Append(autoplay).ToList();
            }

            var working = await game.SelectBeatmap(info, ruleset, mods).ConfigureAwait(false);
            var playable = working.GetPlayableBeatmap(ruleset.RulesetInfo, mods);
            relaxActive = mods.Any(m => m is ModRelax);
            lastAppliedActions = new List<OsuAction>();

            if (agentDriven)
                clickSource = autoClicks ? new OsuAutoGenerator(playable, mods).Generate().Frames.OfType<OsuReplayFrame>().OrderBy(f => f.Time).ToList() : new List<OsuReplayFrame>();

            var loaded = new TaskCompletionSource<bool>();

            await game.OnUpdateThread(() =>
            {
                if (player != null)
                {
                    player.Player.Exit();
                    player.Player.FinishTransforms(false);
                }

                game.SyntheticInput.Deactivate();

                if (userInput)
                {
                    player = new AutoPlayer();
                }
                else
                {
                    player = new AutoReplayPlayer((beatmap, m) =>
                    {
                        if (!agentDriven)
                            return loadedScore ?? autoplay!.CreateScoreFromReplayData(beatmap, m);

                        return new Score
                        {
                            ScoreInfo = new ScoreInfo { User = new APIUser { Username = "agent" }, Mods = m.ToArray() },
                            Replay = new Replay { HasReceivedAllFrames = false },
                        };
                    }, lockstep: true);
                }

                player.Player.OnLoadComplete += _ =>
                {
                    player.ScoreProcessor.NewJudgement += j =>
                    {
                        string key = j.Type.ToString();
                        hitsSinceStep[key] = hitsSinceStep.GetValueOrDefault(key) + 1;
                    };
                    loaded.TrySetResult(true);
                };
            }).ConfigureAwait(false);

            await game.PushScreen(player!.Player).ConfigureAwait(false);
            await loaded.Task.ConfigureAwait(false);
            await player.Entered.ConfigureAwait(false);

            await game.OnUpdateThread(() =>
            {
                ((MasterGameplayClockContainer)player!.GameplayClockContainer).StopUsingBeatmapClock();
                player.GameplayClockContainer.Start();
                player.FinishEntryTransforms();
                player.GameplayClockContainer.FinishTransforms(false);

                if (player.DrawableRuleset.Cursor is { } gameplayCursor)
                {
                    gameplayCursor.Show();
                    gameplayCursor.FinishTransforms(true);
                }
            }).ConfigureAwait(false);

            if (userInput)
            {
                await game.OnUpdateThread(() =>
                    game.SyntheticInput.Activate(player.DrawableRuleset.Playfield.GamefieldToScreenSpace(new Vector2(256, 192)))).ConfigureAwait(false);
            }

            t0 = playable.HitObjects[0].StartTime - 1500;
            t1 = playable.HitObjects[^1].GetEndTime();

            if (player!.Score.Replay.Frames.Count > 0)
                t1 = Math.Max(t1, player.Score.Replay.Frames.Max(f => f.Time));

            return new Dictionary<string, object>
            {
                ["ok"] = true,
                ["t0"] = t0,
                ["t1"] = t1,
                ["objects"] = playable.HitObjects.Count,
                ["title"] = info.GetDisplayTitle(),
                ["difficulty"] = info.DifficultyName,
            };
        }

        private async Task<Dictionary<string, object>> seek(double t)
        {
            var p = player ?? throw new InvalidOperationException("no beatmap loaded");

            phaseWatch.Restart();
            await game.OnUpdateThread(() => p.Player.Seek(t)).ConfigureAwait(false);

            int catchUpFrames = 0;
            double clock = double.NaN;

            while (double.IsNaN(clock))
            {
                clock = await game.OnUpdateThread(() =>
                {
                    double target = p.GameplayClockContainer.CurrentTime;
                    var stable = p.DrawableRuleset.FrameStableClock;

                    if (stable.IsCatchingUp.Value || Math.Abs(stable.CurrentTime - target) > 1e-3)
                        return double.NaN;

                    p.FinishBackgroundTransforms();
                    return target;
                }).ConfigureAwait(false);

                if (++catchUpFrames > 100_000)
                    throw new TimeoutException("seek catch-up did not finish");
            }

            phase(0);
            await game.WaitForDraws(2).ConfigureAwait(false);
            phase(1);

            return new Dictionary<string, object> { ["ok"] = true, ["clock_ms"] = clock, ["catch_up_frames"] = catchUpFrames };
        }

        private Task<Dictionary<string, object>> status() => game.OnUpdateThread(() =>
        {
            var reply = new Dictionary<string, object>
            {
                ["ok"] = true,
                ["loaded"] = player != null,
                ["draw_size"] = new[] { game.DrawSize.X, game.DrawSize.Y },
                ["client_size"] = game.Host.Window == null ? new[] { 0, 0 } : new[] { game.Host.Window.ClientSize.Width, game.Host.Window.ClientSize.Height },
                ["window_title"] = game.Host.Window?.Title ?? "",
            };

            if (player == null)
                return reply;

            var score = player.Score.ScoreInfo;
            player.ScoreProcessor.PopulateScore(score);

            Vector2 tl = player.DrawableRuleset.Playfield.GamefieldToScreenSpace(Vector2.Zero);
            Vector2 br = player.DrawableRuleset.Playfield.GamefieldToScreenSpace(new Vector2(512, 384));

            reply["clock_ms"] = player.GameplayClockContainer.CurrentTime;
            reply["paused"] = player.GameplayClockContainer.IsPaused.Value;
            reply["mouse"] = new[] { game.SyntheticInput.CurrentState.Mouse.Position.X, game.SyntheticInput.CurrentState.Mouse.Position.Y };

            if (player.DrawableRuleset.Cursor?.ActiveCursor is { } activeCursor)
            {
                Vector2 hotspot = activeCursor.ToScreenSpace(activeCursor.OriginPosition);
                reply["cursor"] = new[] { hotspot.X, hotspot.Y };

                var chain = new List<string>();
                for (Drawable? d = activeCursor; d != null; d = d.Parent)
                    chain.Add($"{d.GetType().Name} a={d.Alpha:F3} s={d.Scale.X:F3} c={d.Colour.AverageColour.Linear.R:F2} t={d.Transforms.Count()}");
                reply["cursor_chain"] = chain.ToArray();
            }
            reply["completed"] = player.ScoreProcessor.HasCompleted.Value;
            reply["score"] = score.TotalScore;
            reply["accuracy"] = score.Accuracy;
            reply["combo"] = score.Combo;
            reply["max_combo"] = score.MaxCombo;
            reply["t0"] = t0;
            reply["t1"] = t1;
            reply["play_rect"] = new[] { tl.X, tl.Y, br.X, br.Y };
            reply["sizes"] = new Dictionary<string, object>
            {
                ["player"] = new[] { player.Player.DrawSize.X, player.Player.DrawSize.Y },
                ["drawable_ruleset"] = new[] { player.DrawableRuleset.DrawSize.X, player.DrawableRuleset.DrawSize.Y },
                ["adjustment"] = new[] { player.DrawableRuleset.PlayfieldAdjustmentContainer.DrawSize.X, player.DrawableRuleset.PlayfieldAdjustmentContainer.DrawSize.Y },
                ["playfield"] = new[] { player.DrawableRuleset.Playfield.DrawSize.X, player.DrawableRuleset.Playfield.DrawSize.Y },
            };
            return reply;
        });
    }
}
