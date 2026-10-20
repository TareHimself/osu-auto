using System.Globalization;
using System.Text.RegularExpressions;
using osu.Game.Online.API;
using osu.Game.Rulesets;
using osu.Game.Rulesets.Mods;

namespace OsuAuto.Host;

public static class ModSpec
{
    public static IReadOnlyList<Mod> Parse(Ruleset ruleset, IEnumerable<string> specs) => specs.Select(s => Parse(ruleset, s)).ToList();

    public static IReadOnlyList<string> Split(string? list) =>
        string.IsNullOrWhiteSpace(list) ? Array.Empty<string>() : list.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    public static Mod Parse(Ruleset ruleset, string spec)
    {
        string[] parts = spec.Split(':', 2);
        var api = new APIMod { Acronym = parts[0].Trim().ToUpperInvariant() };

        if (parts.Length == 2)
        {
            foreach (string kv in parts[1].Split(',', StringSplitOptions.RemoveEmptyEntries))
            {
                string[] p = kv.Split('=', 2);
                if (p.Length != 2)
                    throw new ArgumentException($"mod setting '{kv}' must be name=value");
                api.Settings[p[0].Trim()] = parseValue(p[1].Trim());
            }
        }

        var mod = api.ToMod(ruleset);
        if (mod is UnknownMod)
            throw new ArgumentException($"unknown mod {api.Acronym}");
        return mod;
    }

    public static string Tag(IEnumerable<string> specs) =>
        string.Concat(specs.Select(s => "-" + Regex.Replace(s.Replace(':', '_').Replace(',', '_').Replace("=", ""), "[^A-Za-z0-9._]", "")));

    private static object parseValue(string v) =>
        bool.TryParse(v, out bool b) ? b : double.Parse(v, CultureInfo.InvariantCulture);
}
