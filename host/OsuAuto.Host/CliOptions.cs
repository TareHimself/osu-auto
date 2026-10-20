namespace OsuAuto.Host;

public sealed class CliOptions
{
    private readonly Dictionary<string, string?> values = new(StringComparer.OrdinalIgnoreCase);
    public List<string> Positional { get; } = new();

    public static CliOptions Parse(string[] args)
    {
        var o = new CliOptions();

        for (int i = 0; i < args.Length; i++)
        {
            string a = args[i];

            if (!a.StartsWith("--", StringComparison.Ordinal))
            {
                o.Positional.Add(a);
                continue;
            }

            string key = a[2..];
            bool hasValue = i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal);
            o.values[key] = hasValue ? args[++i] : null;
        }

        return o;
    }

    public bool Flag(string name) => values.ContainsKey(name);
    public string? Get(string name) => values.TryGetValue(name, out string? v) ? v : null;
    public string Require(string name) => Get(name) ?? throw new ArgumentException($"--{name} is required");
}
