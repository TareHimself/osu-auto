using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Text;

namespace OsuAuto.Host.Export;

public sealed class Npz : IDisposable
{
    private readonly ZipArchive archive;

    public Npz(string path)
    {
        archive = new ZipArchive(File.Create(path), ZipArchiveMode.Create);
    }

    public void Add(string name, double[] data) => add(name, "<f8", MemoryMarshal.AsBytes(data.AsSpan()), data.Length);
    public void Add(string name, float[] data) => add(name, "<f4", MemoryMarshal.AsBytes(data.AsSpan()), data.Length);
    public void Add(string name, int[] data) => add(name, "<i4", MemoryMarshal.AsBytes(data.AsSpan()), data.Length);
    public void Add(string name, byte[] data) => add(name, "|u1", data, data.Length);

    private void add(string name, string descr, ReadOnlySpan<byte> bytes, int length)
    {
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("npy writer assumes little endian");

        var entry = archive.CreateEntry($"{name}.npy", CompressionLevel.Fastest);
        using var stream = entry.Open();

        string header = $"{{'descr': '{descr}', 'fortran_order': False, 'shape': ({length},), }}";
        int unpadded = 10 + header.Length + 1;
        int padding = (64 - unpadded % 64) % 64;
        header = header + new string(' ', padding) + "\n";

        stream.Write(new byte[] { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0 });
        stream.Write(BitConverter.GetBytes((ushort)header.Length));
        stream.Write(Encoding.ASCII.GetBytes(header));
        stream.Write(bytes);
    }

    public void Dispose() => archive.Dispose();
}
