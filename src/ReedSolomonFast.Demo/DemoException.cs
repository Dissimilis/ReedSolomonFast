namespace ReedSolomonFast.Demo;

internal enum ExitCode
{
    Success = 0,
    CheckFailed = 1,
    Usage = 2,
    Input = 3,
    Io = 4,
}

/// <summary>An expected failure that ends the run with one line on stderr instead of a stack trace.</summary>
internal sealed class DemoException(ExitCode code, string message) : Exception(message)
{
    public ExitCode Code { get; } = code;
}
