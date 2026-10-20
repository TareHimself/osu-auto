using osu.Framework.Input;
using osu.Framework.Input.StateChanges;
using osuTK;
using osuTK.Input;

namespace OsuAuto.Host;

public partial class SyntheticInputManager : UserInputManager
{
    private readonly object sync = new object();
    private bool active;
    private Vector2 position;
    private bool left;
    private bool right;

    public void Activate(Vector2 screenPosition)
    {
        lock (sync)
        {
            active = true;
            position = screenPosition;
            left = right = false;
        }
    }

    public void Deactivate()
    {
        lock (sync)
            active = false;
    }

    public void Set(Vector2 screenPosition, bool leftButton, bool rightButton)
    {
        lock (sync)
        {
            position = screenPosition;
            left = leftButton;
            right = rightButton;
        }
    }

    public override bool HandleHoverEvents
    {
        get
        {
            lock (sync)
                return active || base.HandleHoverEvents;
        }
    }

    protected override List<IInput> GetPendingInputs()
    {
        var real = base.GetPendingInputs();

        lock (sync)
        {
            if (!active)
                return real;

            return new List<IInput>
            {
                new MousePositionAbsoluteInput { Position = position },
                new MouseButtonInput(MouseButton.Left, left),
                new MouseButtonInput(MouseButton.Right, right),
            };
        }
    }
}
