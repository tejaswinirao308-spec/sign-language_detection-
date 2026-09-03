<#
    Starts this dashboard with the project's virtual-environment interpreter.

    This prevents a globally installed Streamlit (which may not include PyAV)
    from being used for the WebRTC webcam component.
#>
[CmdletBinding()]
param(
    [ValidateRange(1, 65535)]
    [int]$Port = 8501
)

$projectRoot = Split-Path -Parent $PSCommandPath
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$app = Join-Path $projectRoot "app.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project interpreter not found: $python. Create/install .venv first."
}

& $python -m streamlit run $app --server.port $Port
