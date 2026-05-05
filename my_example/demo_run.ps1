param(
    [string]$EnvName = "TinyContinuousCartPole-v0",
    [int]$Epochs = 5,
    [int]$StepsPerEpoch = 1000,
    [int]$InitialRandomSteps = 1000,
    [int]$EvalEpisodes = 5,
    [int]$MaxEpisodeSteps = 200,
    [string]$OutputDir = "runs",
    [string]$ExperimentName = "sac_cartpole",
    [switch]$RenderEval,
    [switch]$RenderInfer,
    [switch]$PlotProgress
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$checkpoint = Join-Path $OutputDir "$ExperimentName\checkpoints\latest.pt"

Write-Host "=== SAC demo: training ==="
Write-Host "env=$EnvName epochs=$Epochs steps_per_epoch=$StepsPerEpoch"
Write-Host "outputs=$OutputDir\$ExperimentName"

$trainArgs = @(
    "sac.py",
    "--env", $EnvName,
    "--epochs", $Epochs,
    "--steps-per-epoch", $StepsPerEpoch,
    "--initial-random-steps", $InitialRandomSteps,
    "--eval-episodes", $EvalEpisodes,
    "--max-episode-steps", $MaxEpisodeSteps,
    "--output-dir", $OutputDir,
    "--experiment-name", $ExperimentName
)

if ($PlotProgress) {
    $trainArgs += "--plot-progress"
}

if ($RenderEval) {
    $trainArgs += "--render-eval"
}

python @trainArgs

if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Checkpoint was not created: $checkpoint"
}

Write-Host ""
Write-Host "=== SAC demo: saved outputs ==="
Write-Host "progress:   $OutputDir\$ExperimentName\progress.csv"
Write-Host "checkpoint: $checkpoint"
if ($PlotProgress) {
    Write-Host "plot:       $OutputDir\$ExperimentName\returns.png"
}

Write-Host ""
Write-Host "=== SAC demo: inference with saved policy ==="

$inferArgs = @(
    "infer.py",
    "--checkpoint", $checkpoint,
    "--episodes", 3,
    "--max-episode-steps", $MaxEpisodeSteps
)

if ($RenderInfer) {
    $inferArgs += "--render"
}

python @inferArgs

Write-Host ""
Write-Host "=== demo finished ==="
