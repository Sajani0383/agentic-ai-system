$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$logDir = Join-Path $projectRoot "logs"

if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir | Out-Null
}

$apiOut = Join-Path $logDir "api.out.log"
$apiErr = Join-Path $logDir "api.err.log"
$uiOut = Join-Path $logDir "ui.out.log"
$uiErr = Join-Path $logDir "ui.err.log"

foreach ($file in @($apiOut, $apiErr, $uiOut, $uiErr)) {
    if (Test-Path $file) {
        Remove-Item -LiteralPath $file -Force
    }
}

$api = Start-Process python `
    -ArgumentList "-m", "uvicorn", "adk.agent_api:app", "--host", "127.0.0.1", "--port", "8000" `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $apiOut `
    -RedirectStandardError $apiErr `
    -PassThru

$ui = Start-Process python `
    -ArgumentList "-m", "streamlit", "run", "ui/adk_dashboard.py", "--server.headless", "true", "--server.address", "127.0.0.1", "--server.port", "8501" `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $uiOut `
    -RedirectStandardError $uiErr `
    -PassThru

Start-Sleep -Seconds 5

$apiReady = $false
$uiReady = $false

for ($i = 0; $i -lt 15; $i++) {
    try {
        Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/health | Out-Null
        $apiReady = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

for ($i = 0; $i -lt 20; $i++) {
    try {
        Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8501 | Out-Null
        $uiReady = $true
        break
    } catch {
        Start-Sleep -Seconds 1
    }
}

if ($uiReady) {
    Start-Process "http://127.0.0.1:8501"
}

if ($apiReady) {
    Start-Process "http://127.0.0.1:8000/docs"
}

Write-Output "API PID: $($api.Id)"
Write-Output "UI PID: $($ui.Id)"
Write-Output "API URL: http://127.0.0.1:8000/health"
Write-Output "UI URL: http://127.0.0.1:8501"
Write-Output "API Ready: $apiReady"
Write-Output "UI Ready: $uiReady"
