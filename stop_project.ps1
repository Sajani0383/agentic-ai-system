$ErrorActionPreference = "SilentlyContinue"

$ports = @(8000, 8501)

foreach ($port in $ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -State Listen
    foreach ($connection in $connections) {
        Stop-Process -Id $connection.OwningProcess -Force
    }
}

Write-Output "Stopped any listeners on ports 8000 and 8501."
