$logPath = Join-Path $PSScriptRoot "checkpoints\training.log"

Write-Host "Theo dõi log: $logPath"
Write-Host "Nhấn Ctrl+C để dừng theo dõi."

while (-not (Test-Path -LiteralPath $logPath)) {
    Write-Host "Đang chờ file log được tạo..."
    Start-Sleep -Seconds 2
}

Get-Content -LiteralPath $logPath -Tail 20 -Wait
