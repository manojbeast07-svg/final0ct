param(
    [int]$Runs = 10,
    [string]$OpenCVDir = "C:\opencv\build"
)

# Ensure running from repo root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir
Write-Host "Working directory: $PWD"

function Check-Nvcc {
    if (Get-Command nvcc -ErrorAction SilentlyContinue) {
        nvcc --version
        return $true
    }
    Write-Host "nvcc not found on PATH. Install CUDA Toolkit and reopen shell." -ForegroundColor Yellow
    return $false
}

# 1) Verify nvcc
$nvccOk = Check-Nvcc
if (-not $nvccOk) { Write-Host "Aborting: nvcc missing." -ForegroundColor Red; exit 2 }

# 2) Build using build.ps1 if present
if (Test-Path .\build.ps1) {
    Write-Host "Found build.ps1 - invoking it..."
    & .\build.ps1 -OpenCVDir $OpenCVDir -OutputName "clearscan.exe"
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path .\clearscan.exe)) {
        Write-Host "build.ps1 failed or clearscan.exe missing." -ForegroundColor Red
        exit 3
    }
} else {
    Write-Host "build.ps1 not found. Attempting manual nvcc compile..."
    nvcc -std=c++11 -O3 -o clearscan.exe src\main.cu src\filters.cu -Iinclude -Iinclude\stb -I"$OpenCVDir\include" -L"$OpenCVDir\x64\vc17\lib" -lopencv_core455 -lopencv_imgcodecs455 -lopencv_imgproc455
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path .\clearscan.exe)) {
        Write-Host "Manual nvcc build failed." -ForegroundColor Red
        exit 4
    }
}

# 3) Run single trial to produce outputs and show JSON
Write-Host "Running single GPU trial..."
.\clearscan.exe data\BrokenBone1.jpg | Tee-Object -Variable runout
$lastJson = $runout | Select-Object -Last 3
Write-Host "JSON summary (last lines):"
$lastJson | ForEach-Object { Write-Host $_ }

# 4) Multi-run collection
Remove-Item -Path gpu_runtimes.jsonl -ErrorAction SilentlyContinue
for ($i = 1; $i -le $Runs; $i++) {
    Write-Host "Run $i of $Runs"
    $out = .\clearscan.exe data\BrokenBone1.jpg 2>&1
    $out | Out-File -FilePath "clearscan_run_$i.txt" -Encoding UTF8
    $last = $out | Select-Object -Last 3
    $last -join "`n" | Out-File -FilePath gpu_runtimes.jsonl -Append -Encoding UTF8
}

Write-Host "Collected runtimes in gpu_runtimes.jsonl"

# 5) Package results
$zipName = "report_gpu_vs_cpu_$(Get-Date -Format yyyyMMdd_HHmmss).zip"
$items = @()
if (Test-Path .\clearscan.exe) { $items += '.\clearscan.exe' }
if (Test-Path .\gpu_runtimes.jsonl) { $items += '.\gpu_runtimes.jsonl' }
if (Test-Path .\output) { $items += '.\output\*' }
$items += (Get-ChildItem -Filter 'clearscan_run_*.txt' | ForEach-Object { $_.FullName })

if ($items.Count -gt 0) {
    Compress-Archive -Path $items -DestinationPath $zipName -Force
    Write-Host "Packaged results into $zipName"
} else {
    Write-Host "No items found to package." -ForegroundColor Yellow
}

Write-Host "Done. Upload the zip or gpu_runtimes.jsonl and the output folder for analysis."