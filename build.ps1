param(
    [string]$OpenCVDir = "C:\opencv\build",
    [string]$OutputName = "clearscan.exe"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $scriptDir

function Assert-CommandExists {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found on PATH. Open an x64 Native Tools prompt after installing CUDA Toolkit 13.1 and Visual Studio 2022 Build Tools."
    }
}

function Import-VsDeveloperEnvironment {
    $vsDevCmd = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $vsDevCmd)) {
        return $false
    }

    $envOutput = cmd /c "call `"$vsDevCmd`" -arch=x64 -host_arch=x64 >nul && set"
    foreach ($line in $envOutput) {
        if ($line -match '^(?<name>[^=]+)=(?<value>.*)$') {
            [Environment]::SetEnvironmentVariable($Matches.name, $Matches.value, 'Process')
        }
    }

    return $true
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Write-Host "cl.exe not found on PATH; importing Visual Studio developer environment..."
    if (-not (Import-VsDeveloperEnvironment)) {
        throw "Could not locate VsDevCmd.bat to import the Visual Studio environment."
    }
}

Assert-CommandExists cl.exe

Assert-CommandExists nvcc

$opencvInclude = Join-Path $OpenCVDir 'include'
$opencvX64Dir = Join-Path $OpenCVDir 'x64'

if (-not (Test-Path $opencvX64Dir)) {
    throw "OpenCV x64 directory not found: $opencvX64Dir"
}

$opencvVcDir = Get-ChildItem -Path $opencvX64Dir -Directory | Where-Object { $_.Name -match '^vc1[67]$' } | Select-Object -First 1
if (-not $opencvVcDir) {
    throw "Could not find an OpenCV VC16/VC17 directory under $opencvX64Dir"
}

$opencvLibDir = Join-Path $opencvVcDir.FullName 'lib'

if (-not (Test-Path $opencvInclude)) {
    throw "OpenCV include directory not found: $opencvInclude"
}

if (-not (Test-Path $opencvLibDir)) {
    throw "OpenCV library directory not found: $opencvLibDir"
}

function Find-OpenCvLib {
    param([string]$Pattern)

    $match = Get-ChildItem -Path $opencvLibDir -Filter $Pattern | Select-Object -First 1
    if (-not $match) {
        throw "Could not find an OpenCV library matching '$Pattern' in $opencvLibDir"
    }

    return $match.FullName
}

$opencvWorld = $null
try {
    $opencvWorld = Find-OpenCvLib 'opencv_world*.lib'
} catch {
    $opencvWorld = $null
}

$opencvCore = $null
$opencvImgCodecs = $null
$opencvImgProc = $null

if (-not $opencvWorld) {
    $opencvCore = Find-OpenCvLib 'opencv_core*.lib'
    $opencvImgCodecs = Find-OpenCvLib 'opencv_imgcodecs*.lib'
    $opencvImgProc = Find-OpenCvLib 'opencv_imgproc*.lib'
}

$outputPath = Join-Path $scriptDir $OutputName

$nvccArgs = @(
    '-std=c++11'
    '-O3'
    '-arch=sm_80'
    '-o', $outputPath
    'src\main.cu'
    'src\filters.cu'
    '-Iinclude'
    '-Iinclude\stb'
    "-I$opencvInclude"
    "-L$opencvLibDir"
    '-lcudart'
)

if ($opencvWorld) {
    $nvccArgs += $opencvWorld
} else {
    $nvccArgs += $opencvCore
    $nvccArgs += $opencvImgCodecs
    $nvccArgs += $opencvImgProc
}

Write-Host "Building $OutputName using CUDA and OpenCV from: $OpenCVDir"
Write-Host "Detected OpenCV VC directory: $($opencvVcDir.Name)"
if ($opencvWorld) {
    Write-Host "Linking against: $(Split-Path $opencvWorld -Leaf)"
} else {
    Write-Host "Linking against: $(Split-Path $opencvCore -Leaf), $(Split-Path $opencvImgCodecs -Leaf), $(Split-Path $opencvImgProc -Leaf)"
}
Write-Host "nvcc $($nvccArgs -join ' ')"

& nvcc @nvccArgs

if ($LASTEXITCODE -ne 0) {
    throw "nvcc build failed with exit code $LASTEXITCODE"
}

if (-not (Test-Path $outputPath)) {
    throw "Build completed but output binary was not created: $outputPath"
}

Write-Host "Build succeeded: $outputPath"