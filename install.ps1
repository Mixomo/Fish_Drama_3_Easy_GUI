$ErrorActionPreference = "Stop"
Set-StrictMode -Version 2.0
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Runtime = Join-Path $Root ".runtime"
$UvDir = Join-Path $Runtime "uv"
$UvExe = Join-Path $UvDir "uv.exe"
$Venv = Join-Path $Root ".venv"
$PythonExe = Join-Path $Venv "Scripts\python.exe"
$AudioCpp = Join-Path $Runtime "audio.cpp"
$SpeechCpp = Join-Path $Runtime "speech.cpp"
$Downloads = Join-Path $Runtime "downloads"
$Temp = Join-Path $Runtime "temp"
$Logs = Join-Path $Runtime "logs"
$VulkanRuntimeBin = Join-Path $Runtime "vulkan\bin"
$VulkanSdkStore = Join-Path $Runtime "vulkan-sdk"
$LocalVulkanSdk = ""

function Ensure-Directory([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

foreach ($Path in @($Runtime, $UvDir, $Downloads, $Temp, $Venv,
        (Join-Path $Root "models"), (Join-Path $Root "outputs"),
        (Join-Path $Root "samples"),
        (Join-Path $Runtime "cache\huggingface"), $Logs, $VulkanRuntimeBin)) {
    Ensure-Directory $Path
}

$InstallLog = Join-Path $Logs "install.log"
Start-Transcript -Path $InstallLog -Append | Out-Null
Write-Host "Installer log: $InstallLog" -ForegroundColor DarkGray

$Git = Get-Command git.exe -ErrorAction SilentlyContinue
if (-not $Git) {
    throw "Git for Windows is required. Install Git, then rerun install.bat."
}

if (-not (Test-Path $UvExe)) {
    $UvVersion = "0.11.33"
    $UvZip = Join-Path $Downloads "uv-$UvVersion-windows-x64.zip"
    & curl.exe --fail --location --retry 4 --output $UvZip `
        "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
    if ($LASTEXITCODE -ne 0) { throw "Could not download uv $UvVersion." }
    & tar.exe -xf $UvZip -C $UvDir
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $UvExe)) {
        throw "Could not extract uv into $UvDir."
    }
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $Runtime "python"
$env:UV_PROJECT_ENVIRONMENT = $Venv
$env:UV_CACHE_DIR = Join-Path $Runtime "uv-cache"
$env:HF_HOME = Join-Path $Runtime "cache\huggingface"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Write-Host "`n=== GUI Python environment ===" -ForegroundColor Cyan
& $UvExe python install 3.12
if ($LASTEXITCODE -ne 0) { throw "uv could not install Python 3.12." }
if (-not (Test-Path $PythonExe)) {
    & $UvExe venv --python 3.12 $Venv
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $PythonExe)) {
        throw "The project Python 3.12 environment could not be created at $Venv."
    }
}
& $UvExe pip install --python $PythonExe "gradio==5.49.1" "faster-whisper==1.2.0" "av<19" soundfile requests
if ($LASTEXITCODE -ne 0) { throw "The Drama 3 GUI dependencies could not be installed." }
Write-Host "`n=== audio.cpp Drama 3 C++ runtime ===" -ForegroundColor Cyan
if (-not (Test-Path (Join-Path $AudioCpp ".git"))) {
    if (Test-Path $AudioCpp) {
        throw "audio.cpp source path exists but is not a Git checkout: $AudioCpp"
    }
    & $Git.Source clone --depth 1 --branch drama-3-native `
        https://github.com/JarodMica/audio.cpp.git $AudioCpp
    if ($LASTEXITCODE -ne 0) { throw "Could not clone the Drama 3 audio.cpp branch." }
}

Write-Host "`n=== speech.cpp Drama 3 runtime ===" -ForegroundColor Cyan
$SpeechExe = Join-Path $SpeechCpp "speech.exe"
if (-not (Test-Path (Join-Path $env:WINDIR "System32\vulkan-1.dll"))) {
    throw "speech.cpp's Vulkan runtime requires the GPU driver's vulkan-1.dll. Install/update a Vulkan-capable GPU driver, then rerun install.bat."
}
if (-not (Test-Path $SpeechExe)) {
    $Release = "0.8.8"
    $ArchiveName = "speech-$Release-windows-x64-vulkan.zip"
    $Archive = Join-Path $Downloads $ArchiveName
    $Checksums = Join-Path $Downloads "speech-$Release-SHA256SUMS"
    $ReleaseUrl = "https://github.com/nyosegawa/speech.cpp/releases/download/v$Release"
    & curl.exe --fail --location --retry 4 --output $Archive "$ReleaseUrl/$ArchiveName"
    if ($LASTEXITCODE -ne 0) { throw "Could not download speech.cpp $Release Vulkan release." }
    & curl.exe --fail --location --retry 4 --output $Checksums "$ReleaseUrl/SHA256SUMS"
    if ($LASTEXITCODE -ne 0) { throw "Could not download speech.cpp release checksums." }
    $ExpectedHashLine = Get-Content -LiteralPath $Checksums | Where-Object { $_ -match "\s\*?$([regex]::Escape($ArchiveName))$" } | Select-Object -First 1
    $ExpectedHash = if ($ExpectedHashLine) { ($ExpectedHashLine -split '\s+')[0].ToLowerInvariant() } else { "" }
    $ActualHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash
    if (-not $ExpectedHash -or $ActualHash.ToLowerInvariant() -ne $ExpectedHash) { throw "speech.cpp release SHA-256 verification failed." }
    Ensure-Directory $SpeechCpp
    $SpeechStage = Join-Path $Temp "speech.cpp-$Release"
    if (Test-Path $SpeechStage) { Remove-Item -LiteralPath $SpeechStage -Recurse -Force }
    Ensure-Directory $SpeechStage
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($Archive, $SpeechStage)
    $FoundSpeech = Get-ChildItem -Path $SpeechStage -Filter speech.exe -Recurse | Select-Object -First 1
    if (-not $FoundSpeech) { throw "speech.exe is missing from the verified release archive." }
    Copy-Item -Path (Join-Path $FoundSpeech.Directory.FullName "*") -Destination $SpeechCpp -Recurse -Force
    $SpeechExe = Join-Path $SpeechCpp "speech.exe"
    if (-not (Test-Path $SpeechExe)) { throw "speech.exe is missing after staging the release." }
    $SpeechVersion = (& $SpeechExe --version) -join ""
    if ($LASTEXITCODE -ne 0 -or $SpeechVersion -notlike "speech.cpp $Release,*") { throw "The speech.cpp executable version check failed: $SpeechVersion" }
}
& $Git.Source -C $AudioCpp `
    -c submodule.external/audio.cpp-server-frontends.url=https://github.com/0xShug0/audio.cpp-server-frontends.git `
    submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "audio.cpp frontend submodule setup failed." }

$CudaRoot = $env:CUDA_PATH
$HasCuda = $CudaRoot -and (Test-Path (Join-Path $CudaRoot "bin\nvcc.exe"))
if (-not $HasCuda) {
    Write-Host "CUDA Toolkit not found through CUDA_PATH; skipping the CUDA build." -ForegroundColor Yellow
}

function Test-VulkanSdkInstall([string]$Path) {
    return (
        (Test-Path -LiteralPath (Join-Path $Path "Bin\glslc.exe") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "Bin\glslangValidator.exe") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "Lib\vulkan-1.lib") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $Path "Include\vulkan\vulkan.h") -PathType Leaf)
    )
}

Ensure-Directory $VulkanSdkStore
$LocalVulkanSdk = ""
if (Test-VulkanSdkInstall $VulkanSdkStore) {
    $LocalVulkanSdk = $VulkanSdkStore
} else {
    $ExistingSdk = Get-ChildItem -LiteralPath $VulkanSdkStore -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending |
        Where-Object { Test-VulkanSdkInstall $_.FullName } |
        Select-Object -First 1
    if ($ExistingSdk) { $LocalVulkanSdk = $ExistingSdk.FullName }
}

if (-not $LocalVulkanSdk) {
    Write-Host "Downloading the official LunarG Vulkan SDK into .runtime..." -ForegroundColor Cyan
    $LatestSdk = Invoke-RestMethod -UseBasicParsing -Uri "https://vulkan.lunarg.com/sdk/latest/windows.json"
    $VulkanVersion = [string]$LatestSdk.windows
    if ($VulkanVersion -notmatch '^\d+\.\d+\.\d+\.\d+$') {
        throw "LunarG returned an invalid Vulkan SDK version: $VulkanVersion"
    }
    $SdkHashInfo = Invoke-RestMethod -UseBasicParsing -Uri "https://vulkan.lunarg.com/sdk/sha/$VulkanVersion/windows/vulkan_sdk.exe.json"
    $ExpectedSdkHash = [string]$SdkHashInfo.sha
    if ($ExpectedSdkHash -notmatch '^[0-9a-fA-F]{64}$') {
        throw "LunarG did not provide a valid SHA-256 for Vulkan SDK $VulkanVersion."
    }

    $SdkInstaller = Join-Path $Downloads "vulkan_sdk-$VulkanVersion.exe"
    & curl.exe --fail --location --retry 4 --output $SdkInstaller `
        "https://sdk.lunarg.com/sdk/download/$VulkanVersion/windows/vulkan_sdk.exe?Human=true"
    if ($LASTEXITCODE -ne 0) { throw "Could not download LunarG Vulkan SDK $VulkanVersion." }
    $ActualSdkHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $SdkInstaller).Hash
    if ($ActualSdkHash -ine $ExpectedSdkHash) {
        Remove-Item -LiteralPath $SdkInstaller -Force -ErrorAction SilentlyContinue
        throw "Vulkan SDK $VulkanVersion SHA-256 verification failed."
    }

    $SdkStage = Join-Path $Temp "vulkan-sdk-$VulkanVersion"
    if (Test-Path -LiteralPath $SdkStage) { Remove-Item -LiteralPath $SdkStage -Recurse -Force }
    Ensure-Directory $SdkStage
    & $SdkInstaller --root $SdkStage --accept-licenses --default-answer --confirm-command install copy_only=1
    if ($LASTEXITCODE -ne 0) { throw "LunarG Vulkan SDK $VulkanVersion installation failed." }
    if (-not (Test-VulkanSdkInstall $SdkStage)) {
        throw "The installed Vulkan SDK $VulkanVersion is incomplete."
    }

    $VersionedSdk = Join-Path $VulkanSdkStore $VulkanVersion
    if (Test-Path -LiteralPath $VersionedSdk) { Remove-Item -LiteralPath $VersionedSdk -Recurse -Force }
    Move-Item -LiteralPath $SdkStage -Destination $VersionedSdk
    $LocalVulkanSdk = $VersionedSdk
}

$HasVulkan = $true
$VulkanRoot = $LocalVulkanSdk
$env:VULKAN_SDK = $VulkanRoot
$VulkanLoader = Join-Path $env:WINDIR "System32\vulkan-1.dll"
if (Test-Path $VulkanLoader) {
    Copy-Item $VulkanLoader (Join-Path $VulkanRuntimeBin "vulkan-1.dll") -Force
    Write-Host "Bundled Vulkan loader in $VulkanRuntimeBin (GPU driver remains a system prerequisite)." -ForegroundColor Green
} else {
    Write-Host "System Vulkan loader not found; Vulkan build will use the installed GPU driver's loader." -ForegroundColor Yellow
}
Write-Host "Using full LunarG Vulkan SDK from $LocalVulkanSdk." -ForegroundColor Green
$BuildScript = Join-Path $AudioCpp "scripts\build_windows.ps1"
function Test-Drama3BuildState([string]$BuildDir, [string]$Preset) {
    $cache = Join-Path $BuildDir "CMakeCache.txt"
    $ninja = Join-Path $BuildDir "build.ninja"
    if (-not (Test-Path -LiteralPath $cache -PathType Leaf) -or
        -not (Test-Path -LiteralPath $ninja -PathType Leaf)) { return $false }
    $entries = Get-Content -LiteralPath $cache
    $expected = @("AUDIOCPP_MODEL_SET:STRING=custom", "AUDIOCPP_MODELS:STRING=fish_drama3")
    if ($Preset -eq "windows-cuda-release") {
        $expected += @("ENGINE_ENABLE_CUDA:BOOL=ON", "ENGINE_ENABLE_VULKAN:BOOL=OFF")
    } else {
        $expected += @("ENGINE_ENABLE_CUDA:BOOL=OFF", "ENGINE_ENABLE_VULKAN:BOOL=ON")
    }
    foreach ($entry in $expected) {
        if ($entries -notcontains $entry) { return $false }
    }
    return $true
}
$BuildPresets = @()
$CudaBuildDir = Join-Path $AudioCpp "build\windows-cuda-release"
$VulkanBuildDir = Join-Path $AudioCpp "build\windows-vulkan-release"
$CudaServer = Join-Path $CudaBuildDir "bin\audiocpp_server.exe"
$VulkanServer = Join-Path $VulkanBuildDir "bin\audiocpp_server.exe"
$CudaBuildState = Test-Drama3BuildState $CudaBuildDir "windows-cuda-release"
$VulkanBuildState = Test-Drama3BuildState $VulkanBuildDir "windows-vulkan-release"
if ($HasCuda -or ((Test-Path -LiteralPath $CudaServer -PathType Leaf) -and $CudaBuildState)) {
    $BuildPresets += "windows-cuda-release"
}
if ($HasVulkan -or ((Test-Path -LiteralPath $VulkanServer -PathType Leaf) -and $VulkanBuildState)) {
    $BuildPresets += "windows-vulkan-release"
}
$BuildTargets = @("audiocpp_server")
if ($BuildPresets.Count -eq 0) { throw "No GPU backend is available. Install CUDA Toolkit or the Vulkan SDK, then rerun install.bat." }
$PendingBuilds = @()
foreach ($Preset in $BuildPresets) {
    foreach ($Target in $BuildTargets) {
        $ExpectedBinary = Join-Path $AudioCpp "build\$Preset\bin\$Target.exe"
        $BuildDir = Join-Path $AudioCpp "build\$Preset"
        $BuildMetadata = Test-Drama3BuildState $BuildDir $Preset
        $BinaryValid = $false
        if (Test-Path -LiteralPath $ExpectedBinary -PathType Leaf) {
            $BinaryValid = (Get-Item -LiteralPath $ExpectedBinary).Length -gt 0
        }
        if ($BinaryValid -and $BuildMetadata) {
            Write-Host "Skipping $Preset / $Target; compiled binary and CMake/Ninja build state are present." -ForegroundColor Green
            continue
        }
        $PendingBuilds += [pscustomobject]@{ Preset = $Preset; Target = $Target; Binary = $ExpectedBinary }
    }
}

if ($PendingBuilds.Count -gt 0) {
    if (-not (Test-Path $BuildScript)) { throw "audio.cpp Windows build helper is missing: $BuildScript" }
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $VsWhere)) {
        throw "Visual Studio 2022 C++ Build Tools are required to build missing audio.cpp targets."
    }
    $VcTools = & $VsWhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $VcTools) {
        throw "Install the Visual Studio 2022 C++ build tools workload, then rerun install.bat."
    }
    $Jobs = [Math]::Max(1, [Math]::Min(4, [Environment]::ProcessorCount))
    $TotalBuilds = $PendingBuilds.Count
    $BuildIndex = 0
    foreach ($Build in $PendingBuilds) {
        $Preset = $Build.Preset
        $Target = $Build.Target
        $BuildIndex++
        $Percent = [int](100 * ($BuildIndex - 1) / $TotalBuilds)
        $BuildStatus = "$Preset · $Target · build $BuildIndex/$TotalBuilds"
        Write-Progress -Activity "Compiling Drama 3 C++ backends" -Status $BuildStatus -PercentComplete $Percent
        Write-Host "`n=== [$BuildIndex/$TotalBuilds] $BuildStatus ===" -ForegroundColor Cyan
        $BuildLog = Join-Path $Logs "$Preset-$Target.log"
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $BuildScript `
            -Preset $Preset -ModelSet custom -Models fish_drama3 `
            -Target $Target -Jobs $Jobs *>&1 |
            Tee-Object -FilePath $BuildLog -Append
        if ($LASTEXITCODE -ne 0) { throw "audio.cpp build failed for $Preset / $Target. See $BuildLog." }
        $BuiltBinaryValid = $false
        if (Test-Path -LiteralPath $Build.Binary -PathType Leaf) {
            $BuiltBinaryValid = (Get-Item -LiteralPath $Build.Binary).Length -gt 0
        }
        if (-not $BuiltBinaryValid) {
            throw "audio.cpp reported success, but the expected binary is missing or empty: $($Build.Binary)"
        }
        Write-Progress -Activity "Compiling Drama 3 C++ backends" -Status $BuildStatus -PercentComplete ([int](100 * $BuildIndex / $TotalBuilds))
    }
    Write-Progress -Activity "Compiling Drama 3 C++ backends" -Completed
} else {
    Write-Host "All requested audio.cpp targets are already built; skipping Visual Studio checks and compilation." -ForegroundColor Green
}

$CppServer = if ($BuildPresets -contains "windows-cuda-release") { $CudaServer } else { $VulkanServer }
if (-not (Test-Path $CppServer)) { throw "No CUDA or Vulkan audio.cpp server binary was produced." }
& $PythonExe -c "import gradio, faster_whisper, ctranslate2, soundfile; print('gradio', gradio.__version__); print('audio and transcription imports: OK')"
if ($LASTEXITCODE -ne 0) { throw "Python runtime verification failed." }

Write-Host "`nInstallation completed. Drama 3 models are downloaded on first use." -ForegroundColor Green
Write-Host "Drama 3 inference uses audio.cpp and speech.cpp GPU GGUF runtimes." -ForegroundColor Green
Write-Host "Compiled backends: $($BuildPresets -join ', ')." -ForegroundColor Green
Write-Host "Start the GUI with start.bat." -ForegroundColor Green
Stop-Transcript | Out-Null



