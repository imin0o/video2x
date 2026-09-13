param(
    [string]$VulkanSdk = 'C:\VulkanSDK\1.4.350.0',
    [int]$Jobs = 8,
    [switch]$SkipDependencies,
    [switch]$ConfigureOnly
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$build = Join-Path $repo 'build\art'
$install = Join-Path $build 'install'
New-Item -ItemType Directory -Force -Path $build | Out-Null
$env:PROCESSOR_ARCHITECTURE = 'AMD64'
$env:PreferredToolArchitecture = 'x64'
$env:VULKAN_SDK = $VulkanSdk
if (-not (Test-Path -LiteralPath (Join-Path $VulkanSdk 'Bin\glslangValidator.exe'))) {
    throw 'Vulkan SDK not found; pass -VulkanSdk with the installed SDK path.'
}
$vswhere = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'
$vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if ($LASTEXITCODE -ne 0 -or -not $vs) { throw 'Visual Studio C++ x64 build tools not found.' }
$msbuild = Join-Path $vs 'MSBuild\Current\Bin\amd64\MSBuild.exe'
function Invoke-Logged([string]$Exe, [string[]]$Arguments, [string]$Log) {
    Write-Output "Running $Exe; log: $Log"
    $ErrorActionPreference = 'Continue'
    & $Exe @Arguments *> $Log
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed (exit $LASTEXITCODE); see $Log" }
}
Push-Location $repo
try {
    if (-not $SkipDependencies) {
        Invoke-Logged 'git' @('submodule', 'update', '--init', '--depth', '1',
            'third_party/boost', 'third_party/spdlog', 'third_party/ncnn',
            'third_party/librealesrgan_ncnn_vulkan', 'third_party/librealcugan_ncnn_vulkan',
            'third_party/librife_ncnn_vulkan') (Join-Path $build 'submodules.log')
        Invoke-Logged 'git' @('submodule', 'update', '--init', '--recursive', '--depth', '1',
            '--jobs', "$Jobs", 'third_party/boost') (Join-Path $build 'boost-submodules.log')
    }
    # Match the repository's Windows CI dependency versions.
    $packages = @(
        @('ffmpeg', '7.1', 'ffmpeg-7.1-full_build-shared', 'ffmpeg-shared',
          'https://github.com/GyanD/codexffmpeg/releases/download/7.1/ffmpeg-7.1-full_build-shared.zip'),
        @('ncnn', '20241226', 'ncnn-20241226-windows-vs2022-shared', 'ncnn-shared',
          'https://github.com/Tencent/ncnn/releases/download/20241226/ncnn-20241226-windows-vs2022-shared.zip')
    )
    $archives = @()
    foreach ($package in $packages) {
        $archive = Join-Path $build ($package[2] + '.zip')
        $destination = Join-Path $repo ('third_party\' + $package[3])
        if (-not (Test-Path -LiteralPath $destination)) {
            if (-not (Test-Path -LiteralPath $archive)) {
                Invoke-Logged 'curl.exe' @('-fsSL', '--retry', '3', '-o', ($archive + '.partial'), $package[4]) (Join-Path $build ($package[0] + '-download.log'))
                Move-Item -LiteralPath ($archive + '.partial') -Destination $archive
            }
            Invoke-Logged 'tar.exe' @('-xf', $archive, '-C', (Join-Path $repo 'third_party')) (Join-Path $build ($package[0] + '-extract.log'))
            Move-Item -LiteralPath (Join-Path $repo ('third_party\' + $package[2])) -Destination $destination
        }
        $archiveHash = $null
        if (Test-Path -LiteralPath $archive) { $archiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash }
        $archives += @{ name = $package[0]; version = $package[1]; url = $package[4]; archive_sha256 = $archiveHash }
    }
    $options = @('-S', $repo, '-B', $build, '-G', 'Visual Studio 17 2022',
        '-A', 'x64', '-T', 'host=x64', '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
        '-DVIDEO2X_USE_EXTERNAL_NCNN=OFF', '-DVIDEO2X_USE_EXTERNAL_SPDLOG=OFF',
        '-DVIDEO2X_USE_EXTERNAL_BOOST=OFF', '-DCMAKE_BUILD_TYPE=Release',
        "-DCMAKE_INSTALL_PREFIX=$install", '-DBOOST_INCLUDE_LIBRARIES=program_options')
    Invoke-Logged 'cmake' $options (Join-Path $build 'configure.log')
    $properties = & $msbuild (Join-Path $build 'libvideo2x.vcxproj') /p:Configuration=Release /p:Platform=x64 /getProperty:PreferredToolArchitecture,VCToolArchitecture
    if ($LASTEXITCODE -ne 0) { throw 'MSBuild architecture query failed.' }
    $architecture = ($properties -join "`n") | ConvertFrom-Json
    if ($architecture.Properties.PreferredToolArchitecture -ne 'x64' -or $architecture.Properties.VCToolArchitecture -ne 'Native64Bit') {
        throw 'MSBuild must report x64 / Native64Bit before building.'
    }
    $manifest = @{
        recorded_at = (Get-Date).ToString('o'); repository = (& git rev-parse HEAD)
        submodules = @(& git submodule status --recursive); visual_studio = $vs
        architecture = $architecture.Properties; vulkan_sdk = $VulkanSdk
        cmake = (& cmake --version | Select-Object -First 1)
        packages = $archives; configure_arguments = $options
    }
    $manifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $build 'environment.json') -Encoding UTF8
    if (-not $ConfigureOnly) {
        Invoke-Logged 'cmake' @('--build', $build, '--config', 'Release', '--parallel', "$Jobs", '--target', 'install') (Join-Path $build 'build.log')
        Write-Output "Installed CLI: $install\bin\video2x.exe"
    }
} finally {
    Pop-Location
}
