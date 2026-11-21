# Builds the Downs & Vogel Fortran simulator as a Python extension (Windows, Python 3.11 venv, gfortran).
# Sources come from vendor/tep2py-master (upstream: github.com/camaramm/tep2py). Generated files go to sim/build (git-ignored).
# `numpy.f2py -c` failed with numpy 2.x on this setup, so the f2py wrappers are generated and compiled by hand.
#
# Prerequisite: a gfortran toolchain on PATH, e.g.  winget install BrechtSanders.WinLibs.POSIX.UCRT
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = "$root\.venv\Scripts\python.exe"

if (-not (Get-Command gfortran -ErrorAction SilentlyContinue)) {
    $pk = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Filter "BrechtSanders.WinLibs*" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pk) { $env:PATH = "$($pk.FullName)\mingw64\bin;$env:PATH" }
}
if (-not (Get-Command gfortran -ErrorAction SilentlyContinue)) { throw "gfortran not found. Install WinLibs (winget install BrechtSanders.WinLibs.POSIX.UCRT) and retry." }
$gbin = Split-Path (Get-Command gfortran).Source

$build = "$PSScriptRoot\build"
if (Test-Path $build) { Remove-Item $build -Recurse -Force }
New-Item -ItemType Directory $build | Out-Null
$vs = "$root\vendor\tep2py-master"
Copy-Item "$vs\src\tep\temain_mod.f", "$vs\src\tep\teprob.f", "$vs\temain_mod-smart.pyf" $build
Set-Location $build

$npdir = & $py -c "import numpy,os;print(os.path.dirname(numpy.__file__))"
$pyinc = & $py -c "import sysconfig;print(sysconfig.get_paths()['include'])"
$pydll = & $py -c "import sys,os;print(os.path.join(sys.base_prefix,'python%d%d.dll' % sys.version_info[:2]))"
$tag = & $py -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))"

& $py -m numpy.f2py temain_mod-smart.pyf --build-dir gen | Out-Null
$inc = @("-I$pyinc", "-I$npdir\_core\include", "-I$npdir\f2py\src", "-I.\gen")
gcc -O2 -DMS_WIN64 @inc -c gen\temain_modmodule.c "$npdir\f2py\src\fortranobject.c"
gfortran -O2 -std=legacy -w -c temain_mod.f teprob.f gen\temain_mod-f2pywrappers.f
$objs = Get-ChildItem *.o | ForEach-Object Name
gfortran -shared -o "temain_mod$tag" @objs $pydll -static-libgcc -static-libgfortran -static-libquadmath '-Wl,-Bstatic' -lwinpthread '-Wl,-Bdynamic'

Copy-Item "temain_mod$tag" $PSScriptRoot
if (Test-Path "$gbin\libwinpthread-1.dll") { Copy-Item "$gbin\libwinpthread-1.dll" $PSScriptRoot }
Set-Location $root
Write-Host "built sim\temain_mod$tag"
