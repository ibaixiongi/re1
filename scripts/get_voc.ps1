param(
  [string]$Dest = "."
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls11 -bor [Net.SecurityProtocolType]::Tls } catch {}

Write-Host "[VOC] Destination: $Dest"
$destPath = Join-Path $Dest "VOCdevkit"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

# File definitions with multiple mirrors (official + mirror)
$files = @(
  @{ Name = 'VOCtrainval_06-Nov-2007.tar'; Urls = @(
      'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
      'https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar'
    )},
  @{ Name = 'VOCtest_06-Nov-2007.tar'; Urls = @(
      'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
      'https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar'
    )},
  @{ Name = 'VOCtrainval_11-May-2012.tar'; Urls = @(
      'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
      'https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar'
    )}
)

function Test-TarArchive([string]$path) {
  if (-not (Test-Path $path)) { return $false }
  # quick size sanity (should be > 50MB)
  $fi = Get-Item $path
  if ($fi.Length -lt 50000000) { return $false }
  # try listing
  try {
    & tar -tf $path > $null 2> $null
    return ($LASTEXITCODE -eq 0)
  } catch {
    return $false
  }
}

function Ensure-Downloaded([hashtable]$file) {
  $out = Join-Path $Dest $file.Name
  if (Test-TarArchive $out) {
    Write-Host "[VOC] Found valid $($file.Name)"
    return $out
  }
  foreach ($u in $file.Urls) {
    Write-Host "[VOC] Downloading $($file.Name) from $u"
    try {
      Invoke-WebRequest -UseBasicParsing -Headers @{ 'User-Agent' = 'Mozilla/5.0' } -Uri $u -OutFile $out -TimeoutSec 1800
    } catch {
      Write-Warning "[VOC] Download failed from $u: $($_.Exception.Message)"
      continue
    }
    if (Test-TarArchive $out) { return $out }
    else {
      Write-Warning "[VOC] Invalid archive from $u, trying next mirror..."
      Remove-Item -Force $out -ErrorAction SilentlyContinue
    }
  }
  throw "[VOC] Unable to download a valid archive for $($file.Name) from any mirror."
}

function Extract-Archive([string]$tarPath, [string]$dest) {
  Write-Host "[VOC] Extracting $(Split-Path $tarPath -Leaf) ..."
  try {
    & tar -xf $tarPath -C $dest
    if ($LASTEXITCODE -eq 0) { return }
  } catch {}
  # fallback to 7z if available
  $seven = Get-Command 7z -ErrorAction SilentlyContinue
  if ($seven) {
    Write-Host "[VOC] 'tar' failed; trying 7z ..."
    & 7z x -y "`"$tarPath`"" -o"`"$dest`"" > $null
    return
  }
  throw "[VOC] Extraction failed and 7z not available."
}

foreach ($f in $files) {
  $downloaded = Ensure-Downloaded $f
  Extract-Archive -tarPath $downloaded -dest $Dest
}

if (-not (Test-Path $destPath)) {
  Write-Error "[VOC] Expected folder '$destPath' not found after extraction."
  exit 1
}

Write-Host "[VOC] Done. Point configs data_root to: $destPath"
