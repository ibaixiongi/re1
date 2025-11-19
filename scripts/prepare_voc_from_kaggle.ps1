param(
  [Parameter(Mandatory=$true)][string]$KagglePath,
  [string]$Dest = ".",
  [switch]$GenerateImageSets,
  [double]$TrainRatio = 0.9
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Join-Paths([string]$a, [string]$b) { return [System.IO.Path]::Combine($a, $b) }

Write-Host "[Kaggle->VOC] Kaggle source: $KagglePath"
Write-Host "[Kaggle->VOC] Destination: $Dest"

if (-not (Test-Path $KagglePath)) { throw "KagglePath not found: $KagglePath" }
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

$destVocRoot = Join-Paths $Dest 'VOCdevkit'
New-Item -ItemType Directory -Force -Path $destVocRoot | Out-Null

function Copy-IfExists($src, $dst) {
  if (Test-Path $src) {
    Write-Host "[Kaggle->VOC] Copying $(Resolve-Path $src) -> $dst"
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    robocopy $src $dst /E /NFL /NDL /NJH /NJS | Out-Null
    return $true
  }
  return $false
}

# Detect common layouts and copy into $Dest\VOCdevkit
$copied = $false

# Case 1: Already contains VOCdevkit
$vocKit = Get-ChildItem -Directory -Recurse -LiteralPath $KagglePath -Filter 'VOCdevkit' -ErrorAction SilentlyContinue | Select-Object -First 1
if ($vocKit -and (Test-Path (Join-Paths $($vocKit.FullName) 'VOC2007')) ) {
  $copied = Copy-IfExists $vocKit.FullName $destVocRoot
}

# Case 2: Direct VOC2007 / VOC2012 at root
if (-not $copied) {
  $cand2007 = Join-Paths $KagglePath 'VOC2007'
  $cand2012 = Join-Paths $KagglePath 'VOC2012'
  if ((Test-Path (Join-Paths $cand2007 'Annotations')) -and (Test-Path (Join-Paths $cand2007 'JPEGImages'))) {
    $copied = $true
    Copy-IfExists $cand2007 (Join-Paths $destVocRoot 'VOC2007') | Out-Null
  }
  if ((Test-Path (Join-Paths $cand2012 'Annotations')) -and (Test-Path (Join-Paths $cand2012 'JPEGImages'))) {
    $copied = $true
    Copy-IfExists $cand2012 (Join-Paths $destVocRoot 'VOC2012') | Out-Null
  }
}

# Case 3: VOCtrainval_* folders
if (-not $copied) {
  $trainvalFolders = Get-ChildItem -Directory -LiteralPath $KagglePath -Filter 'VOCtrainval*' -ErrorAction SilentlyContinue
  foreach ($f in $trainvalFolders) {
    $innerKit = Join-Paths $f.FullName 'VOCdevkit'
    if (Test-Path $innerKit) {
      Copy-IfExists $innerKit $destVocRoot | Out-Null
      $copied = $true
    }
  }
}

if (-not (Test-Path (Join-Paths $destVocRoot 'VOC2007')) -and -not (Test-Path (Join-Paths $destVocRoot 'VOC2012'))) {
  throw "Could not find VOC folders in KagglePath. Make sure you extracted the dataset (it should contain VOC2007 and/or VOC2012 with Annotations and JPEGImages)."
}

function Ensure-ImageSets($year) {
  $yearRoot = Join-Paths $destVocRoot $year
  $anno = Join-Paths $yearRoot 'Annotations'
  $imgs = Join-Paths $yearRoot 'JPEGImages'
  if (-not (Test-Path $anno) -or -not (Test-Path $imgs)) {
    Write-Warning "[$year] Missing Annotations or JPEGImages; skipping ImageSets generation."
    return
  }
  $main = Join-Paths (Join-Paths $yearRoot 'ImageSets') 'Main'
  New-Item -ItemType Directory -Force -Path $main | Out-Null

  $trainvalPath = Join-Paths $main 'trainval.txt'
  if (-not (Test-Path $trainvalPath)) {
    $ids = Get-ChildItem -LiteralPath $anno -Filter *.xml | ForEach-Object { [System.IO.Path]::GetFileNameWithoutExtension($_.Name) }
    Set-Content -Path $trainvalPath -Value ($ids -join "`n") -Encoding ASCII
    Write-Host "[$year] Generated Main/trainval.txt with $($ids.Count) ids"
  } else {
    Write-Host "[$year] Found existing Main/trainval.txt"
  }

  if ($GenerateImageSets) {
    $ids = Get-Content -LiteralPath $trainvalPath
    $count = $ids.Count
    if ($count -gt 0) {
      $rng = New-Object System.Random
      $shuffled = $ids | Sort-Object { $rng.Next() }
      $split = [int]($TrainRatio * $count)
      $trainIds = $shuffled[0..($split-1)]
      $valIds = $shuffled[$split..($count-1)]
      Set-Content -Path (Join-Paths $main 'train.txt') -Value ($trainIds -join "`n") -Encoding ASCII
      Set-Content -Path (Join-Paths $main 'val.txt') -Value ($valIds -join "`n") -Encoding ASCII
      Write-Host "[$year] Generated Main/train.txt ($($trainIds.Count)) and Main/val.txt ($($valIds.Count))"
    }
  }

  # test.txt is required for VOC2007 official test; we cannot synthesize without the official list
  if ($year -eq 'VOC2007') {
    $testPath = Join-Paths $main 'test.txt'
    if (-not (Test-Path $testPath)) {
      Write-Warning "[VOC2007] test.txt not found. Evaluation on VOC2007 test requires the official list (use get_voc.ps1 to fetch VOCtest_06-Nov-2007). Training is unaffected."
    } else {
      Write-Host "[VOC2007] Found Main/test.txt"
    }
  }
}

if (Test-Path (Join-Paths $destVocRoot 'VOC2007')) { Ensure-ImageSets 'VOC2007' }
if (Test-Path (Join-Paths $destVocRoot 'VOC2012')) { Ensure-ImageSets 'VOC2012' }

Write-Host "[Kaggle->VOC] Prepared at: $destVocRoot"
Write-Host "Set data_root in configs/voc_10_10.yaml to: $destVocRoot"

