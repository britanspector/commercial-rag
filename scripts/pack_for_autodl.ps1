# commercial-rag → AutoDL 打包脚本（Windows / PowerShell）
# 用法：
#   .\scripts\pack_for_autodl.ps1                    # 默认 tier=essential+recommended
#   .\scripts\pack_for_autodl.ps1 -Tier minimal      # 仅代码+评测集
#   .\scripts\pack_for_autodl.ps1 -Tier full         # 含 PDF 与 MinerU 输出
#   .\scripts\pack_for_autodl.ps1 -OutDir D:\backup
#
# 注意：本脚本仅生成压缩包，不会上传。请手动 scp / AutoDL 面板上传到服务器。

param(
    [ValidateSet("minimal", "essential", "recommended", "full")]
    [string]$Tier = "essential",
    [string]$OutDir = "",
    [switch]$IncludeEvalArtifacts,
    [switch]$IncludeNotes
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

if (-not $OutDir) {
    $OutDir = Join-Path $ProjectRoot "dist"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$ArchiveName = "commercial-rag-autodl-$Tier-$Stamp.zip"
$ArchivePath = Join-Path $OutDir $ArchiveName
$StageDir = Join-Path $env:TEMP "commercial-rag-pack-$Stamp"
$StageRoot = Join-Path $StageDir "commercial-rag"

if (Test-Path $StageDir) { Remove-Item -Recurse -Force $StageDir }
New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null

function Copy-IfExists($RelPath) {
    $Src = Join-Path $ProjectRoot $RelPath
    if (-not (Test-Path $Src)) {
        Write-Host "[skip] 不存在: $RelPath" -ForegroundColor DarkYellow
        return
    }
    $Dst = Join-Path $StageRoot $RelPath
    $DstParent = Split-Path -Parent $Dst
    if ($DstParent -and -not (Test-Path $DstParent)) {
        New-Item -ItemType Directory -Force -Path $DstParent | Out-Null
    }
    if (Test-Path $Src -PathType Container) {
        Copy-Item -Recurse -Force $Src $Dst
    } else {
        Copy-Item -Force $Src $Dst
    }
    Write-Host "[ok]   $RelPath"
}

Write-Host "`n=== commercial-rag 打包 tier=$Tier ===" -ForegroundColor Cyan
Write-Host "项目根: $ProjectRoot"
Write-Host "输出:   $ArchivePath`n"

# ── 所有 tier 共有：代码与文档 ──
$Always = @(
    "README.md",
    ".gitignore",
    "requirements.txt",
    "src",
    "scripts",
    "docs"
)
foreach ($p in $Always) { Copy-IfExists $p }

# ── minimal：+ 评测集 ──
if ($Tier -in @("minimal", "essential", "recommended", "full")) {
    Copy-IfExists "data/eval/eval_questions.jsonl"
    Get-ChildItem -Path (Join-Path $ProjectRoot "data/eval") -Filter "eval_*.csv" -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-IfExists ("data/eval/" + $_.Name) }
}

# ── essential：+ 已构建索引（免重跑 embed/BM25）──
if ($Tier -in @("essential", "recommended", "full")) {
    Copy-IfExists "data/parsed/chunks.jsonl"
    Copy-IfExists "data/parsed/doc_manifest.jsonl"
    Copy-IfExists "data/parsed/documents.jsonl"
    Copy-IfExists "data/parsed/chunk_summary.csv"
    Copy-IfExists "data/parsed/embed_summary.csv"
    Copy-IfExists "data/parsed/parse_summary.csv"
    Copy-IfExists "data/vector/milvus.db"
    Copy-IfExists "data/vector/bm25_index.pkl"
    Copy-IfExists "data/parsed/.gitkeep"
    Copy-IfExists "data/vector/.gitkeep"
    Copy-IfExists "data/raw_pdfs/.gitkeep"
}

# ── recommended：+ MinerU 中间结果（省解析时间，体积大）──
if ($Tier -in @("recommended", "full")) {
    Copy-IfExists "data/parsed/mineru"
}

# ── full：+ 原始 PDF ──
if ($Tier -eq "full") {
    Copy-IfExists "data/raw_pdfs"
}

if ($IncludeEvalArtifacts) {
    Copy-IfExists "data/eval/eval_detail.jsonl"
    Get-ChildItem -Path (Join-Path $ProjectRoot "data/eval") -Filter "eval_misses*.jsonl" -ErrorAction SilentlyContinue |
        ForEach-Object { Copy-IfExists ("data/eval/" + $_.Name) }
    Copy-IfExists "data/eval/eval_rerank_results.csv"
    Copy-IfExists "data/eval/eval_rerank_answer_results.csv"
}

if ($IncludeNotes) {
    Copy-IfExists "notes"
}

# 清理 staging 中的 __pycache__ / .obsidian
Get-ChildItem -Path $StageRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path $StageRoot -Recurse -Directory -Filter ".obsidian" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# 写入 manifest
$Manifest = @"
commercial-rag AutoDL pack
tier: $Tier
created: $(Get-Date -Format o)
project_root: $ProjectRoot

目录说明:
  minimal     代码 + docs + 评测集（服务器全量重跑数据）
  essential   + chunks/milvus/bm25（跳过 embed，适合 POC 迁移）
  recommended + mineru/（跳过 PDF 解析）
  full        + raw_pdfs/（完整 24 份 POC）

服务器解压:
  unzip commercial-rag-autodl-*.zip -d /root/autodl-tmp/
  cd /root/autodl-tmp/commercial-rag

Agent 上下文: docs/CURSOR_AGENT_CONTEXT.md
中期实验:     docs/midterm-summary.md
"@
$Manifest | Out-File -FilePath (Join-Path $StageRoot "PACK_MANIFEST.txt") -Encoding utf8

Write-Host "`n正在压缩..." -ForegroundColor Cyan
if (Test-Path $ArchivePath) { Remove-Item -Force $ArchivePath }
Compress-Archive -Path $StageRoot -DestinationPath $ArchivePath -CompressionLevel Optimal

Remove-Item -Recurse -Force $StageDir

$SizeMB = [math]::Round((Get-Item $ArchivePath).Length / 1MB, 2)
Write-Host "`n完成: $ArchivePath ($SizeMB MB)" -ForegroundColor Green
Write-Host "上传示例: scp `"$ArchivePath`" root@<autodl-host>:/root/autodl-tmp/"
