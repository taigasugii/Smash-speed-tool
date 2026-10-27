# Badminton Analytics の自律エージェント：Claude Code を繰り返し起動して、agent/ROADMAP.md の課題を 1 つずつ進める。
#
#   run.ps1 -Until 12:00   毎晩 0:00 にタスク スケジューラから。12:00 を過ぎたら次の回は始めない
#   run.ps1 -Once          課題を 1 つだけ進める（午後に手で動かすとき。既定は 3 時間が目安）
#
# 止める時刻になったら .agent/WRAPUP を置き、Claude には今の作業を区切りのいい所まで終えてコミットしてもらう。
# それでも終わらないときは、止める時刻 + GraceMinutes で打ち切る（午後の利用枠を使いすぎないため）。
#   run.ps1 -OnRequest     .agent/REQUEST（GUI の「1 つ進める」）があるときだけ -Once で動く。5 分おきに起動される
#
# 変更は agent/dev ブランチにためて push し、main への PR を 1 つ開いておく（人が見てマージする）。
# 品質チェック（agent/eval_gate.py）が通らない変更は取り消し、.agent/failed にパッチとして残す。
# PR へのコメントは、次の回の指示として Claude に渡す。
param(
    [switch]$Once,
    [switch]$OnRequest,
    [string]$Until = "",
    [int]$Hours = 3,
    [int]$GraceMinutes = 60,
    [int]$MaxTurns = 300,
    [string]$Model = "",
    [int]$IdleLimit = 3
)

$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$OutputEncoding = [Text.Encoding]::UTF8
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$St = Join-Path $Root ".agent"
$Logs = Join-Path $St "logs"
$Failed = Join-Path $St "failed"
New-Item -ItemType Directory -Force $St, $Logs, $Failed | Out-Null
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$Settings = Join-Path $Root "agent\claude-settings.json"
$Branch = "agent/dev"
$Utf8 = New-Object System.Text.UTF8Encoding $false
$StopFile = Join-Path $St "STOP"
$WrapFile = Join-Path $St "WRAPUP"
$RequestFile = Join-Path $St "REQUEST"
$LockFile = Join-Path $St "lock.json"
$StatusFile = Join-Path $St "status.json"
$HistoryFile = Join-Path $St "history.jsonl"
$StateFile = Join-Path $St "state.json"
$FailureFile = Join-Path $St "last_failure.md"

# ---------------------------------------------------------------- 小さな道具
function Log([string]$msg) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Write-Host $line
    [IO.File]::AppendAllText((Join-Path $Logs ("run-{0}.log" -f (Get-Date -Format "yyyyMMdd"))), $line + "`r`n", $Utf8)
}
function Save-Json($path, $obj) { [IO.File]::WriteAllText($path, ($obj | ConvertTo-Json -Depth 8), $Utf8) }
function Read-Json($path) {
    if (Test-Path $path) { try { return ([IO.File]::ReadAllText($path, $Utf8) | ConvertFrom-Json) } catch { } }
    return $null
}
# 外部コマンドを実行して、出力を 1 つの文字列で返す（終了コードは $script:Code）
function X([string]$exe) {
    $a = $args
    try {
        $out = & $exe @a 2>&1 | ForEach-Object { "$_" }
    } catch {
        $script:Code = 1      # 起動できなかった
        return "$_"
    }
    $script:Code = $LASTEXITCODE
    return ($out -join "`n")
}

function Set-Status([hashtable]$kv) {
    foreach ($k in $kv.Keys) { $script:Status[$k] = $kv[$k] }
    $script:Status.updated = (Get-Date -Format s)
    Save-Json $StatusFile $script:Status
}
function Add-History([hashtable]$h) {
    [IO.File]::AppendAllText($HistoryFile, (($h | ConvertTo-Json -Compress -Depth 6) + "`n"), $Utf8)
}
function Notify([string]$text) {
    # .agent/config.json の notify_url（例：ntfy.sh のトピック）があれば、スマホに知らせる
    $cfg = Read-Json (Join-Path $St "config.json")
    if (-not $cfg -or -not $cfg.notify_url) { return }
    try { Invoke-RestMethod -Method Post -Uri $cfg.notify_url -Body ([Text.Encoding]::UTF8.GetBytes($text)) -TimeoutSec 20 | Out-Null } catch { Log "notify failed: $_" }
}

# ---------------------------------------------------------------- Git
# 作業ツリーに残った変更をパッチにして消す（前の回が途中で止められたときなど）
function Save-Leftover([string]$tag) {
    $f = Join-Path $Failed ("{0}-{1}.patch" -f (Get-Date -Format "yyyyMMdd-HHmmss"), $tag)
    X git add -A | Out-Null
    X git diff --cached --binary "--output=$f" | Out-Null
    X git reset -q --hard HEAD | Out-Null
    Log "saved uncommitted changes to $f"
}

function Prepare-Branch {
    $script:MergeNote = ""
    X git fetch origin --prune | Out-Null
    if ($script:Code -ne 0) { Log "git fetch failed (offline?)" }
    if (X git status --porcelain) { Save-Leftover "leftover" }
    X git rev-parse --verify --quiet "refs/remotes/origin/$Branch" | Out-Null
    $hasRemote = ($script:Code -eq 0)
    X git rev-parse --verify --quiet "refs/heads/$Branch" | Out-Null
    if ($script:Code -ne 0) {
        $from = if ($hasRemote) { "origin/$Branch" } else { "origin/main" }
        X git checkout -q -B $Branch $from | Out-Null
    } else {
        X git checkout -q $Branch | Out-Null
        # スマホの GitHub から agent/dev に直接入れた修正などを取り込む
        if ($hasRemote) {
            X git merge -q --no-edit "origin/$Branch" | Out-Null
            if ($script:Code -ne 0) { X git merge --abort | Out-Null; Log "could not merge origin/$Branch" }
        }
    }
    # PR がマージ済み（main と中身が同じ）なら、main から始め直す
    X git diff --quiet origin/main HEAD | Out-Null
    if ($script:Code -eq 0) {
        X git reset -q --hard origin/main | Out-Null
    } else {
        X git merge -q --no-edit origin/main | Out-Null
        if ($script:Code -ne 0) {
            X git merge --abort | Out-Null
            $script:MergeNote = "main の新しい変更と agent/dev がぶつかっています。課題より先に ``git merge origin/main`` でぶつかった所を直してコミットしてください。"
            Log "merge with origin/main has conflicts; asking the agent to resolve"
        }
    }
}

# ---------------------------------------------------------------- PR
function Get-PR {
    if (-not $script:HasGh) { return $null }
    $j = X gh pr list --head $Branch --state open --json "number,url"
    if ($script:Code -ne 0 -or -not $j) { return $null }
    $prs = @($j | ConvertFrom-Json)
    if ($prs.Count -eq 0) { return $null }
    return $prs[0]
}

# PR に付いた新しいコメント（前回反映した時刻より後のもの）
function Get-Feedback {
    $pr = Get-PR
    if (-not $pr) { return "" }
    $script:Status.pr_url = $pr.url
    $st = Read-Json $StateFile
    $since = if ($st -and $st.feedback_since) { [datetime]$st.feedback_since } else { [datetime]"2000-01-01" }
    $items = @()
    $v = X gh pr view $pr.number --json "comments,reviews"
    if ($script:Code -eq 0 -and $v) {
        $v = $v | ConvertFrom-Json
        foreach ($c in $v.comments) { if ([datetime]$c.createdAt -gt $since) { $items += "- コメント：" + $c.body } }
        foreach ($r in $v.reviews) { if ($r.body -and [datetime]$r.submittedAt -gt $since) { $items += "- レビュー（" + $r.state + "）：" + $r.body } }
    }
    $ic = X gh api "repos/{owner}/{repo}/pulls/$($pr.number)/comments"
    if ($script:Code -eq 0 -and $ic) {
        foreach ($c in @($ic | ConvertFrom-Json)) {
            if ([datetime]$c.created_at -gt $since) { $items += "- " + $c.path + " の " + $c.line + " 行目へのコメント：" + $c.body }
        }
    }
    return ($items -join "`n")
}

function Section([string]$path, [string]$start, [string]$end, [int]$maxLines) {
    if (-not (Test-Path $path)) { return "" }
    $lines = [IO.File]::ReadAllLines($path, $Utf8)
    $on = $false; $out = @()
    foreach ($l in $lines) {
        if ($on -and $end -and $l.StartsWith($end)) { break }
        if ($on) { $out += $l }
        if (-not $on -and $l.StartsWith($start)) { $on = $true }
    }
    return (($out | Select-Object -First $maxLines) -join "`n").Trim()
}

function Update-PR {
    if (-not $script:HasGh) { Log "gh not found; skipped PR"; return }
    $commits = X git log --reverse "--format=- %s" "origin/main..HEAD"
    $gate = Read-Json (Join-Path $St "gate.json")
    $gateLines = if ($gate) { ($gate.steps | ForEach-Object { "- " + $_.summary }) -join "`n" } else { "" }
    $questions = Section (Join-Path $Root "agent\QUESTIONS.md") "## 未回答" "## 済み" 60
    $journal = Section (Join-Path $Root "agent\JOURNAL.md") "---" "" 80
    $body = @"
自律エージェント（agent/run.ps1）の作業をまとめた PR です。課題ごとにコミットを分けています。
よければマージしてください。直してほしい所は、この PR にコメントすると次の回に反映します。

## 課題（コミット）
$commits

## 品質チェック（最後の回）
$gateLines

## あなたへの質問（agent/QUESTIONS.md）
$questions

## 作業の記録（agent/JOURNAL.md の新しい方から）
$journal
"@
    $bf = Join-Path $St "pr_body.md"
    [IO.File]::WriteAllText($bf, $body, $Utf8)
    $pr = Get-PR
    if ($pr) {
        X gh pr edit $pr.number --body-file $bf | Out-Null
        $script:Status.pr_url = $pr.url
    } else {
        $url = X gh pr create --base main --head $Branch --title "Badminton Analytics：エージェントの作業" --body-file $bf
        if ($script:Code -eq 0) { $script:Status.pr_url = ($url -split "`n")[-1] } else { Log "gh pr create failed: $url" }
    }
}

# ---------------------------------------------------------------- 1 回分の指示
function Build-Prompt([string]$feedback) {
    $base = [IO.File]::ReadAllText((Join-Path $Root "agent\prompt.md"), $Utf8)
    $left = [int]($script:Deadline - (Get-Date)).TotalMinutes
    $how = if ($Once) { "手動で 1 回だけ" } else { "夜間の連続運転（終わるとすぐ次の回が始まる）" }
    $ctx = @("", "## 今回の状況", "",
        ("- 今：{0}。止める時刻：{1}（あと約 {2} 分）。" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $script:Deadline.ToString("HH:mm"), $left),
        ("- 止める時刻を過ぎたら（作業の区切りごとに ``date`` と ``ls .agent/WRAPUP`` で確かめる）、新しいことは始めず、今の作業を区切りのいい所まで終えて、品質チェック → 記録 → コミットして終わってください。課題が途中なら ROADMAP を ``[~]`` にし、JOURNAL に続きの手順を書きます。{0} を過ぎると強制的に止められ、コミットしていない変更は消えます。" -f $script:Deadline.AddMinutes($GraceMinutes).ToString("HH:mm")),
        ("- 動かし方：{0}" -f $how))
    if ($script:MergeNote) { $ctx += "- " + $script:MergeNote }
    if ($feedback) { $ctx += @("", "### あなたへのコメント（PR に付いた新しいもの。最優先）", $feedback) }
    if (Test-Path $FailureFile) {
        $ctx += @("", "### 前の回で品質チェックが通らず取り消した変更", [IO.File]::ReadAllText($FailureFile, $Utf8))
    }
    if (Test-Path $HistoryFile) {
        $recent = [IO.File]::ReadAllLines($HistoryFile, $Utf8) | Select-Object -Last 5 | ForEach-Object {
            try { $h = $_ | ConvertFrom-Json; "- {0} {1} → {2}：{3}" -f $h.start, $h.task, $h.result, $h.summary } catch { }
        }
        if ($recent) { $ctx += @("", "### 最近の回", ($recent -join "`n")) }
    }
    return $base + ($ctx -join "`n") + "`n"
}
# ---------------------------------------------------------------- 1 回分
function Invoke-Iteration {
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    $log = Join-Path $Logs "claude-$ts.jsonl"
    Prepare-Branch
    $before = X git rev-parse HEAD
    $feedback = Get-Feedback
    $pf = Join-Path $St "prompt-current.md"
    [IO.File]::WriteAllText($pf, (Build-Prompt $feedback), $Utf8)
    Set-Status @{ state = "running"; iteration = $script:Iter; task = ""; message = "Claude が作業しています"; log = $log }
    Log "iteration $($script:Iter): claude started (log $log)"

    $cargs = @("-p", "--settings", "`"$Settings`"", "--permission-mode", "acceptEdits", "--max-turns", "$MaxTurns",
        "--output-format", "stream-json", "--verbose")
    if ($Model) { $cargs += @("--model", $Model) }
    $p = Start-Process -FilePath $script:Claude -ArgumentList $cargs -WorkingDirectory $Root -NoNewWindow -PassThru `
        -RedirectStandardInput $pf -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $null = $p.Handle
    $killed = ""
    while (-not $p.HasExited) {
        $now = Get-Date
        if (Test-Path $StopFile) { $killed = "stop" } elseif ($now -ge $script:Deadline.AddMinutes($GraceMinutes)) { $killed = "deadline" }
        if ($killed) { & taskkill /PID $p.Id /T /F 2>&1 | Out-Null; break }
        if ($now -ge $script:Deadline -and -not (Test-Path $WrapFile)) {
            [IO.File]::WriteAllText($WrapFile, "止める時刻を過ぎました。区切りのいい所まで終えてコミットしてください。`n", $Utf8)
            Set-Status @{ state = "wrapping"; message = ("止める時刻を過ぎたので、区切りのいい所で終えてもらっています（遅くとも {0}）" -f $script:Deadline.AddMinutes($GraceMinutes).ToString("HH:mm")) }
            Log "iteration $($script:Iter): deadline reached; asked claude to wrap up"
        }
        Start-Sleep -Seconds 10
    }
    if (Test-Path $WrapFile) { Remove-Item $WrapFile }
    $p.WaitForExit()

    # Claude の最後の報告
    $res = $null
    if (Test-Path $log) {
        foreach ($l in [IO.File]::ReadAllLines($log, $Utf8)) { if ($l -match '^\{\s*"type"\s*:\s*"result"') { try { $res = $l | ConvertFrom-Json } catch { } } }
    }
    $text = if ($res) { [string]$res.result } else { "" }
    $task = if ($text -match '(?m)^TASK:\s*(.+?)\s*$') { $Matches[1] } else { "" }
    $result = if ($text -match '(?m)^RESULT:\s*(\w+)') { $Matches[1] } else { "" }
    $summary = if ($text -match '(?m)^SUMMARY:\s*(.+?)\s*$') { $Matches[1] } else { "" }
    $info = @{ start = $ts; task = $task; result = $result; summary = $summary; log = $log
        turns = $(if ($res) { $res.num_turns } else { $null }); killed = $killed; pushed = $false }

    # 利用枠の上限・起動の失敗
    if (-not $killed -and (-not $res -or $res.is_error)) {
        $errText = $text + "`n" + $(if (Test-Path "$log.err") { [IO.File]::ReadAllText("$log.err", $Utf8) } else { "" })
        if ($errText -match '(?i)usage limit|limit reached|hit your limit|rate.?limit') {
            $info.result = "limit"
            $script:LimitReset = if ($errText -match '\|(\d{10})') { [DateTimeOffset]::FromUnixTimeSeconds([long]$Matches[1]).LocalDateTime } else { (Get-Date).AddMinutes(30) }
        } else {
            $info.result = "error"
            $info.summary = (($errText.Trim() -split "`n") | Select-Object -Last 3) -join " "
        }
    }

    if ((X git rev-parse --abbrev-ref HEAD) -ne $Branch) { X git checkout -q -f $Branch | Out-Null }
    if (X git status --porcelain) { Save-Leftover $(if ($killed) { $killed } else { "uncommitted" }) }
    $after = X git rev-parse HEAD
    if ($after -eq $before) {
        Log "iteration $($script:Iter): no commits ($($info.result))"
        Add-History $info
        return $info
    }

    # 品質チェック（Claude が最後の状態で通していれば、やり直さない）
    Set-Status @{ state = "gate"; task = $task; message = "品質チェックをしています" }
    $gate = Read-Json (Join-Path $St "gate.json")
    $tree = X git rev-parse "HEAD^{tree}"
    $gout = ""
    if ($gate -and $gate.ok -and $gate.tree -eq $tree) {
        $gateOk = $true
    } else {
        $gout = X $Py "agent\eval_gate.py"
        $gateOk = ($script:Code -eq 0)
        [IO.File]::WriteAllText((Join-Path $Logs "gate-$ts.log"), $gout, $Utf8)
    }
    if ($gateOk) {
        $out = X git push --force-with-lease origin "${Branch}:${Branch}"
        if ($script:Code -eq 0) { $info.pushed = $true; Update-PR } else { Log "push failed: $out" }
        if (Test-Path $FailureFile) { Remove-Item $FailureFile }
        Save-Json $StateFile @{ feedback_since = (Get-Date).ToUniversalTime().ToString("o") }
        Notify ("[Badminton Analytics] {0}：{1}" -f $task, $summary)
    } else {
        $dir = Join-Path $Failed $ts
        X git format-patch -q "$before..HEAD" -o $dir | Out-Null
        X git reset -q --hard $before | Out-Null
        $info.result = "gate_failed"
        $tail = (($gout -split "`n") | Select-Object -Last 40) -join "`n"
        [IO.File]::WriteAllText($FailureFile, "課題：$task`nパッチ：$dir`n`n``````n$tail`n``````n", $Utf8)
        Log "iteration $($script:Iter): gate failed; changes saved to $dir"
        Notify ("[Badminton Analytics] {0}：品質チェックが通らず取り消しました" -f $task)
    }
    Log "iteration $($script:Iter): $($info.result) $task $summary"
    Add-History $info
    return $info
}

# ---------------------------------------------------------------- 本体
if ($OnRequest -and -not (Test-Path $RequestFile)) { exit 0 }
$lk = Read-Json $LockFile
if ($lk -and (Get-Process -Id $lk.pid -ErrorAction SilentlyContinue)) { Log "already running (pid $($lk.pid))"; exit 0 }
if ($OnRequest) { Remove-Item $RequestFile; $Once = $true }
$Mode = if ($Once) { "once" } else { "window" }

$now = Get-Date
if ($Until) {
    $script:Deadline = [datetime]::ParseExact($Until, "HH:mm", $null)
    if ($script:Deadline -le $now) { Log "past $Until; nothing to do"; exit 0 }
} else {
    $script:Deadline = $now.AddHours($Hours)
}
Save-Json $LockFile @{ pid = $PID; started = (Get-Date -Format s) }
foreach ($f in @($StopFile, $WrapFile)) { if (Test-Path $f) { Remove-Item $f } }
$script:Status = [ordered]@{
    state = "starting"; mode = $Mode; pid = $PID; started = (Get-Date -Format s); deadline = $script:Deadline.ToString("s")
    iteration = 0; task = ""; message = ""; log = ""; pr_url = ""; updated = ""
}

# 動いている間はスリープしない
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null

try {
    $script:Claude = @(Get-Command claude.exe, claude.cmd -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)[0]
    if (-not $script:Claude -and (Test-Path "$env:USERPROFILE\.local\bin\claude.exe")) { $script:Claude = "$env:USERPROFILE\.local\bin\claude.exe" }
    $script:HasGh = [bool](Get-Command gh -ErrorAction SilentlyContinue)
    if (-not $script:Claude) { throw "claude が見つかりません（agent/README.md の準備を見てください）" }
    if (-not (Test-Path $Py)) { throw ".venv がありません（agent/README.md の準備を見てください）" }
    Set-Status @{ state = "running"; message = "準備しています" }
    Log "agent started: mode=$Mode deadline=$($script:Deadline) claude=$($script:Claude)"

    $script:Iter = 0; $idle = 0; $errors = 0
    while ($true) {
        if (Test-Path $StopFile) { Log "stop requested"; break }
        if ($script:Iter -gt 0 -and (Get-Date) -ge $script:Deadline.AddMinutes(-30)) { Log "less than 30 minutes left"; break }
        $script:Iter++
        $info = Invoke-Iteration
        if ($info.killed) { break }
        switch ($info.result) {
            "no_task" { Log "roadmap has nothing to do"; Notify "[Badminton Analytics] ROADMAP の課題がなくなりました"; $idle = $IdleLimit }
            "limit" {
                if ($Once -or $script:LimitReset -ge $script:Deadline) { Log "usage limit reached"; $idle = $IdleLimit }
                else {
                    Set-Status @{ state = "waiting_limit"; message = ("利用枠の上限です。{0} まで待ちます" -f $script:LimitReset.ToString("HH:mm")) }
                    Log "usage limit; waiting until $($script:LimitReset)"
                    while ((Get-Date) -lt $script:LimitReset.AddMinutes(2) -and -not (Test-Path $StopFile)) { Start-Sleep -Seconds 30 }
                }
            }
            "error" { $errors++; if ($errors -ge 3) { Log "3 errors in a row"; $idle = $IdleLimit } elseif (-not $Once) { Start-Sleep -Seconds 300 } }
            default { $errors = 0; if ($info.pushed) { $idle = 0 } else { $idle++ } }
        }
        if ($Once -or $idle -ge $IdleLimit) { break }
    }
    Set-Status @{ state = "idle"; message = "止まっています"; task = "" }
} catch {
    Log "error: $_"
    Set-Status @{ state = "error"; message = "$_" }
} finally {
    [Win32.Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    if (Test-Path $LockFile) { Remove-Item $LockFile }
    if (Test-Path $StopFile) { Remove-Item $StopFile }
    Log "agent finished"
}
