# Badminton Analytics の自律エージェント：Claude Code を繰り返し起動して、agent/ROADMAP.md の課題を 1 つずつ進める。
#
#   run.ps1 -Until 12:00   毎晩 0:00 にタスク スケジューラから。12:00 を過ぎたら次の回は始めない
#   run.ps1 -Once          課題を 1 つだけ進める（午後に手で動かすとき。既定は 3 時間が目安）
#
# 止める時刻になったら .agent/WRAPUP を置き、Claude には今の作業を区切りのいい所まで終えてコミットしてもらう。
# それでも終わらないときは、止める時刻 + GraceMinutes で打ち切る（午後の利用枠を使いすぎないため）。
#   run.ps1 -OnRequest     .agent/REQUEST（GUI の「1 つ進める」）があるときだけ -Once で動く。5 分おきに起動される
#
# 夜間の連続運転は、.agent/PAUSE（GUI の「一時停止」）があるあいだと、PC がキーボードやマウスで使われているあいだは
# 一時停止する。作業中の回には区切りのいい所まで終えてもらい、再開したら次の回から続ける。
#
# 変更は agent/dev ブランチにためて push し、main への PR を 1 つ開いておく（人が見てマージする）。
# 品質チェック（agent/eval_gate.py）が通らない変更は取り消し、.agent/failed にパッチとして残す。
# PR へのコメントと、自分が立てた Issue は、次の回の指示として Claude に渡す（他人の書き込みは読まない）。
# 対応した Issue には返事をし、済んだものは閉じる。
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
$PauseFile = Join-Path $St "PAUSE"
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
# JSON の配列を 1 件ずつの配列にする（PowerShell 5.1 の ConvertFrom-Json は配列をまとめて 1 つで返すため）
function From-JsonList([string]$j) {
    if (-not $j) { return @() }
    $x = $j | ConvertFrom-Json
    return @($x | ForEach-Object { $_ })
}
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

# ---------------------------------------------------------------- 一時停止
# 手動：.agent/PAUSE があるあいだ。
# 自動：.agent/config.json の auto_pause（既定はオン）。3 分ほど続けて操作されたら「使っている」とみなし、
#       auto_pause_idle_minutes 分（既定 10 分）操作がなければ再開する。猫がマウスに触れた程度では止めない。
# -Once（「1 つ進める」）は人が頼んだものなので、どちらでも止めない。
$script:BusySince = $null
$script:AutoPaused = $false
$script:ResumeMinutes = 10
function Get-PauseReason {
    if ($Once) { return "" }
    if (Test-Path $PauseFile) { return "manual" }
    $cfg = Read-Json (Join-Path $St "config.json")
    if ($cfg -and $cfg.auto_pause -eq $false) { $script:BusySince = $null; $script:AutoPaused = $false; return "" }
    $script:ResumeMinutes = if ($cfg -and $cfg.auto_pause_idle_minutes -gt 0) { [double]$cfg.auto_pause_idle_minutes } else { 10 }
    $idleSec = [Win32.Input]::IdleMs() / 1000.0
    $now = Get-Date
    if ($idleSec -lt 120) {
        if (-not $script:BusySince) { $script:BusySince = $now.AddSeconds(-$idleSec) }
    } else {
        $script:BusySince = $null
    }
    if ($script:AutoPaused) {
        if ($idleSec -ge $script:ResumeMinutes * 60) { $script:AutoPaused = $false; $script:BusySince = $null }
    } elseif ($script:BusySince -and ($now - $script:BusySince).TotalMinutes -ge 3) {
        $script:AutoPaused = $true
    }
    if ($script:AutoPaused) { return "active" }
    return ""
}

# 一時停止のあいだ待つ。止める・時刻切れのときは $false
function Wait-WhilePaused {
    $reason = Get-PauseReason
    if (-not $reason) { return $true }
    Log "paused ($reason)"
    while ($reason) {
        $msg = if ($reason -eq "manual") { "一時停止しています（「再開」を押すと続きから始めます）" } else {
            "PC が使われているので待っています（{0} 分操作がなければ再開します）" -f $script:ResumeMinutes }
        Set-Status @{ state = "paused"; paused = $reason; message = $msg; task = "" }
        if (Test-Path $StopFile) { return $false }
        if ((Get-Date) -ge $script:Deadline.AddMinutes(-30)) { Log "still paused with less than 30 minutes left"; return $false }
        Start-Sleep -Seconds 30
        $reason = Get-PauseReason
    }
    Log "resumed"
    Set-Status @{ state = "running"; paused = ""; message = "再開しました" }
    return $true
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

# 人だけが直すファイル。claude-settings.json で書き換えを禁じているが、python などからは書けてしまうので、
# 回の終わりに確かめる。前の回の中身か、main の中身（main から取り込んだだけ）と同じなら構わない
$Protected = @("agent/run.ps1", "agent/setup.ps1", "agent/claude-settings.json", "agent/eval_gate.py", "agent/baseline.json")
function Get-ProtectedChanges([string]$before) {
    $bad = @()
    foreach ($f in $Protected) {
        $now = X git rev-parse --verify --quiet "HEAD:$f"
        if ($now -eq (X git rev-parse --verify --quiet "${before}:$f")) { continue }
        if ($now -eq (X git rev-parse --verify --quiet "origin/main:$f")) { continue }
        $bad += $f
    }
    return $bad
}

# ---------------------------------------------------------------- PR
function Get-PR {
    if (-not $script:HasGh) { return $null }
    $j = X gh pr list --head $Branch --state open --json "number,url"
    if ($script:Code -ne 0 -or -not $j) { return $null }
    $prs = @(From-JsonList $j)
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
    # 自分（gh でログインしている人）の書き込みだけを読む。公開リポジトリで他人の書き込みを指示として扱わないため
    $v = X gh pr view $pr.number --json "comments,reviews"
    if ($script:Code -eq 0 -and $v) {
        $v = $v | ConvertFrom-Json
        foreach ($c in $v.comments) { if ($c.author.login -eq $script:Me -and [datetime]$c.createdAt -gt $since) { $items += "- コメント：" + $c.body } }
        foreach ($r in $v.reviews) { if ($r.author.login -eq $script:Me -and $r.body -and [datetime]$r.submittedAt -gt $since) { $items += "- レビュー（" + $r.state + "）：" + $r.body } }
    }
    $ic = X gh api "repos/{owner}/{repo}/pulls/$($pr.number)/comments"
    if ($script:Code -eq 0 -and $ic) {
        foreach ($c in (From-JsonList $ic)) {
            if ($c.user.login -eq $script:Me -and [datetime]$c.created_at -gt $since) { $items += "- " + $c.path + " の " + $c.line + " 行目へのコメント：" + $c.body }
        }
    }
    return ($items -join "`n")
}

# ---------------------------------------------------------------- Issue
# 自分が立てた、開いている Issue（古い順）。エージェントの質問への答え待ちのものは飛ばす
$AgentMark = "🤖 エージェント"
$QuestionMark = "🤖 エージェント（質問）"
function Clip([string]$s, [int]$n) { if (-not $s) { return "" }; if ($s.Length -gt $n) { return $s.Substring(0, $n) + "…" }; return $s }

function Get-Issues {
    $script:IssueNums = @()
    if (-not $script:HasGh -or -not $script:Me) { return "" }
    $j = X gh issue list --state open --author $script:Me --json "number,title,body,author,comments,createdAt" --limit 20
    if ($script:Code -ne 0 -or -not $j) { return "" }
    $items = @()
    foreach ($i in ((From-JsonList $j) | Where-Object { $_.author.login -eq $script:Me } | Sort-Object createdAt)) {
        $mine = @($i.comments | Where-Object { $_.author.login -eq $script:Me })
        if ($mine.Count -gt 0 -and ([string]$mine[-1].body).StartsWith($QuestionMark)) { continue }
        $script:IssueNums += [int]$i.number
        $lines = @(("#{0} {1}" -f $i.number, $i.title), (Clip $i.body 3000))
        foreach ($c in $mine) {
            $who = if (([string]$c.body).StartsWith($AgentMark)) { "エージェントの返事" } else { "あなたのコメント" }
            $lines += "- {0}：{1}" -f $who, (Clip $c.body 1500)
        }
        $items += ($lines -join "`n")
    }
    return ($items -join "`n`n")
}

# Claude が対応した Issue に返事をし、済んだものは閉じる（返事の中身は .agent/issue_reply.md）
function Send-IssueReply($info, [string]$text, [bool]$pushed) {
    $f = Join-Path $St "issue_reply.md"
    if ($text -notmatch '(?m)^ISSUE:\s*#?(\d+)') { if (Test-Path $f) { Remove-Item $f }; return }
    $n = [int]$Matches[1]
    if ($script:IssueNums -notcontains $n) { Log "ignored reply to issue #$n (not in this run's list)"; if (Test-Path $f) { Remove-Item $f }; return }
    $st = if ($text -match '(?m)^ISSUE_STATUS:\s*(\w+)') { $Matches[1] } else { "partial" }
    $reply = if (Test-Path $f) { [IO.File]::ReadAllText($f, $Utf8).Trim() } else { $info.summary }
    $head = switch ($st) { "done" { "$AgentMark（対応しました）" } "question" { $QuestionMark } default { "$AgentMark（途中）" } }
    $body = "$head`n`n$reply"
    if ($pushed -and $script:Status.pr_url) { $body += "`n`n変更は $($script:Status.pr_url) に入っています（マージすると main に入ります）。" }
    [IO.File]::WriteAllText($f, $body, $Utf8)
    $out = X gh issue comment $n --body-file $f
    if ($script:Code -ne 0) { Log "issue comment failed: $out" }
    elseif ($st -eq "done") { X gh issue close $n | Out-Null }
    Remove-Item $f
    $info.issue = $n
    Log "replied to issue #$n ($st)"
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
function Build-Prompt([string]$feedback, [string]$issues) {
    $base = [IO.File]::ReadAllText((Join-Path $Root "agent\prompt.md"), $Utf8)
    $left = [int]($script:Deadline - (Get-Date)).TotalMinutes
    $how = if ($Once) { "手動で 1 回だけ" } else { "夜間の連続運転（終わるとすぐ次の回が始まる）" }
    $ctx = @("", "## 今回の状況", "",
        ("- 今：{0}。止める時刻：{1}（あと約 {2} 分）。" -f (Get-Date -Format "yyyy-MM-dd HH:mm"), $script:Deadline.ToString("HH:mm"), $left),
        ("- 止める時刻を過ぎたら（作業の区切りごとに ``date`` と ``ls .agent/WRAPUP`` で確かめる）、新しいことは始めず、今の作業を区切りのいい所まで終えて、品質チェック → 記録 → コミットして終わってください。課題が途中なら ROADMAP を ``[~]`` にし、JOURNAL に続きの手順を書きます。{0} を過ぎると強制的に止められ、コミットしていない変更は消えます。" -f $script:Deadline.AddMinutes($GraceMinutes).ToString("HH:mm")),
        ("- 動かし方：{0}" -f $how))
    if ($script:MergeNote) { $ctx += "- " + $script:MergeNote }
    if ($feedback) { $ctx += @("", "### あなたへのコメント（PR に付いた新しいもの。最優先）", $feedback) }
    if ($issues) { $ctx += @("", "### Issue での頼みごと（古い順。PR へのコメントの次に優先し、1 回に 1 つ）", $issues) }
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
    $issues = Get-Issues
    $replyFile = Join-Path $St "issue_reply.md"
    if (Test-Path $replyFile) { Remove-Item $replyFile }
    $pf = Join-Path $St "prompt-current.md"
    [IO.File]::WriteAllText($pf, (Build-Prompt $feedback $issues), $Utf8)
    Set-Status @{ state = "running"; iteration = $script:Iter; task = ""; message = "Claude が作業しています"; log = $log }
    Log "iteration $($script:Iter): claude started (log $log)"

    $cargs = @("-p", "--settings", "`"$Settings`"", "--permission-mode", "acceptEdits", "--max-turns", "$MaxTurns",
        "--output-format", "stream-json", "--verbose")
    if ($Model) { $cargs += @("--model", $Model) }
    $p = Start-Process -FilePath $script:Claude -ArgumentList $cargs -WorkingDirectory $Root -NoNewWindow -PassThru `
        -RedirectStandardInput $pf -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    $null = $p.Handle
    $killed = ""
    $pausedWrap = $false
    while (-not $p.HasExited) {
        $now = Get-Date
        if (Test-Path $StopFile) { $killed = "stop" } elseif ($now -ge $script:Deadline.AddMinutes($GraceMinutes)) { $killed = "deadline" }
        if ($killed) { & taskkill /PID $p.Id /T /F 2>&1 | Out-Null; break }
        if ($now -ge $script:Deadline -and -not (Test-Path $WrapFile)) {
            [IO.File]::WriteAllText($WrapFile, "止める時刻を過ぎました。区切りのいい所まで終えてコミットしてください。`n", $Utf8)
            Set-Status @{ state = "wrapping"; message = ("止める時刻を過ぎたので、区切りのいい所で終えてもらっています（遅くとも {0}）" -f $script:Deadline.AddMinutes($GraceMinutes).ToString("HH:mm")) }
            Log "iteration $($script:Iter): deadline reached; asked claude to wrap up"
        }
        if (-not (Test-Path $WrapFile)) {
            $reason = Get-PauseReason
            if ($reason) {
                [IO.File]::WriteAllText($WrapFile, "一時停止します（PC が使われ始めた、または「一時停止」が押された）。区切りのいい所まで終えてコミットしてください。`n", $Utf8)
                $pausedWrap = $true
                Set-Status @{ state = "wrapping"; paused = $reason; message = "一時停止のため、区切りのいい所で終えてもらっています" }
                Log "iteration $($script:Iter): pause ($reason); asked claude to wrap up"
            }
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
        turns = $(if ($res) { $res.num_turns } else { $null }); killed = $killed; pushed = $false; paused = $pausedWrap }

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
        # 質問への答えなど、コードを変えずに済む頼みごともある
        if (-not $killed -and $info.result -notin @("error", "limit")) { Send-IssueReply $info $text $false }
        Log "iteration $($script:Iter): no commits ($($info.result))"
        Add-History $info
        return $info
    }

    # 守るファイルが書き換えられていたら、この回の変更は入れない
    $touched = @(Get-ProtectedChanges $before)
    if ($touched.Count -gt 0) {
        $dir = Join-Path $Failed "$ts-protected"
        X git format-patch -q "$before..HEAD" -o $dir | Out-Null
        X git reset -q --hard $before | Out-Null
        $info.result = "protected"
        $info.summary = "書き換えてはいけないファイル（{0}）が変わっていたので、取り消しました" -f ($touched -join "、")
        [IO.File]::WriteAllText($FailureFile, "課題：$task`nパッチ：$dir`n`n$($info.summary)。これらのファイルは人だけが直します。`n", $Utf8)
        Log "iteration $($script:Iter): protected files changed ($($touched -join ', ')); changes saved to $dir"
        Notify ("[Badminton Analytics] {0}：{1}" -f $task, $info.summary)
        Add-History $info
        return $info
    }

    # 品質チェックは毎回ここでやり直す（.agent/gate.json は Claude の側からも書けるので、その結果は使わない）
    Set-Status @{ state = "gate"; task = $task; message = "品質チェックをしています" }
    $gout = X $Py "agent\eval_gate.py"
    $gateOk = ($script:Code -eq 0)
    [IO.File]::WriteAllText((Join-Path $Logs "gate-$ts.log"), $gout, $Utf8)
    if ($gateOk) {
        $out = X git push --force-with-lease origin "${Branch}:${Branch}"
        if ($script:Code -eq 0) { $info.pushed = $true; Update-PR; Send-IssueReply $info $text $true } else { Log "push failed: $out" }
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
    iteration = 0; task = ""; message = ""; log = ""; pr_url = ""; paused = ""; updated = ""
}

# 動いている間はスリープしない
Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[Win32.Power]::SetThreadExecutionState([uint32]2147483649) | Out-Null
# 最後にキーボードやマウスが操作されてからの時間（ミリ秒）
Add-Type -Namespace Win32 -Name Input -MemberDefinition @'
[StructLayout(LayoutKind.Sequential)] struct LASTINPUTINFO { public uint cbSize; public uint dwTime; }
[DllImport("user32.dll")] static extern bool GetLastInputInfo(ref LASTINPUTINFO plii);
public static uint IdleMs() {
    var li = new LASTINPUTINFO(); li.cbSize = (uint)Marshal.SizeOf(li);
    if (!GetLastInputInfo(ref li)) return uint.MaxValue;
    return (uint)Environment.TickCount - li.dwTime;
}
'@

try {
    $script:Claude = @(Get-Command claude.exe, claude.cmd -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)[0]
    if (-not $script:Claude -and (Test-Path "$env:USERPROFILE\.local\bin\claude.exe")) { $script:Claude = "$env:USERPROFILE\.local\bin\claude.exe" }
    $script:HasGh = [bool](Get-Command gh -ErrorAction SilentlyContinue)
    $script:Me = if ($script:HasGh) { (X gh api user --jq .login).Trim() } else { "" }
    if ($script:HasGh -and $script:Code -ne 0) { Log "gh is not logged in; PR comments and issues are skipped"; $script:Me = "" }
    if (-not $script:Claude) { throw "claude が見つかりません（agent/README.md の準備を見てください）" }
    if (-not (Test-Path $Py)) { throw ".venv がありません（agent/README.md の準備を見てください）" }
    Set-Status @{ state = "running"; message = "準備しています" }
    Log "agent started: mode=$Mode deadline=$($script:Deadline) claude=$($script:Claude)"

    $script:Iter = 0; $idle = 0; $errors = 0
    while ($true) {
        if (Test-Path $StopFile) { Log "stop requested"; break }
        if (-not (Wait-WhilePaused)) { break }
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
            default { $errors = 0; if ($info.pushed) { $idle = 0 } elseif (-not $info.paused) { $idle++ } }
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
