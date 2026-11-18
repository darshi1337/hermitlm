# Register a Windows scheduled task that starts the resume-safe retrain at
# every logon. Run this PowerShell in an elevated (Administrator) session.
$ErrorActionPreference = "Stop"
$taskName = "HermitLM Retrain (XPU)"
$root = Split-Path -Parent $PSScriptRoot
$script = Join-Path $root "scripts\retrain_resume.ps1"

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`"" `
    -WorkingDirectory $root
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 2)

try {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force -ErrorAction Stop
} catch {
    Write-Host "Highest run level unavailable (run this script as Administrator for best results); retrying as Limited."
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -RunLevel Limited -Force
}
Write-Host "Registered '$taskName'. It runs at logon and skips already-completed steps."
