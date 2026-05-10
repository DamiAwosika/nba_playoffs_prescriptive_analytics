# Daily refresh: run ETL, predictions, archive props, then push to Railway.
# Schedule with Windows Task Scheduler to run daily (e.g. 11 AM ET).
# All output is logged to logs/daily_refresh_YYYY-MM-DD.log
#
# Task Scheduler action:
#   Program:   powershell.exe
#   Arguments: -ExecutionPolicy Bypass -File C:\dev\nba\scripts\daily_refresh.ps1
#   Start in:  C:\dev\nba

Set-Location "C:\dev\nba"

New-Item -ItemType Directory -Path "C:\dev\nba\logs" -Force | Out-Null
$logFile = "C:\dev\nba\logs\daily_refresh_$(Get-Date -Format 'yyyy-MM-dd').log"

& {
    Write-Output "=========================================="
    Write-Output "NBA Daily Refresh -- $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Output "=========================================="

    & "C:\Users\damia\anaconda3\shell\condabin\conda-hook.ps1"
    conda activate nba_ML

    Write-Output "`n--- 1/5  ETL: ingest games ---"
    python scripts/run_etl.py

    Write-Output "`n--- 2/5  Archive pre-game odds ---"
    python scripts/archive_odds.py

    Write-Output "`n--- 3/5  Archive pre-game player props ---"
    python scripts/archive_player_props.py
    
    Write-Output "`n--- 4/6  Train Game Probability Models ---"
    python nba_ml\training\train.py --tune --n-iter 100


    Write-Output "`n--- 5/6  Refresh injuries + playoff predictions ---"
    python scripts/predict_playoffs.py

    Write-Output "`n--- 6/6  Push seed data to Railway ---"
    Copy-Item nba.db seed\nba.db -Force
    git add -f seed/nba.db
    git commit -m "Daily seed data refresh $(Get-Date -Format 'yyyy-MM-dd')"
    git push

    Write-Output "`n=========================================="
    Write-Output "Completed -- $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    Write-Output "=========================================="
} *>&1 | Out-File -Encoding utf8 -Append $logFile
