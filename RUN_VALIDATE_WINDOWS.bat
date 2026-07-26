@echo off
REM ===================================================================
REM  MNQ - IS THE 0.62 RESULT REAL?
REM
REM  The sweep found a configuration making +$4,140 at a 0.62 entry
REM  threshold, positive in both sample halves. That is encouraging but
REM  it is NOT proof, for three reasons:
REM
REM    - only 91 trades (45 in the out-of-sample half)
REM    - the profit is lopsided: $781 in one half, $3,478 in the other
REM    - 216 configurations were scored, and with that many, some pass
REM      the both-halves filter by pure chance
REM
REM  This script attacks the result instead of celebrating it:
REM
REM    1. THRESHOLD CURVE - a real edge strengthens gradually as the
REM       confidence bar rises. Profit at exactly one threshold with
REM       losses either side is a spike in noise.
REM
REM    2. QUARTERLY BREAKDOWN - two halves can hide one lucky quarter.
REM
REM    3. PERMUTATION TEST - the decisive one. It keeps the stops,
REM       targets, costs and market, and destroys ONLY the link between
REM       model confidence and outcome, several hundred times. If
REM       shuffled predictions make similar money, the profit came from
REM       the trade management, not the model.
REM
REM    4. MULTIPLE-COMPARISON ACCOUNTING - how many configurations
REM       would pass the both-halves filter by luck alone.
REM
REM  Reuses the models you already trained. Takes 10-25 minutes.
REM ===================================================================

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [X] Setup has not been run. Double-click RUN_ME_WINDOWS.bat first.
    pause
    exit /b 1
)

set VPY=.venv\Scripts\python.exe

echo.
echo ============================================================
echo   Validating the 0.62 configuration
echo ============================================================
echo.

%VPY% -m mnq.cli validate --profile wide --min-probability 0.62 --permutations 300

echo.
echo ============================================================
echo   DONE
echo.
echo   Read the VERDICT at the bottom. The single most important
echo   number is the permutation p-value:
echo.
echo     p below 0.05  = unlikely to be luck, worth paper trading
echo     p 0.05 to 0.20 = weak, not established
echo     p above 0.20  = the model adds nothing, do not trade it
echo.
echo   Send Claude the whole output from VALIDATION down.
echo ============================================================
echo.
pause
