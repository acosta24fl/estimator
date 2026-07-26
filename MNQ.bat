@echo off
REM ===================================================================
REM  MNQ TRADING SYSTEM - single entry point
REM
REM  Double-click this. It sets itself up if needed, then shows a menu.
REM  There is no separate setup step any more.
REM
REM  Everything expensive - the Python environment, the downloaded price
REM  history, the trained models - is kept OUTSIDE this folder, in:
REM
REM      %USERPROFILE%\mnq-data
REM
REM  So when you get a new version of the code, just extract it and run
REM  this file. It reuses what is already there: no reinstall, no
REM  re-download, no retraining. The code folder is disposable; the data
REM  folder is the one that matters.
REM ===================================================================

cd /d "%~dp0"

if "%MNQ_HOME%"=="" set "MNQ_HOME=%USERPROFILE%\mnq-data"
set "VENV=%MNQ_HOME%\venv"
set "VPY=%VENV%\Scripts\python.exe"
set "REQ_MARKER=%MNQ_HOME%\installed-requirements.txt"

if not exist "%MNQ_HOME%" mkdir "%MNQ_HOME%"

echo.
echo ============================================================
echo   MNQ Trading System
echo ============================================================
echo   Code   : %CD%
echo   Data   : %MNQ_HOME%
echo.

REM ---- bootstrap: only does work when something is actually missing ----
if not exist "%VPY%" goto :install
if not exist "%REQ_MARKER%" goto :install
fc /b "%REQ_MARKER%" requirements.txt >nul 2>&1
if errorlevel 1 goto :install
goto :ready

:install
echo   Setting up (first run, or the requirements changed).
echo   This takes a few minutes once, then never again.
echo.

if not exist "%VPY%" (
    set "PYCMD="
    py --version >nul 2>&1 && set "PYCMD=py"
    if not defined PYCMD python --version >nul 2>&1 && set "PYCMD=python"
    if not defined PYCMD (
        echo   [X] Python is not installed, or was installed without
        echo       "Add python.exe to PATH" ticked.
        echo.
        echo       Install it from https://www.python.org/downloads/
        echo       and TICK "Add python.exe to PATH" on the first screen.
        echo.
        pause
        exit /b 1
    )
    echo   Creating the Python environment...
    %PYCMD% -m venv "%VENV%"
    if errorlevel 1 (
        echo   [X] Could not create the environment.
        pause
        exit /b 1
    )
)

echo   Installing packages...
"%VPY%" -m pip install --upgrade pip --quiet
"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo   [X] Package installation failed. Check your internet connection.
    pause
    exit /b 1
)
copy /y requirements.txt "%REQ_MARKER%" >nul
echo   Setup complete.
echo.

:ready
REM ---- show what already exists, so nothing is repeated needlessly ----
set "HAVE_DATA=no"
set "HAVE_MODEL=no"
if exist "%MNQ_HOME%\artifacts\data\MNQF_1h.csv" set "HAVE_DATA=yes"
if exist "%MNQ_HOME%\artifacts\models\ensemble.joblib" set "HAVE_MODEL=yes"

:menu
echo.
echo ------------------------------------------------------------
echo   Price data downloaded : %HAVE_DATA%
echo   Models trained        : %HAVE_MODEL%
echo ------------------------------------------------------------
echo.
echo   1  Update price data        (run weekly - builds history)
echo   2  Compare configurations   (the 4-way experiment)
echo   3  Train + backtest         (train, price it, sweep thresholds)
echo   4  Validate a result        (permutation test - is it real?)
echo   5  Cross-instrument test    (train on ES/YM/RTY, predict MNQ)
echo   6  Show status
echo   7  Run the tests
echo.
echo   0  Exit
echo.
set /p CHOICE=  Choose a number:

if "%CHOICE%"=="1" goto :do_fetch
if "%CHOICE%"=="2" goto :do_experiment
if "%CHOICE%"=="3" goto :do_backtest
if "%CHOICE%"=="4" goto :do_validate
if "%CHOICE%"=="5" goto :do_crossval
if "%CHOICE%"=="6" goto :do_status
if "%CHOICE%"=="7" goto :do_tests
if "%CHOICE%"=="0" exit /b 0
echo   Not a valid choice.
goto :menu

:do_fetch
echo.
echo   Downloading MNQ history and the cross-asset basket...
echo   Each run merges into the cache, so weekly runs grow your history
echo   past Yahoo's 60-day intraday limit.
echo.
"%VPY%" -m mnq.cli fetch --profile wide
set "HAVE_DATA=yes"
goto :done

:do_experiment
echo.
echo   Comparing 4 configurations (timeframe x cross-asset).
echo   Takes 20-60 minutes.
echo.
"%VPY%" -m mnq.cli experiment
goto :done

:do_backtest
echo.
echo   Training the wide profile, then backtesting and sweeping.
echo   Takes 30-90 minutes.
echo.
"%VPY%" -m mnq.cli train --profile wide
if errorlevel 1 goto :done
"%VPY%" -m mnq.cli backtest --profile wide
"%VPY%" -m mnq.cli sweep --profile wide --min-trades 15
set "HAVE_MODEL=yes"
echo.
echo   Look at is_net_usd and oos_net_usd. BOTH must be positive -
echo   and even then, run option 4 before believing it.
goto :done

:do_validate
echo.
set "THRESH="
set /p THRESH=  Entry threshold to test [press Enter for 0.62]:
if "%THRESH%"=="" set "THRESH=0.62"
echo.
echo   Stress-testing threshold %THRESH% with 300 permutations...
echo.
"%VPY%" -m mnq.cli validate --profile wide --min-probability %THRESH% --permutations 300
echo.
echo   The p-value decides it:
echo     below 0.05   real enough to paper trade
echo     0.05 - 0.20  not established
echo     above 0.20   the model adds nothing - do not trade it
goto :done

:do_crossval
echo.
echo   Training on ES, YM and RTY - never on MNQ - then predicting MNQ.
echo.
echo   This is the strongest test available. There are no shared bars, so
echo   a pattern that transfers is a property of index futures rather
echo   than something memorised about MNQ. It also triples the training
echo   data using symbols you already download.
echo.
echo   Takes 30-60 minutes.
echo.
"%VPY%" -m mnq.cli crossval
echo.
echo   AUC 0.55+ = the edge transfers, and pooled training is worth doing.
echo   AUC near 0.50 = it does not, and the earlier result was noise.
goto :done

:do_status
echo.
"%VPY%" -m mnq.cli status
goto :done

:do_tests
echo.
"%VPY%" -m pytest tests -q
goto :done

:done
echo.
echo ------------------------------------------------------------
echo   Finished. Your data and models are kept in:
echo   %MNQ_HOME%
echo ------------------------------------------------------------
pause
goto :menu
