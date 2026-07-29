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

REM  NOTE: the Python detection below deliberately uses labels rather than
REM  an if ( ... ) block. Inside a parenthesised block cmd.exe expands every
REM  %VAR% once, when it parses the whole block - so a variable set inside
REM  the block still reads as its old (here: empty) value further down it.
REM  That silently turned "%PYCMD% -m venv" into "-m venv". Keep it flat.

if exist "%VPY%" goto :have_venv

set "PYCMD="
py --version >nul 2>&1
if not errorlevel 1 set "PYCMD=py"
if defined PYCMD goto :got_python

python --version >nul 2>&1
if not errorlevel 1 set "PYCMD=python"
if defined PYCMD goto :got_python

python3 --version >nul 2>&1
if not errorlevel 1 set "PYCMD=python3"
if defined PYCMD goto :got_python

echo   [X] Python is not installed, or was installed without
echo       "Add python.exe to PATH" ticked.
echo.
echo       Install it from https://www.python.org/downloads/
echo       and TICK "Add python.exe to PATH" on the first screen.
echo.
pause
exit /b 1

:got_python
echo   Creating the Python environment...
%PYCMD% -m venv "%VENV%"
if errorlevel 1 goto :venv_failed
if not exist "%VPY%" goto :venv_failed
goto :have_venv

:venv_failed
echo.
echo   [X] Could not create the environment using "%PYCMD%".
echo.
echo       If you installed Python from the Microsoft Store, install it
echo       from https://www.python.org/downloads/ instead - the Store
echo       build cannot always create virtual environments.
echo.
pause
exit /b 1

:have_venv
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
echo   3  Train + backtest         (pooled 4-instrument - the main path)
echo   4  Validate a result        (permutation test - is it real?)
echo   5  Cross-instrument test    (train on ES/YM/RTY, predict MNQ)
echo   6  Ingest purchased history  (real vendor data - the big one)
echo   7  Open the dashboard      (live prices, updates itself, watch only)
echo   8  AUTOPILOT               (same page + records simulated trades)
echo   9  Show status
echo  10  Run the tests
echo.
echo   0  Exit
echo.
set /p CHOICE=  Choose a number:

if "%CHOICE%"=="1" goto :do_fetch
if "%CHOICE%"=="2" goto :do_experiment
if "%CHOICE%"=="3" goto :do_backtest
if "%CHOICE%"=="4" goto :do_validate
if "%CHOICE%"=="5" goto :do_crossval
if "%CHOICE%"=="6" goto :do_ingest
if "%CHOICE%"=="7" goto :do_dashboard
if "%CHOICE%"=="8" goto :do_auto
if "%CHOICE%"=="9" goto :do_status
if "%CHOICE%"=="10" goto :do_tests
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
echo   Trains on MNQ, ES, YM and RTY together, then predicts MNQ.
echo.
echo   The cross-instrument test (option 5) showed the pattern belongs to
echo   index futures rather than to MNQ, which makes MNQ's own 13,700 bars
echo   an arbitrary limit. Pooling gives roughly four times the training
echo   data from symbols already downloaded. Every fold still trains only
echo   on bars earlier than the window it is tested on.
echo.
echo   Then it prices the result after slippage and commission and says
echo   whether the edge is bigger than its own error bar.
echo.
echo   Takes 45-120 minutes.
echo.
"%VPY%" -m mnq.cli train --pooled
if errorlevel 1 goto :done
"%VPY%" -m mnq.cli backtest --profile wide
"%VPY%" -m mnq.cli sweep --profile wide --min-trades 15
set "HAVE_MODEL=yes"
echo.
echo   Read the PROFITABILITY block, not the backtest total. A positive
echo   total whose 95%% interval includes zero is not an edge.
echo   Then run option 4 before believing any of it.
goto :done

:do_validate
echo.
set "THRESH="
set /p THRESH=  Entry threshold to test [press Enter for 0.58]:
if "%THRESH%"=="" set "THRESH=0.58"
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

:do_ingest
echo.
echo   Turns purchased contract files into one adjusted continuous series.
echo.
echo   Yahoo's NQ=F splices the front month with no adjustment, so every
echo   quarterly roll leaves a gap of tens of points that is not a real move.
echo   Every momentum feature and every label treats it as if it were.
echo.
echo   Point this at the folder your vendor's CSVs are in. NQ is the same
echo   index as MNQ at ten times the multiplier, and it has 25 years of
echo   history instead of two.
echo.
set "SRC="
set /p SRC=  Folder containing the vendor files:
if "%SRC%"=="" goto :menu
set "VEND="
set /p VEND=  Vendor [firstrate / databento / generic, Enter for firstrate]:
if "%VEND%"=="" set "VEND=firstrate"
set "ROOTSYM="
set /p ROOTSYM=  Product root [Enter for NQ]:
if "%ROOTSYM%"=="" set "ROOTSYM=NQ"
set "SRCINT="
set /p SRCINT=  Bar interval of those files [Enter for 1m]:
if "%SRCINT%"=="" set "SRCINT=1m"
echo.
"%VPY%" -m mnq.cli ingest "%SRC%" --vendor %VEND% --root %ROOTSYM% --interval %SRCINT% --resample 1h
echo.
echo   Check the roll schedule above. Gaps should be tens of points, not
echo   hundreds, and each roll should land a few days before expiry.
goto :done

:do_dashboard
echo.
echo   Starts a local web page at http://localhost:8000
echo.
echo   It pulls fresh 1-minute prices every 60 seconds and re-reads the
echo   market on the same cadence, so the page updates by itself - you do
echo   not need to refresh it. Watch only: it says what it would do and
echo   opens nothing. Option 8 is the same page with simulated trades on.
echo.
echo   Every check and every decision is written to
echo   artifacts\decisions.jsonl and shown live on the page.
echo.
echo   Nothing leaves your machine. It opens in ITS OWN window so this menu
echo   stays usable - close that window to stop the server.
echo.
REM  `start` opens a second console and returns immediately. Running it in
REM  this window instead would block the menu for as long as the server ran,
REM  which meant opening a whole new Command Prompt just to pick option 3.
start "MNQ dashboard" cmd /k ""%VPY%" -m mnq.cli dashboard --open"
goto :done

:do_auto
echo.
echo   Runs the system unattended until you stop it.
echo.
echo   Every 5 minutes it pulls fresh bars. Every 10 minutes it looks for
echo   a signal. It manages open positions with the same rules the backtest
echo   used, writes every closed trade to a journal, and once a week it
echo   retrains on the pooled instruments.
echo.
echo   PAPER TRADING ONLY. No broker is connected and no orders are placed.
echo   It records what would have happened - which is the evidence needed
echo   before risking money, and the only honest thing to automate at this
echo   stage.
echo.
echo   The dashboard opens automatically. Watch "Paper results vs backtest":
echo   it tests whether the live win rate still matches the 37.5%% the
echo   backtest produced, and says when the sample is still too small to
echo   tell - which it will be for the first few weeks.
echo.
echo   It opens in ITS OWN window so this menu stays usable. Close that
echo   window to stop it. Progress is saved, so stopping and restarting
echo   loses nothing.
echo.
echo   Train a model first (option 3) or it can only collect bars. If you
echo   train while it is running it picks the new model up on its own.
echo.
start "MNQ autopilot" cmd /k ""%VPY%" -m mnq.cli auto --open"
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
