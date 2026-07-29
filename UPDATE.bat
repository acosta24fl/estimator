@echo off
REM ===================================================================
REM  APPLY AN UPDATE  -  drag a .patch file onto this icon.
REM
REM  Why this exists: updates used to mean downloading a zip, extracting
REM  it to a new folder, and running from there. That produced a dozen
REM  "mnqtradingsystem (11)" folders, none of them a git checkout, so
REM  patches could not be applied and nothing could be pushed. One folder
REM  that is a real clone plus this script replaces all of it.
REM
REM  Usage, either way round:
REM      drag  live-dashboard.patch  onto UPDATE.bat
REM      or run UPDATE.bat and paste the path when it asks
REM
REM  NOTE: flat labels, no if ( ... ) blocks around anything that sets and
REM  then reads a variable. cmd.exe expands every %VAR% in a parenthesised
REM  block once, at parse time, so a variable set inside the block still
REM  reads as its old value further down it. That bug cost an afternoon in
REM  MNQ.bat; it is not repeated here.
REM ===================================================================

cd /d "%~dp0"
echo.
echo ============================================================
echo   Applying an update
echo ============================================================
echo   Folder : %CD%
echo.

REM ---- this has to be a real clone, or nothing below can work ----
git rev-parse --git-dir >nul 2>&1
if errorlevel 1 goto :not_a_repo

set "PATCHFILE=%~1"
if not "%PATCHFILE%"=="" goto :have_patch

echo   Drag the .patch file into this window and press Enter
echo   (or paste its full path).
echo.
set /p PATCHFILE=  Patch file:
if "%PATCHFILE%"=="" goto :no_patch

:have_patch
REM  Dragging a path with spaces into a console wraps it in quotes; ~ strips
REM  them so the exist test and git see a bare path either way.
set "PATCHFILE=%PATCHFILE:"=%"
if not exist "%PATCHFILE%" goto :missing_patch

echo   Patch  : %PATCHFILE%
echo.

REM ---- refuse to run on a dirty tree: git am would fail halfway ----
git diff --quiet
if errorlevel 1 goto :dirty
git diff --cached --quiet
if errorlevel 1 goto :dirty

REM  Clear any half-finished apply left by an earlier attempt.
git am --abort >nul 2>&1

echo   Fetching the latest first...
git pull --ff-only
echo.

echo   Applying...
git am "%PATCHFILE%"
if errorlevel 1 goto :apply_failed

echo.
echo   Applied. Pushing...
git push
if errorlevel 1 goto :push_failed

echo.
echo ============================================================
echo   Done. Run MNQ.bat - the new code is live.
echo ============================================================
echo.
git log --oneline -3
echo.
pause
exit /b 0

:not_a_repo
echo   [X] This folder is not a git checkout, so updates cannot be
echo       applied to it.
echo.
echo       You are probably running from an extracted zip. Clone the
echo       repository once, and work in that folder from now on:
echo.
echo         cd %%USERPROFILE%%\Desktop
echo         git clone https://github.com/acosta24fl/estimator.git mnq
echo         cd mnq
echo         git checkout claude/mnq-futures-trading-system-y49pzi
echo.
echo       Your downloaded price history and trained models are NOT in
echo       this folder - they live in %%USERPROFILE%%\mnq-data and are
echo       reused automatically. Nothing is lost by switching folders.
echo.
pause
exit /b 1

:no_patch
echo   Nothing to apply.
pause
exit /b 1

:missing_patch
echo   [X] No file at:
echo       %PATCHFILE%
echo.
echo       If you downloaded a .zip, extract it first - git cannot read
echo       a patch that is still inside a zip.
echo.
pause
exit /b 1

:dirty
echo   [X] This folder has uncommitted changes, and applying a patch on
echo       top of them would fail halfway and leave a mess.
echo.
echo       To throw your local changes away and take the update:
echo         git reset --hard
echo.
echo       To keep them for later:
echo         git stash
echo.
git status --short
echo.
pause
exit /b 1

:apply_failed
echo.
echo   [X] The patch did not apply.
echo.
echo       Most likely this folder is not at the commit the patch was
echo       written against. Nothing has been changed - the apply was
echo       rolled back. Send the message above and it can be rebuilt.
echo.
git am --abort >nul 2>&1
pause
exit /b 1

:push_failed
echo.
echo   [!] Applied locally, but the push failed.
echo.
echo       The commit is safe in this folder. Usually this means the
echo       remote moved on; try:
echo         git pull --rebase
echo         git push
echo.
pause
exit /b 1
