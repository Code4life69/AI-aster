@echo off
cd /d "C:\Ai Asistant"

echo.
echo ==========================================
echo   Aster GitHub Update Helper
echo ==========================================
echo.

git status
if errorlevel 1 (
    echo.
    echo Git is not working in this folder.
    echo Make sure this is the repo folder and Git is installed.
    pause
    exit /b 1
)

echo.
echo Pulling latest changes from GitHub first...
git pull --ff-only origin main
if errorlevel 1 (
    echo.
    echo Pull failed. Read the message above.
    pause
    exit /b 1
)

echo.
echo Current status after pull:
git status

echo.
set /p msg=Enter commit message: 
if "%msg%"=="" set msg=Update latest changes

echo.
echo Staging project files...
git add -A
git reset -- "Update-Aster-GitHub.bat" >nul 2>&1
git reset -- "Update-Aster-GitHub-Safe.bat" >nul 2>&1

echo.
echo Final staged files:
git status

echo.
echo Committing...
git commit -m "%msg%"
if errorlevel 1 (
    echo.
    echo Commit did not complete.
    echo This usually means there were no staged changes to commit.
    pause
    exit /b 1
)

echo.
echo Pushing to GitHub...
git push origin main
if errorlevel 1 (
    echo.
    echo Push failed. Read the message above.
    pause
    exit /b 1
)

echo.
echo Done. Your latest changes should now be on GitHub.
pause
