@echo off
cd /d "d:\doc\wangsx\Agent memory"

echo === Step 1: Remove temp files from staging ===
git rm --cached ".git_add_output.txt" 2>nul
git rm --cached ".git_status_output.txt" 2>nul
del ".git_add_output.txt" 2>nul
del ".git_status_output.txt" 2>nul

echo === Step 2: Re-stage all files with updated .gitignore ===
git add -A

echo === Step 3: Configure git user (if not set) ===
git config user.name >nul 2>&1
if errorlevel 1 (
    git config user.name "Agent Memory Team"
    git config user.email "team@agentmemory.dev"
    echo Git user configured: Agent Memory Team
) else (
    echo Git user already configured: 
    git config user.name
)

echo === Step 4: Create first commit ===
git commit -m "feat: initial commit - 金融长文档智能阅读理解系统

- Frontend: React + Vite + Tailwind CSS dashboard
- Backend: FastAPI Python service with Qwen LLM integration
- Features: PDF preprocessing, retrieval, reasoning pipeline
- Dataset: public_dataset_a (financial contracts, reports, insurance, regulatory, research)
- CI/CD: GitHub Actions workflow
- Docs: architecture design docs (MVP + full chain)"

echo === Step 5: Show commit result ===
git log --oneline -1
echo.
echo Done! Repository initialized successfully.

del "_setup_git.bat"
