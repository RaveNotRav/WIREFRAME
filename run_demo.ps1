param(
    [switch]$Clean,
    [switch]$SeedAdmin
)

if ($Clean) {
    Write-Host "Cleaning state..." -ForegroundColor Yellow
    if (Test-Path "wireframe.db") { Remove-Item "wireframe.db" -Force }
    if (Test-Path "uploads") { Remove-Item -Recurse -Force "uploads" }
    if (Test-Path "outputs") { Remove-Item -Recurse -Force "outputs" }
}

if (-not (Test-Path "venv")) {
    Write-Host "Checking for Python 3.12..." -ForegroundColor Cyan
    $pyCheck = py -3.12 -c "print('ok')" 2>&1
    if ($pyCheck -notmatch "ok") {
        Write-Host "ERROR: Python 3.12 is required but not found. Please install Python 3.12." -ForegroundColor Red
        exit 1
    }
    Write-Host "Creating virtual environment with Python 3.12..." -ForegroundColor Cyan
    py -3.12 -m venv venv
}

# Activate venv
. .\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r requirements.txt --quiet

# Seed DB if it doesn't exist or is clean
$db_exists = Test-Path "wireframe.db"

$env:PYTHONPATH="."
$env:WIREFRAME_MOCK="1"

if ($SeedAdmin) {
    python scripts\seed_demo_admin.py
} elseif (-not $db_exists) {
    Write-Host "Initializing Database and Migrations..." -ForegroundColor Cyan
    python -c "from backend.database import get_connection, run_migrations; run_migrations(get_connection())"
    
    # Generate random passwords
    $inspector_pw = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 12 | % {[char]$_})
    $lgu_pw = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 12 | % {[char]$_})
    $admin_pw = -join ((48..57) + (65..90) + (97..122) | Get-Random -Count 12 | % {[char]$_})
    
    if ($env:INSPECTOR_PW) { $inspector_pw = $env:INSPECTOR_PW }
    if ($env:LGU_PW) { $lgu_pw = $env:LGU_PW }
    if ($env:ADMIN_PW) { $admin_pw = $env:ADMIN_PW }
    
    python -c "
import sqlite3
from backend.database import get_connection
from backend.auth import hash_password

conn = sqlite3.connect('wireframe.db')
def make_hash_salt(pw):
    h, s = hash_password(pw)
    return h, s

h1, s1 = make_hash_salt('$inspector_pw')
conn.execute('INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)', 
             ('demo_inspector', h1, s1, 'field_inspector'))

h2, s2 = make_hash_salt('$lgu_pw')
conn.execute('INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)', 
             ('demo_lgu', h2, s2, 'lgu_engineer'))

h3, s3 = make_hash_salt('$admin_pw')
conn.execute('INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)', 
             ('demo_admin', h3, s3, 'admin'))

conn.commit()
conn.close()

with open('passwords_demo_admin.txt', 'w', encoding='utf-8') as f:
    f.write('username: demo_admin\npassword: $admin_pw\n')
"
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "Demo users created:" -ForegroundColor Green
    Write-Host "field_inspector : demo_inspector / $inspector_pw"
    Write-Host "lgu_engineer    : demo_lgu       / $lgu_pw"
    Write-Host "admin           : demo_admin     / $admin_pw"
    Write-Host "==========================================" -ForegroundColor Green
}

Write-Host "Starting WIREFRAME in DEMO MOCK mode..." -ForegroundColor Green
Write-Host "Backend API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:8000/analyze.html" -ForegroundColor Cyan

python -m uvicorn main:app --host 0.0.0.0 --port 8000
