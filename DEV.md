## Development Guide

### Setup a local dev environment

- **Create and activate a virtual environment** (recommended; do this once per machine):

```bash
# From the project root
python -m venv venv

# Mac / Linux
source venv/bin/activate

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# If you encounter a permission error, run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

- **Install dependencies** (inside the virtualenv):

```bash
pip install -r requirements.txt
```

### Code Coverage Report

- **Generate a Terminal Text Report**:

```bash
# Mac / Linux
PYTHONPATH=. pytest --cov=. --cov-report=term-missing

# Windows (PowerShell)
$env:PYTHONPATH="."; pytest --cov=. --cov-report=term-missing
```
- **Generate a HTML Report (open in browser)**:

```bash
# Mac / Linux
PYTHONPATH=. pytest --cov=. --cov-report=html

# Windows (PowerShell)
$env:PYTHONPATH="."; pytest --cov=. --cov-report=html
```


### Running Tests

- **Start a local Postgres for testing** (Docker, Mac/Linux/Windows with Docker):

```bash
docker run --name backend-test-db \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=audiodb \
  -p 5432:5432 \
  -d postgres:16
```

- **Run all tests**:

```bash
# Mac / Linux
PYTHONPATH=. pytest

# Windows (PowerShell)
$env:PYTHONPATH="."; pytest
```

- **Run only unit / application-layer tests**:

```bash
# Mac / Linux
PYTHONPATH=. pytest tests/test_auth_use_cases.py -q

# Windows (PowerShell)
$env:PYTHONPATH="."; pytest tests/test_auth_use_cases.py -q
```

- **Run API-level end-to-end tests**  
  (these load the real FastAPI app in memory with `TestClient`, so you do not need to start a server or a database):

```bash
# Mac / Linux
PYTHONPATH=. pytest tests/test_end_to_end.py -q

# Windows (PowerShell)
$env:PYTHONPATH="."; pytest tests/test_end_to_end.py -q

```

- **Run metrics unit tests**  
  (these exercise the audio/transcription metrics logic in `metrics.py` with lightweight stubs):

```bash
# Mac / Linux
PYTHONPATH=. pytest tests/test_metrics.py -q

# Windows (PowerShell)
$env:PYTHONPATH="."; pytest tests/test_metrics.py -q

```

- **Run Postgres-backed tests as well**  
  (requires a running Postgres instance; locally we typically point to `localhost:5432`):

```bash
# Mac / Linux
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/audiodb"
export TEST_WITH_DB=1
PYTHONPATH=. pytest tests/test_auth_postgres_integration.py -q

# Windows (cmd)
set DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/audiodb
set TEST_WITH_DB=1
PYTHONPATH=. pytest tests/test_auth_postgres_integration.py -q

# Windows (PowerShell)
$env:DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/audiodb"
$env:TEST_WITH_DB = "1"
$env:PYTHONPATH="."; pytest tests/test_auth_postgres_integration.py -q
```
