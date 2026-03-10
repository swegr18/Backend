# Hexagonal Architecture Backend

This backend follows the Hexagonal Architecture (Ports and Adapters) pattern.

## Architecture Layers

```
backend/
├── domain/              # Core business logic (innermost layer)
│   ├── entities/        # Business entities
│   └── ports/           # Interfaces/contracts
├── application/         # Use cases / Application services
│   └── use_cases/       # Business operations
├── infrastructure/      # External adapters (outermost layer)
│   ├── api/            # FastAPI HTTP adapter
│   ├── persistence/    # Database adapters
│   ├── config.py       # Configuration
│   └── container.py    # Dependency injection
└── main.py             # Application entry point
```

## Key Principles

1. **Domain Layer** - Pure business logic, no external dependencies
2. **Application Layer** - Orchestrates use cases
3. **Infrastructure Layer** - Implements ports with external technologies
4. **Dependency Rule** - Dependencies point inward (Infrastructure → Application → Domain)

## Getting Started

1. Create and activate virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\Activate.ps1
# If you encounter a permission error, run: Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate it (Mac/Linux)
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the server:

```bash
uvicorn main:app --reload
```

API available at `http://localhost:8000/api/v1`

## Docker Setup

### Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Run in detached mode
docker-compose up -d

# Stop the container
docker-compose down
```

### Using Docker directly

```bash
# Build the image
docker build -t fastapi-backend .

# Run the container
docker run -p 8000:8000 fastapi-backend

# Run with environment variables
docker run -p 8000:8000 -e DEBUG=True fastapi-backend
```

API will be available at `http://localhost:8000/api/v1`
