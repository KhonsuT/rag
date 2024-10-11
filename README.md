# Instructions

## Requirements
- Poertry v1.6.1
- Python v3.10.12
- Docker v24.0.7
- Docker Compose v2.29.3
- Disk space 6Gb minimum

## Formating rules
The pyproject comes with black install, be sure to run blacks before pushing your updates:
```
poetry run python -m black <file/directory>
```

## Hosting Ollama Locally
### To Install Ollama
```
curl -fsSL https://ollama.com/install.sh | sh
```
Can also be installed manually, ref: https://github.com/ollama/ollama/blob/main/docs/linux.md

### Adding Ollama as a startup service
Create user group for ollama
```
sudo useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama
sudo usermod -a -G ollama $(whoami)
```
Create a service file in /etc/systemd/system/ollama.service:
```
[Unit]
Description=Ollama Service
After=network-online.target

[Service]
ExecStart=/usr/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="PATH=$PATH"
Environment="OLLAMA_HOST=0.0.0.0:11434" #only needed when running in wsl

[Install]
WantedBy=default.target
```

### Pulling specific model
For list of ollama supported models ref: https://ollama.com/library?sort=popular
```
ollama pull <modeld-name-and-version>
```
For example, when pulling latest llava:
```
ollama pull llava:latest
```

## Hosting langfuse locally

Running in docker container:
# Clone the Langfuse repository
```
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# Start the server and database
docker compose up
```
Navigate to http://www.localhost:3000/

### Next Steps:
## Installing POSTGRES for LANGFUSE
Langfuse uses postgres by default to store its data, it is required for langfuse.
Instructions on postgres installation:
```
docker pull postgres
```
After successfully pulling docker image
```
docker run --name myPostgresDb -p 5455:5432 -e POSTGRES_USER=postgresUser -e POSTGRES_PASSWORD=postgresPW -e POSTGRES_DB=postgresDB -d postgres
```
Check postgres is running by
```
docker container list
```

## LANGFUSE additional environments
langfuse might require additional environment variables to run
Set user email by:
```
export LANGFUSE_INIT_USER_EMAIL="<your email>"
```

After above steps restart langfuse server:
```
docker container restart langfuse-langfuse-server-1

# or navigate to langfuse folder 

docker compose down
docker compose up
```
- Register an account with langfuse and login
- Create a project
- Navigate to settings and create an api key, update API key in the python file. 

# Troubleshooting
## incompatible chromabd version
for some reason later version of python comes with a old version of sqlite3 which chromadb is incompatible with. 
Chromadb requires 3.34 or newer. Update the poetry env to use 3.10.12 seem to resolve this issue
```
poetry env use 3.10.12
poetry install
```