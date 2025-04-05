
#!/bin/bash

# 1. Update und Upgrade
echo "=== System wird aktualisiert ==="
sudo apt update && sudo apt upgrade -y

# Notwendige Pakete und Visual Studio Code installieren
sudo apt install -y apt-transport-https code

# 2. Git-Repository herunterladen ins Home-Verzeichnis
echo "=== ml_project Repo herunterladen ==="
cd ~
git clone https://github.com/kame-hameha/ml_project
cd ml_project

# 3. Virtuelle Umgebung erstellen und aktivieren
echo "=== Virtuelle Umgebung wird erstellt ==="
python3 -m venv .venv
source .venv/bin/activate

# 4. Benötigte Bibliotheken installieren
echo "=== Python-Bibliotheken werden installiert ==="
pip install --upgrade pip
pip install ipykernel
python3 -m ipykernel install --user --name=ml_project_kernel

# 5. Weitere Pakete installieren (auch innerhalb Jupyter nutzbar)
pip install opencv-python tensorflow matplotlib pandas scipy jupyter

echo "=== Installation abgeschlossen ==="

echo "Neustart"
sudo reboot now

# Starte Visual Studio Code via: "code --js-flags="--nodecommit_pooled_pages"" 