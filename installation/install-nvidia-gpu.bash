#!/bin/bash

# Shebang -> Interpreter mit Bash ausführen
# Downloas files manually:
# cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_amd64.deb
# cudnn-local-repo-ubuntu2404-9.14.0_1.0-1_amd64.deb

# Bei Fehler abbrechen etc.
set -euo pipefail

# Falls nicht admin, als admin neustarten
if [[ $EUID -ne 0 ]]; then
  echo "===Starte Skript neu mit Administratorrechten...==="
  exec sudo bash "$0" "$@"
fi

# Absoluten Pfad zum Skript und Skript-Ordner ermitteln, dann dorthin wechseln (Für das ausführen in Phase 2)
SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(cd -- "$(dirname -- "$SCRIPT_PATH")" && pwd -P)"
cd "$SCRIPT_DIR"

# Überspringen im zweiten durchlauf:
REAL_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
if [[ ! -f "$REAL_HOME/.config/autostart/cuda-resume.desktop" ]];
then

#Was für Pakete habe ich im System?
echo "===Überprüfe und update Pakete, unnötige werden vernichtet==="
apt update

# Pakete updaten
apt full-upgrade -y
sudo apt install -y build-essential dkms linux-headers-$(uname -r)

# unnötige Pakete entfernen
apt autoremove -y

# Nvidia Repo hinzufügen
echo "=== Füge NVIDIA CUDA-Repo hinzu ==="
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update

# cuda driver installieren
echo "===Installieren von Cuda-Treibern==="
apt-get install -y cuda-drivers

# Pfade und Benutzer auflösen
SCRIPT_PATH="$(readlink -f "$0")"
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"

AUTOSTART_DIR="$REAL_HOME/.config/autostart"
AUTOSTART_FILE="$AUTOSTART_DIR/cuda-resume.desktop"

# Ordner anlegen
install -d -m 755 "$AUTOSTART_DIR"

# .desktop in Autostart
cat > "$AUTOSTART_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=CUDA Resume Script
# Terminal öffnen und Skript ausführen, damit sudo nach PW fragen kann
Exec=gnome-terminal -- bash -lc '$SCRIPT_PATH --resume; read -p "Fertig. Enter zum Schließen..."'
Terminal=false
X-GNOME-Autostart-enabled=true
X-GNOME-Autostart-Delay=5
NoDisplay=false
EOF

# Besitz/Modus korrigieren
chown "$REAL_USER:$REAL_USER" "$AUTOSTART_FILE"
chmod 644 "$AUTOSTART_FILE"

echo
read -p "Soll das System jetzt neu gestartet werden? [y/N]: " answer

case "${answer,,}" in
    y|yes)
        echo "Starte neu..."
        sleep 1
        reboot 0
        ;;
    *)
        echo "Neustart abgebrochen. Starte manuell neu, wenn du bereit bist."
        ;;
esac

else

# Cuda installieren aus arbeitsverzeichnis (Quelle https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)
echo "===installieren von Cuda aus dem Arbeitsverzeichnis==="
dpkg -i cuda-repo-ubuntu2404-13-0-local_13.0.2-580.95.05-1_amd64.deb
cp /var/cuda-repo-ubuntu2404-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cuda-toolkit-13-0

# PATH setzen für Cuda
cat > /etc/profile.d/cuda.sh <<'EOF'
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH='${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda-13.0/lib64'
EOF
source /etc/profile.d/cuda.sh

# cudNN installieren aus arbeitsverzeichnis (Quelle https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=24.04&target_type=deb_local)
echo "====Installation von cudNN aus dem Arbeitsverzeichnis==="
dpkg -i cudnn-local-repo-ubuntu2404-9.14.0_1.0-1_amd64.deb
cp /var/cudnn-local-repo-ubuntu2404-9.14.0/cudnn-*-keyring.gpg /usr/share/keyrings/
apt-get update
apt-get -y install cudnn

# Python installieren, falls es nicht da ist
echo "=== Installiere Python, falls nicht vorhanden ==="
apt install -y python3

# pip installieren
echo "=== Pip wird installiert ==="
apt install python3-pip -y

# Virtuelle Umgebung erstellen und aktivieren
echo "=== Virtuelle Umgebung wird erstellt ==="
apt install -y python3 python3-venv
python3 -m venv .venv
source .venv/bin/activate

# Benötigte Bibliotheken installieren
echo "=== Python-Bibliotheken werden installiert ==="
pip install --upgrade pip
pip install ipykernel
python3 -m ipykernel install --user --name=ml_project_kernel

# Tensorflow und Jupyter Notebook installieren
echo "=== Installiere Visual Studio Code in Jupyter Notebook ==="

# Voraussetzungen
apt install -y wget gpg apt-transport-https ca-certificates

# Microsoft GPG-Key und Repo
wget -qO- https://packages.microsoft.com/keys/microsoft.asc  | gpg --dearmor > /usr/share/keyrings/packages.microsoft.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main" > /etc/apt/sources.list.d/vscode.list
apt update

# VS Code installieren
apt install -y code

# Jupyter Notebook installieren (in der venv)
pip install opencv-python tensorflow matplotlib pandas scipy jupyter notebook jupyterlab

# tensorflow [GPU] installieren
pip install "tensorflow[and-cuda]"

# Testen, ob GPU erkennbar ist und tensorflow bassd
python - << 'PY'
import tensorflow as tf
print("TensorFlow-Version:", tf.__version__)
print("Verfügbare GPUs:", tf.config.list_physical_devices('GPU'))
PY

# Pfade und Benutzer auflösen
SCRIPT_PATH="$(readlink -f "$0")"
REAL_USER="${SUDO_USER:-$USER}"
REAL_HOME="$(getent passwd "$REAL_USER" | cut -d: -f6)"

rm "$REAL_HOME/.config/autostart/cuda-resume.desktop"

fi
