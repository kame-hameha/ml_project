
# Projektdateien
1. install software.bash (nicht ausführen, sondern Zeile für Zeile > Terminal)
2. create-symbols-dataset.py
3. training-symbols-dataset.ipynb 
4. (Optional) training-symbols-dataset-comparison.ipynb # Mehrfache Trainings starten 
5. (Optional) Datensätze (Symbols 2023): dataset1, dataset2, dataset3 (440 Datenpunkte)

###################################################################################
# https://medium.com/@claudia.nikel/how-to-setup-a-jupyter-notebook-in-vs-code-w-virtual-env-kernels-install-packages-884cf643375e
# get a fresh start
sudo apt update
sudo apt upgrade

# Install inside venv <>
0. Install Visual Studio Code
sudo apt install apt-transport-https
sudo apt update
sudo apt install code

1. Install Jupyter extension in VS Code 
>>> Click on Extenstions (CTRL+SHIFT+X) >>> Enter Jupyter >>> Install Jupyter

2. Create a project folder (e.g. in home directory /home/pi) & navigate into it
mkdir ki-project
2.1 Open folder (ki-project) in VS Code

3. Create a virtual environment
python3 -m venv my_virtual_env
4. Activate a virtual environment
source my_virtual_env/bin/activate

5. Install ipykernel
pip install ipykernel
6. Create new kernel
python3 -m ipykernel install --user --name=myproject_kernel

7. Install packages within your Jupyter notebook
pip install opencv-python
pip install tensorflow
pip install matplotlib
pip install pandas
pip install scipy
pip install jupyter

8. Start & open Jupyter notebook
jupyter notebook
8.1 Create: New Jupyter Notebook (cmd+shift+p)
8.2. Select correct kernel for project >>> myproject_kernel (my_virtual_env/bin/python3)
###################################################################################
# Typtical errors
# 
1. Fehler:
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""

sudo apt install python3-opencv
sudo apt install qtwayland5
sudo apt autoremove
sudo reboot now
