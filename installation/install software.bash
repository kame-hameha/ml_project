
# 0. get a fresh start
sudo apt update
sudo apt upgrade

# 0. Install Visual Studio Code
sudo apt install apt-transport-https
sudo apt update
sudo apt install code

# 1. Install Jupyter extension in VS Code 
#    >>> Click on Extenstions (CTRL+SHIFT+X) 
#    >>> Enter Jupyter 
#    >>> Install Jupyter

# 2. Create a project folder 
#    e.g. in your home directory "/home/pi" & navigate into it
mkdir ki_project
# 2.1 Open folder
#    e.g. ki_project (in VS Code or via terminal)

# 3. Open a terminal in VS Code
# 3.1 Create a virtual environment (venv) 
python3 -m venv my_virtual_env
# 3.2 Activate the virtual environment
#     (you need to do this for every terminal, 
#      VS Code remembers the sourcing, though!)
source my_virtual_env/bin/activate
# Alternatively use absolute path:
# source /home/pi/my_virtual_env/bin/activate

# 4. Install ipykernel
pip install ipykernel
# 4.1 Create new kernel
python3 -m ipykernel install --user --name=my_project_kernel

# 5. Install packages within your Jupyter notebook
pip install opencv-python tensorflow matplotlib pandas scipy jupyter

# 6. Start & open Jupyter notebook
jupyter notebook
# 6.1 Create 
#     >>> New Jupyter Notebook
#     Alternatively use command "cmd+shift+p"
# 6.2. Select correct kernel for project >>> my_project_kernel (my_virtual_env/bin/python3)

###################################################################################
# Typtical errors you might run into
# 
1. Fehler:
qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in ""

sudo apt install python3-opencv
sudo apt install qtwayland5
sudo apt autoremove
sudo reboot now
