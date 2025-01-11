1.Tensorflow
https://www.server-world.info/en/note?os=Ubuntu_24.04&p=tensorflow&f=2

2. NVIDIA : CUDA 12.0 : install
https://www.server-world.info/en/note?os=Ubuntu_24.04&p=nvidia&f=2

3. NVIDIA : Install Graphic Driver
https://www.server-world.info/en/note?os=Ubuntu_24.04&p=nvidia

4. Python 3.12 : Install
https://www.server-world.info/en/note?os=Ubuntu_24.04&p=python&f=1

5. Optional: Install VS Code
5.1 Download .deb package
https://code.visualstudio.com/docs/setup/linux
5.2 Sign in into github & MS (Copilot)
5.3 Install Github Desktop
wget -qO - https://apt.packages.shiftkey.dev/gpg.key | gpg --dearmor | sudo tee /usr/share/keyrings/shiftkey-packages.gpg > /dev/null
sudo sh -c 'echo "deb [arch=amd64 signed-by=/usr/share/keyrings/shiftkey-packages.gpg] https://apt.packages.shiftkey.dev/ubuntu/ any main" > /etc/apt/sources.list.d/shiftkey-packages.list'
sudo apt update && sudo apt install github-desktop


### Problems / Error fixing

1. Add CUDA GPU support in Python / Jupyter files:

# Enable Nvidia GPUs by un-commenting this line 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/"

2. 
