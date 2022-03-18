**<details><summary>Click here to install *using Docker*</summary><p>**

**<details><summary>What is Docker?</summary><p>**

[This tutorial explains why Docker is useful.](https://www.youtube.com/watch?v=YFl2mCHdv24)
</p></details>


**<details><summary>Do you have Docker installed? If not, *click here*</summary><p>**

**<details><summary>Linux/MacOS &nbsp;&nbsp; *(command line)*</summary><p>**

<p>
Download and install Docker 1.21 or greater for [Linux](https://docs.docker.com/engine/installation/) or [MacOS](https://store.docker.com/editions/community/docker-ce-desktop-mac).

> Consider using a Docker 'convenience script' to install (search on your OS's Docker installation webpage).

Launch docker according to your Docker engine's instructions, typically ``docker start``.  

> **Note:** You must have sudo or root access to *install* Docker. If you do not wish to *run* Docker as sudo/root, you need to configure user groups as described here: https://docs.docker.com/install/linux/linux-postinstall/
</p></details>


**<details><summary>Windows &nbsp;&nbsp; *(GUI & command line)*</summary><p>**

<p>
Download and install [Docker Toolbox for Windows](https://docs.docker.com/toolbox/toolbox_install_windows/). 

Launch Kitematic.

> If on first startup Kitematic displays a red error suggesting that you run using VirtualBox, do so.

> **Note:** [Docker Toolbox for MacOS](https://docs.docker.com/toolbox/toolbox_install_mac/) has not yet been tested.
<p>

</p></details>
</p></details>

\
A Dockerfile is provided to build images with CUDA support. Build from the github repo:
```
docker build -t topaz https://github.com/tbepler/topaz.git
```

or download the source code and build from the source directory
```
git clone https://github.com/tbepler/topaz
cd topaz
docker build -t topaz .
```
</p></details>