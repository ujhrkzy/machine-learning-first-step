## Issue テンプレート
- [ローカル開発環境構築手順 for mac](#mac)
	- [前提](#mac precondition)
	- [手順](#mac procedure)


<a id="mac"></a>
<a href="#mac"></a>  
## ローカル開発環境構築手順 for mac
<a id="mac precondition"></a>
<a href="#mac precondition"></a>  

#### 前提<br>
* brewインストール済み
* pycharmインストール済み
* python version: 3.5.2

<a id="mac procedure"></a>
<a href="#mac procedure"></a>  
#### 手順

```sh
#> install pyenv
brew install pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bash_profile

# <create virtual python environment>
cd ${作業用フォルダ}
pyenv install --list
pyenv install -v ${python version}
# e.g. pyenv install -v 3.5.2
pyenv versions
pyenv virtualenv ${python version} openpose-${python version}
# e.g. pyenv virtualenv 3.5.2 tensorboard-3.5.2
pyenv local openpose-${python version}
# e.g. pyenv local tensorboard-3.5.2
# アンインストールするときは下記コマンドを実行する。
# pyenv uninstall ${name}

# <install python library>
pip install --upgrade pip
pip install pip-tools
pip3 install -r tensorboard-req.txt

```
