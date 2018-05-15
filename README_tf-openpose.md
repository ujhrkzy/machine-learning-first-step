## Issue テンプレート
- [ローカル開発環境構築手順 for mac](#mac)
	- [前提](#mac precondition)
	- [手順](#mac procedure)
	- [動作確認](#mac check)


<a id="mac"></a>
<a href="#mac"></a>  
## ローカル開発環境構築手順 for mac
<a id="mac precondition"></a>
<a href="#mac precondition"></a>  

#### 前提<br>
* brewインストール済み
* pycharmインストール済み
* opencvインストール済み
* protobufインストール済み
* https://github.com/ildoonet/tf-pose-estimation のsrc, 学習モデルをダウンロード済み
* python version: 3.6.0

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
# e.g. pyenv install -v 3.6.0
pyenv versions
pyenv virtualenv ${python version} openpose-${python version}
# e.g. pyenv virtualenv 3.6.0 openpose-3.6.0
pyenv local openpose-${python version}
# e.g. pyenv local openpose-3.6.0
# アンインストールするときは下記コマンドを実行する。
# pyenv uninstall ${name}

# <install python library>
pip install --upgrade pip
pip install pip-tools
pip3 install -r tf-openpose-req.txt


# <edit matplotlibrc>
# matplotlibrcファイルの場所の確認
python -c "import matplotlib;print(matplotlib.matplotlib_fname())"
# 「backend : macosx」を「backend : Tkagg」に変更する

# <edit pycharm settings>
# pycharmを開く（作業用フォルダを指定する）

# Preference -> Project -> Project Interpreter -> Add local 画面を開く
# Existing environmentにチェック
# Interpreter に /usr/local/Cellar/pyenv/{yyyymmdd}/.pyenv/versions/3.6.0/envs/openpose-{version}/bin/python3 を指定する

# Preference -> Project -> Project Structure 画面を開く
# srcフォルダを「Source」フォルダに指定する

```


<a id="mac check"></a>
<a href="#mac check"></a>  
#### 動作確認
```sh
python3 run_webcam.py
```
