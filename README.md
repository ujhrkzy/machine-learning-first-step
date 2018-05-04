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
* python version: 3.6.0
* https://github.com/ujhrkzy/machine-learning-first-step 直下のファイルを作業用フォルダに配置済み


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
pyenv virtualenv ${python version} ml-1st-step-${python version}
# e.g. pyenv virtualenv 3.6.0 ml-1st-step-3.6.0
pyenv local ml-1st-step-${python version}
# e.g. pyenv local ml-1st-step-3.6.0
# アンインストールするときは下記コマンドを実行する。
# pyenv uninstall ${name}

# <install python library>
pip install --upgrade pip
pip install pip-tools
pip3 install -r requirements.txt


# <edit matplotlibrc>
# matplotlibrcファイルの場所の確認
python -c "import matplotlib;print(matplotlib.matplotlib_fname())"
# 「backend : macosx」を「backend : Tkagg」に変更する

# <edit pycharm settings>
# pycharmを開く（作業用フォルダを指定する）

# Preference -> Project -> Project Interpreter -> Add local 画面を開く
# Existing environmentにチェック
# Interpreter に /usr/local/Cellar/pyenv/{yyyymmdd}/.pyenv/versions/3.6.0/envs/ml-1st-step-{version}/bin/python3 を指定する

# Preference -> Project -> Project Structure 画面を開く
# cnn, dnn, linear_regression, logistic_regressionフォルダを「Source」フォルダに指定する

```