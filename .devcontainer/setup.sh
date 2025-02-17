sudo pip install --upgrade pip
sudo pip install -r requirements.txt
echo "bash shell/lint.sh" > .git/hooks/pre-commit
chmod a+x .git/hooks/pre-commit
