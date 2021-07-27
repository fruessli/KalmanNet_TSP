# KalmanNet_TSP

## How to run:

Git can be used like normal in Colab but with ! in front. Eg, !git clone.\
`!` is used to run shell commands.
We can also run files with `!python filename.py`.

We can also push pull etc like the normal console. The space on colab is only temporary, so we need to make sure that **save often anything you want to keep.**\
Saving colab files on GitHub is fine though. But we have clone etc. every time we open the files on Colab anew. **The space on Colab is only temporary!**\
We can simply use a push at the end of my file, so we can save easy at the end.

Basic Git commands: https://confluence.atlassian.com/bitbucketserver/basic-git-commands-776639767.html\
Notable commands:<br>
`git clone /path/to/repository` or `git clone https://github.com/user/repository`<br>
`git add <filename>`<br>
`git commit -m "Commit message"`<br>
`git push origin <branchname>`<br>
`git checkout <branchname>`<br>
`git pull`

1. Clone the repository.
'!git clone https://github.com/fruessli/KalmanNet_TSP_fr'
2. Move to the correct folder.
'%cd KalmanNet_TSP_fr/'
3. Switch to the correct branch.
'!git checkout main-branch'
4. Run the files.
'!python test.py'
5. Edit the files by simply double clicking into them. The changes are saved automatically.

It is also possible to commit, push etc in Colab, however I recommended to log in while cloning.
'uname = "fruessli"
!git config --global user.email 'name@email.com'
!git config --global user.name '$uname'

from getpass import getpass
password = getpass('Password:')
!git clone https://$uname:$password@github.com/fruessli/KalmanNet_TSP_fr
%cd KalmanNet_TSP_fr/'

Now the files can easily be edited etc.

Commiting and pushing is done as usual.

'!git add .
!git commit -m "commit msg"
!git push origin branchname'
