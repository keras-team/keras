import os
from github import Github
import filecmp
import pdfkit
import requests
import shutil

repo_id = "keras"


def send_message(pull_request, owner, files_changed, doc_link=""):
    message = """Hello, I'm a bot! I can make mistakes, notify JoshuaRM if I made one.

I see that you modified the following file{plural}: 
{files_changed}

The owner of those file is @{owner}

@{owner} could you please take a look at it whenever 
you have the time and add a review? Thank you in advance for the help.
"""

    files_changed_formatted = '\n'.join(f'* `{x}`' for x in files_changed)
    plural = 's' if len(files_changed) > 1 else ''
    message = message.format(files_changed=files_changed_formatted,
                             owner=owner,
                             plural=plural)
    
    if doc_link != "":
        message+=f"\nA PDF version of the documentation can be found [Here]({doc_link})"

    pull_request.create_issue_comment(message)


def examine_single_pull_request(pull_request):
    os.system("mkdir master pull")
    os.system("git clone https://github.com/keras-team/keras.git master")
    os.system("git clone --single-branch --branch master git@github.com:" 
              + pull_request.user.login + "/" + repo_id + ".git pull")
    
    dir_cmp = filecmp.dircmp("master/docs", "pull/docs")
    
    if dir_cmp.diff_files:
        #Build documentation
        os.chdir("pull/docs")
        os.system("python autogen.py")
        os.system("mkdocs build")

        os.chdir('./site')
        
        #Convert index.html to PDF
        try:
            with open('index.html') as html:
                pdfkit.from_file(html,'../output.pdf')
        except:
            pass           
        
        os.chdir("../../..")
        
        files = {'file':('Documentation.pdf', open('pull/docs/output.pdf', 'rb'))}
        
        #Post file
        response = requests.post('https://api.anonymousfiles.io/', files=files)
        
        #Send comment to pull request
        send_message(pull_request, pull_request.user.login, dir_cmp.diff_files, response.json()['url'])
        
        shutil.rmtree("master")
        shutil.rmtree("pull")


def examine_pull_requests():

    client = Github(os.environ['GITHUB_TOKEN'])
    repo = client.get_user().get_repo(repo_id)

    for pull_request in repo.get_pulls():
        examine_single_pull_request(pull_request)


if __name__ == '__main__':
    examine_pull_requests()