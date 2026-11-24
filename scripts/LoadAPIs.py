import os

import httpx
from dotenv import load_dotenv

from issue_classifier.ingest.ingest import issues_getter

load_dotenv()
api_token = os.getenv("API_TOKEN", "")
if not api_token:
    raise RuntimeError("API_TOKEN is missing")

headers = {
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2026-03-10',
    'Authorization': f'Bearer {api_token}'
}
params_labels = ['feature','bug','docs','support']
feat_labels = ['feature', 'New Feature','c:new feature','Feature request']
bugs_labels = ['bug', 'Bug','bug', 'bug']
doc_labels = ['docs', 'Documentation', 'documentation','Documentation']
question_labels = ['question','Question', 'question','Usage']
url_list = [
            "https://api.github.com/repos/fastapi/fastapi/issues",
            "https://api.github.com/repos/scikit-learn/scikit-learn/issues",
            "https://api.github.com/repos/supabase/supabase/issues",
            "https://api.github.com/repos/huggingface/transformers/issues"
            ]
repos_list = ["fastapi","scikit-learn","supabase","transformers"]

repo_conf = {
    'fastapi': {
        'feature':'feature',
        'bug':'bug',
        'doc':'docs',
        'question':'question'
    },
    'scikit':{
        'feature':'New Feature',
        'bug':'Bug',
        'doc':'Documentation',
        'question':'Question'
    },
    'supabase':{
        'feature':'c:new feature',
        'bug':'bug',
        'doc':'documentation',
        'question':'question'
    },
    'transformers':{
        'feature':'Feature request',
        'bug':'bug',
        'doc':'Documentation',
        'question':'Usage'
    }
}

with httpx.Client(timeout=30.0) as client:
    df_final = issues_getter(repos_conf=repo_conf,url_list=url_list, headers=headers,client=client)

    print(df_final)