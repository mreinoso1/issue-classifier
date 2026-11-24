import httpx
import pandas as pd

def params_dict(labels: str) -> dict:
    return {"labels": labels, "state": "closed","per_page":100}

def filter_pr(json_aux:dict, json_url:list) -> None:
    for k in range(len(json_aux)):
        if json_aux[k].get('pull_request'): #Cleaning issues that are PR
            continue
        json_url.append(json_aux[k])

def next_page(url:str, headers:dict, current_params:dict | None,client:httpx.Client,json_url:list[dict]) -> None:
    """
    Appends the cleaned issues of each page to json_url
    """
    while url:
        response = client.get(url=url,headers=headers,params=current_params)
        response.raise_for_status()
        json_aux = response.json()
        filter_pr(json_aux=json_aux,json_url=json_url)
        url = response.links.get("next",{}).get('url')
        current_params = None
    
def add_to_finalDf(clean_json:dict,repos_list:str,params_labels:str) -> pd.DataFrame:
    """
    Used after next_page(). It adds the APIs to df_final via pandas concatenation
    """
    if not clean_json:
        return pd.DataFrame(columns=["number", "title", "body", "repo", "label"])
    df_aux = pd.json_normalize(clean_json)[['number','title','body',]]
    df_aux['repo'] = repos_list
    df_aux['label'] = params_labels
    return df_aux

def issues_getter(repos_conf,url_list:list[str], headers:dict, client:httpx.Client)-> pd.DataFrame:
    
    df_final = pd.DataFrame()
    i=0
    dfs = []
    for repo_name, repo_labels in repos_conf.items():
        
        for labels, github_label in repo_labels.items():
            url = url_list[i]
            json_url = []
            params = params_dict(github_label)
            next_page(url=url,headers=headers,current_params=params,client = client,json_url=json_url)
            if len(json_url) == 0:
                continue
            dfs.append(add_to_finalDf(clean_json=json_url,repos_list=repo_name,params_labels=labels))

        i+=1
        
    if not dfs:
        return pd.DataFrame(columns=["number", "title", "body", "repo", "label"])

    df_final = pd.concat(dfs,ignore_index=True)
    df_final = df_final.drop_duplicates(['number','repo'], keep=False)
    return df_final
