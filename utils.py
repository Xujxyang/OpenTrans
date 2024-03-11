import os
import random
import torch
# from networks import *
# from operator import itemgetter
import numpy as np
import math
from torch import nn
# import tqdm
import pandas as pd
import json
import re
from collections import Counter
from copy import deepcopy
from collections import OrderedDict
from weightwatcher.RMT_Util import *
from pdb import set_trace as bp


# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


import datetime
import time
def xx_days_ago(date_str, days_ago=1, format_str='%b %d, %Y'):
    """
    Returns the datetime string of xx days ago from the given datetime string.

    Parameters:
    - date_str (str): The input datetime string.
    - days_ago (int): Number of days to subtract.
    - format_str (str): Format of the datetime string.

    Returns:
    - str: The datetime string of xx days ago.
    """

    # Parse the datetime string
    dt = datetime.datetime.strptime(date_str, format_str)

    # Subtract the desired number of days
    new_dt = dt - datetime.timedelta(days=days_ago)

    # Convert the datetime object back to string
    return new_dt.strftime(format_str)



import requests
from bs4 import BeautifulSoup
import time
import datetime
def get_hf_first_commit_date(model_name):
    # URL of the web page you want to extract
    url = "https://huggingface.co/%s/commits/main"%model_name
    # Connect to the website and fetch the content
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve web page.")
        return None
    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.content, 'html.parser')
    s = soup.get_text().split("commited on")[-1].strip()
    if "day" in s and "ago" in s:
        today = datetime.datetime.today().strftime('%b %d, %Y')
        _days = int(s.split()[0])
        s = xx_days_ago(today, _days)
    if "month" in s and "ago" in s:
        today = datetime.datetime.today().strftime('%b %d, %Y')
        _months = int(s.split()[1])
        s = xx_days_ago(today, _months*30)
    if not ("," in s):
        s += ", %d"%datetime.datetime.today().year
    # return time.mktime(datetime.datetime.strptime(s, "%b %d, %Y").timetuple())
    return s

def model_names_from_csv(csv_file):
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        headers = next(csv_reader)
        model_column_index = headers.index("Model")

        for row in csv_reader:
            if len(row) > 0:
                model_value = row[model_column_index]
                model_list.append(model_value)
    return model_list

def get_hf_model_type(model_name):
    # URL of the web page you want to extract
    url = "https://huggingface.co/%s"%model_name
    # Connect to the website and fetch the content
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve web page.")
        return None
    #this prints raw html
    #print(response.text)
    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.content, 'html.parser')
    #tags = soup.find_all("a", class_="tag tag-white")
    hrefs = soup.find("a", href=re.compile("pipeline"))
    print(hrefs.get_text())
    return hrefs.get_text()

def get_model_datasource(model_name):
    huggingface_url = f"https://huggingface.co/{model_name}"
    response = requests.get(huggingface_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        model_datasource_element = soup.find_all("a", class_ = "tag mr-0 mb-0 md:mr-0 md:mb-0 tag-indigo", href=re.compile("dataset"))
        if model_datasource_element:
            model_source = [element.get_text(strip=True) for element in model_datasource_element]
            return model_source
        else:
            return ["Not Found"]
    else:
        return "Failed Connection"

def get_model_architecture(model_name):
    huggingface_url = f"https://huggingface.co/{model_name}"
    response = requests.get(huggingface_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        model_architecture_element = soup.find_all("a", class_="tag tag-purple")
        if model_architecture_element:
            model_architecture = [element.get_text(strip=True) for element in model_architecture_element if element.get_text(strip=True) != 'text-generation-inference']
            return model_architecture
        else:
            return ["Not Found"]
    else:
        return "Failed Connection"

def get_model_language(model_name):
    huggingface_url = f"https://huggingface.co/{model_name}"
    response = requests.get(huggingface_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        model_language_element = soup.find_all("a", class_="tag tag-green", href=re.compile("language"))
        if model_language_element:
            model_language = [element.get_text(strip=True) for element in model_language_element]
            return model_language
        else:
            return ["Not Found"]
    else:
        return "Failed Connection"

def get_model_downloads(model_name):
    huggingface_url = f"https://huggingface.co/{model_name}"
    response = requests.get(huggingface_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        model_downloads_element = soup.find("dd", attrs={"class": "font-semibold"})
        if model_downloads_element:
            model_downloads = model_downloads_element.get_text(strip=True)
            return model_downloads
        else:
            return "Not Found"
    else:
        return "Failed Connection"

def create_model_csv(csv_path, output_path):
    model_names = model_names_from_csv(csv_path)
    model_types, model_datasources, model_architectures, model_languages, model_downloads = [], [], [], [], []
    for model_name in model_names:
        model_types.append(get_hf_model_type(model_name))
        model_datasources.append(get_model_datasource(model_name))
        model_architectures.append(get_model_architecture(model_name))
        model_languages.append(get_model_language(model_name))
        model_downloads.append(get_model_downloads(model_name))
    model_dict = {"Name": model_names, "Type": model_types, "Data Source": model_datasources, \
    "Architecture": model_architectures, "Languages": model_languages, "Downloads": model_downloads}
    model_df = pd.DataFrame(model_dict)
    model_df.to_csv(output_path)
    return


def attn_split_qkv(module, name):
    m, n = module.weight.shape
    if m == n // 3:
        q, k, v = deepcopy(module), deepcopy(module), deepcopy(module)
        dim = m
        q.weight = torch.nn.Parameter(q.weight[:, :dim])
        k.weight = torch.nn.Parameter(k.weight[:, dim:2*dim])
        v.weight = torch.nn.Parameter(v.weight[:, 2*dim:])
        module_names = [name+"_q", name+"_k", name+"_v"]
        module_shapes = [q.weight.shape, k.weight.shape, v.weight.shape]
        modules = [q, k, v]
    elif n == m // 3:
        q, k, v = deepcopy(module), deepcopy(module), deepcopy(module)
        dim = n
        q.weight = torch.nn.Parameter(q.weight[:dim, :])
        k.weight = torch.nn.Parameter(k.weight[dim:2*dim, :])
        v.weight = torch.nn.Parameter(v.weight[2*dim:, :])
        module_names = [name+"_q", name+"_k", name+"_v"]
        module_shapes = [q.weight.shape, k.weight.shape, v.weight.shape]
        modules = [q, k, v]
    else:
        module_names = [name]
        module_shapes = [module.weight.shape]
        modules = [module]
    return module_names, module_shapes, modules


def get_module_names_shapes(model, return_modules=False):
    module_names = []
    module_shapes = []
    modules = []
    for name, module in model.named_modules():
        # if hasattr(module, 'weight') and len(module.weight.shape) > 1 and (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
        # if isinstance(module, nn.Conv2d) or isinstance(m, nn.Linear):
        if type(module).__name__.lower() in ["conv2d", "conv1d", "linear"]:
            if "attn" in name or "attention" in name:
                _module_names, _module_shapes, _modules = attn_split_qkv(module, name)
                module_names += _module_names
                module_shapes += _module_shapes
                modules += _modules
            else:
                module_names.append(name)
                # module_shapes.append(module.conv.weight.shape)
                module_shapes.append(module.weight.shape)
                modules.append(module)
    if return_modules:
        # return module_names, module_shapes, modules
        return module_names, module_shapes, nn.ModuleList(modules)
    else:
        return module_names, module_shapes

def matrix_esd_estimator(matrix, shape=None, filter_zeros=True, EVALS_THRESH=0.00001, fix_fingers='xmin_mid', xmin_pos=2, bins=100):
    #print("after weight data",torch.max(m.weight.data))
    #print("after matrix ",torch.max(matrix))
    if shape is None:
        shape = matrix.shape

    eigs = torch.square(torch.linalg.svdvals(matrix.type(torch.cuda.FloatTensor)).flatten())
    # del matrix
    # ascending order
    eigs, _ = torch.sort(eigs, descending=False)

    if filter_zeros:
        #print(f"{name} Filter Zero")
        nz_eigs = eigs[eigs > EVALS_THRESH]
        N = len(nz_eigs)
        # somethines N may equal 0, if that happens, we don't filter eigs
        if N == 0:
            #print(f"{name} No non-zero eigs, use original total eigs")
            nz_eigs = eigs
            N = len(nz_eigs)
    else:
        #print(f"{name} Skip Filter Zero")
        nz_eigs = eigs
        N = len(nz_eigs)

    spectral_norm = nz_eigs[-1].item()
    fnorm = torch.sum(nz_eigs).item()
    log_nz_eigs = torch.log(nz_eigs)

    if fix_fingers == 'xmin_mid':
        i = int(len(nz_eigs) / xmin_pos)
        xmin = nz_eigs[i]
        n = float(N - i)
        seq = torch.arange(n).cuda()
        # Xuanzhe Xiao, Zeng Li, Chuanlong Xie, and Fengwei Zhou. Heavy-tailed regularization of weight matrices in deep neural networks.
        final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
        final_D = torch.max(torch.abs(1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n))
    else:
        alphas = torch.zeros(N-1)
        Ds     = torch.ones(N-1)
        if fix_fingers == 'xmin_peak':
            hist_nz_eigs = torch.log10(nz_eigs)
            min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
            counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
            boundaries = torch.linspace(min_e, max_e, bins + 1)
            h = counts, boundaries
            ih = torch.argmax(h[0])
            xmin2 = 10 ** h[1][ih]
            xmin_min = torch.log10(0.95 * xmin2)
            xmin_max = 1.5 * xmin2

        for i, xmin in enumerate(nz_eigs[:-1]):
            if fix_fingers == 'xmin_peak':
                if xmin < xmin_min:
                    continue
                if xmin > xmin_max:
                    break

            n = float(N - i)
            seq = torch.arange(n).cuda()
            alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
            alphas[i] = alpha
            if alpha > 1:
                Ds[i] = torch.max(torch.abs(
                    1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n
                ))

        min_D_index = torch.argmin(Ds)
        final_alpha = alphas[min_D_index]
        final_D = Ds[min_D_index]

    final_alpha = final_alpha.item()
    final_D = final_D.item()
    alpha_weighted = final_alpha * np.log10(spectral_norm)
    stable_rank = fnorm / spectral_norm
    hard_rank = matrix_rank(np.sqrt(nz_eigs.cpu().numpy()), shape[1])
    entropy = matrix_entropy(np.sqrt(nz_eigs.cpu().numpy()), shape[1])
    log_alpha_norm = np.log10(np.sum([ev**final_alpha for ev in nz_eigs.cpu().numpy()]))
    log_norm = np.log10(fnorm)
    log_spectral_norm = np.log10(spectral_norm)

    return nz_eigs, final_D, final_alpha, alpha_weighted, entropy, log_alpha_norm, log_norm, log_spectral_norm, hard_rank, fnorm, spectral_norm, stable_rank


# https://github.com/CalculatedContent/WeightWatcher/blob/master/weightwatcher/WW_powerlaw.py#L52
def net_esd_estimator(
            net,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers='xmin_mid',
            xmin_pos=2,
            conv_norm=0.5,
            filter_zeros=True,
            model_name=None,
            save_dir=None,
):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = OrderedDict({
        'D': [],
        'M': [],
        'N': [],
        'alpha': [],
        'alpha_weighted': [],
        'entropy': [],
        'log_alpha_norm': [],
        'log_norm': [],
        'log_spectral_norm': [],
        'longname': [],
        'matrix_rank': [],
        'norm': [],
        'num_evals': [],
        'stable_rank': [],
        'xmax': [],
        'xmin': [],
        'spectral_norm': [],
        'params': []
        # 'eigs':[],
        })
    if model_name and save_dir:
        save_path = os.path.join(save_dir, "_%s.csv"%(model_name.replace('/', '--')))
        if os.path.exists(save_path):
            layer_stats = pd.read_csv(save_path)
            for k in results.keys():
                results[k] = layer_stats[k].tolist()
    # print("=================================")
    # print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    # print("=================================")
    # iterate through layers
    module_names, module_shapes, modules = get_module_names_shapes(net, return_modules=True)
    # for name, m in net.named_modules():
    for name, m in zip(module_names, modules):
        if name in results["longname"]: continue
        # if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        if type(m).__name__.lower() in ["conv2d", "conv1d", "linear"]:
            # if isinstance(m, nn.Linear) and max(m.weight.shape)/min(m.weight.shape) >= 8: continue # ignore classifier layer
            if type(m).__name__.lower() == "linear" and max(m.weight.shape)/min(m.weight.shape) >= 8: continue # ignore classifier layer
            matrix = m.weight.data.clone()
            matrix = matrix.cuda()
            # i have checked that the multiplication won't affect the weights value
            #print("before", torch.max(m.weight.data))
            # normalization and tranpose Conv2d
            # if type(m).__name__.lower() in ["conv2d", "conv1d"]:
            if len(matrix.shape) > 2:
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            #print("after weight data",torch.max(m.weight.data))
            #print("after matrix ",torch.max(matrix))
            nz_eigs, final_D, final_alpha, alpha_weighted, entropy, log_alpha_norm, log_norm, log_spectral_norm, hard_rank, fnorm, spectral_norm, stable_rank = matrix_esd_estimator(matrix, m.weight.shape, filter_zeros, EVALS_THRESH, fix_fingers, xmin_pos, bins)
            del matrix

            results['D'].append(final_D)
            results['M'].append(m.weight.shape[0])
            results['N'].append(m.weight.shape[1])
            results['alpha'].append(final_alpha)
            results['alpha_weighted'].append(alpha_weighted)
            results['entropy'].append(entropy)
            results['log_alpha_norm'].append(log_alpha_norm)
            results['log_norm'].append(log_norm)
            results['log_spectral_norm'].append(log_spectral_norm)
            results['longname'].append(name)
            results['matrix_rank'].append(hard_rank)
            results['norm'].append(fnorm)
            results['num_evals'].append(len(nz_eigs))
            results['spectral_norm'].append(spectral_norm)
            results['stable_rank'].append(stable_rank)
            results['xmax'].append(nz_eigs[-1].item())
            results['xmin'].append(nz_eigs[0].item())
            # results['eigs'].append(eigs.detach().cpu().numpy())
            m_parameters = filter(lambda p: p.requires_grad, m.parameters())
            params = sum([np.prod(p.size()) for p in m_parameters])
            results['params'].append(params)

            if model_name and save_dir:
                layer_stats = pd.DataFrame({key:results[key] for key in results if key!='eigs'})
                save_path = os.path.join(save_dir, "_%s.csv"%(model_name.replace('/', '--')))
                layer_stats.to_csv(save_path)

    return results


def load_data(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "gsm8k":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))
    elif args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                qes = json_res["question"].strip() + " Answer Choices:"

                for opt in json_res["options"]:
                    opt = opt.replace(')', ') ')
                    qes += f" ({opt}"

                questions.append(qes)
                answers.append(json_res["correct"])
    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "asdiv":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["Instances"]
            for line in json_data:
                q = line['input'].strip()
                a = line['output'][0]
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("addsub", "singleeq", "multiarith"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "csqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    elif args.dataset == "strategyqa":
        if 'task' in args.dataset_path:
            with open(args.dataset_path) as f:
                json_data = json.load(f)["examples"]
                for line in json_data:
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    questions.append(q)
                    answers.append(a)
        else:
            with open(args.dataset_path, encoding='utf-8') as f:
                json_data = json.load(f)
                for line in json_data:
                    q = line["question"].strip()
                    if line['answer']:
                        a = 'yes'
                    else:
                        a = 'no'
                    questions.append(q)
                    answers.append(a)
    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    elif args.dataset == 'time_zone':
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line['question'].strip()
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    else:
        raise NotImplementedError

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    return questions, answers


# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(args)->list:
    set_random_seed(args.random_seed)
    questions, answers = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset


# read the generated/prepared prompt json file
# return a string of prefix prompt before each question
def create_input_prompt(args, cot_flag:bool)->str:
    x, z, y = [], [], []

    with open(args.prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["prompt"]
        for line in json_data:
            x.append(line["question"])
            z.append(line["rationale"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))

    prompt_text = ""
    for i in index_list:
        if cot_flag:
            if args.dataset == "strategyqa":
                prompt_text += x[i] + " " + z[i] + " " + \
                            "So the answer is" + " " + y[i] + ".\n\n"
            else:
                prompt_text += x[i] + " " + z[i] + " " + \
                            args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
        else:
            prompt_text += x[i] + " " + args.direct_answer_trigger_for_fewshot + " " + y[i] + ".\n\n"
    return prompt_text


# https://github.com/shizhediao/active-prompt/tree/main
def answer_extraction(args, response):
    # TODO align pred_ans vs. response to filter out redundant output
    pred_ans = ""
    temp = response
    # temp = ""
    # if args.model == 'gpt-3.5-turbo':
    #     temp = response['choices'][0]['message']['content']
    # else:
    #     temp = response['choices'][0].text
    if args.dataset in ("gsm8k", "svamp", "asdiv", "addsub", "singleeq", "multiarith"):
        temp = temp.replace(",", "")
        temp = [s for s in re.findall(r'-?\d+\.?\d*', temp)]
    elif args.dataset in ("aqua", "csqa"):
        temp = re.findall(r'A|B|C|D|E', temp)
    elif args.dataset in ("strategyqa", "coin_flip"):
        temp = temp.lower()
        temp = re.sub("\"|\'|\n|\.|\s|\:|\,"," ", temp)
        temp = temp.split(" ")
        temp = [i for i in temp if i in ("yes", "no")]
    elif args.dataset in ("last_letters"):
        temp = re.sub("\"|\'|\n|\.|\s","", temp)
        temp = [temp]
    elif args.dataset in ('time_zone'):
        temp = temp.split('The answer is ')[-1].replace('.', '')
        temp = [temp]

    if len(temp) != 0:
        answer = temp[-1]
        # index = response.index(answer) + len(answer) - 1
        # if there is . at the end of answer, remove it
        # e.g. answer = 64.
        if answer != "":
            if answer[-1] == ".":
                answer = answer[:-1]

        # round the answer to nearest integer
        if args.dataset in ("gsm8k", "svamp"):
            try:
                answer = str(round(float(answer)))
            except:
                answer = "" # no sol or sol doesn't have valid format
        elif args.dataset in ("last_letters"):
            try:
                answer = answer[-args.concat_length:]
            except:
                answer = ""
        pred_ans = answer
    else:
        pred_ans = ""
        # index = 0
    return pred_ans
    # return pred_ans, index


def find_most_frequent(arr, n):
    # method 1: return max(arr[:n], key=arr.count)
    # method 2:
    arr_acounts = Counter(arr[:n])
    most_frequent_item, frequency = arr_acounts.most_common(1)[0]
    return frequency, most_frequent_item
