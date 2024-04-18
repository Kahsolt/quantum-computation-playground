#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/04/18

import os
import json
from pprint import pprint as pp
from re import compile as Regex
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from time import sleep
from traceback import print_exc
from typing import List, Dict, Any

import requests as R
from bs4 import BeautifulSoup

BASE_PATH = Path(__file__).parent
KEY_FILE = BASE_PATH / 'API_KEY.txt'
TMP_FILE = BASE_PATH / 'tmp_alive_board.json'

class Src(Enum):
  Github = 'github'
  Gitee = 'gitee'
  PyQPanda = 'pyqpanda'
  PyVQNet = 'pyvqnet'

@dataclass
class Repo:
  url: str
  name: str
  org: str = ''
  stars: int = -1
  last_commit: str = ''
  total_commits: int = -1
  last_release: str = ''
  last_version: str = ''

  @property
  def src(self) -> Src:
    if 'github.com' in self.url: return Src.Github
    if 'gitee.com' in self.url: return Src.Gitee
    if 'pyqpanda' in self.url: return Src.PyQPanda
    if 'vqnet' in self.url: return Src.PyVQNet

REPO_LIST = [
  # quantum computing
  Repo('https://github.com/Qiskit/qiskit', 'Qiskit', 'IBM'),
  Repo('https://github.com/quantumlib/Cirq', 'Cirq', 'Google'),
  Repo('https://github.com/microsoft/qsharp', 'Q#', '微软'),
  Repo('https://github.com/NVIDIA/cuQuantum', 'cuQuantum', 'nVidia'),
  Repo('https://github.com/PennyLaneAI/pennylane', 'PennyLane', 'Xanadu'),
  Repo('https://github.com/XanaduAI/strawberryfields', 'StrawberryFields', 'Xanadu'),
  Repo('https://github.com/ProjectQ-Framework/ProjectQ', 'ProjectQ', 'ETH Zurich'),
  Repo('https://github.com/mit-han-lab/torchquantum', 'torchquantum', 'MIT'),
  Repo('https://github.com/tensorflow/quantum', 'tensorflow-quantum', 'Google'),

  Repo('https://gitee.com/mindspore/mindquantum', 'Mindquantum', '华为'),
  Repo('https://github.com/PaddlePaddle/Quantum', 'PaddleQuantum', '百度'),
  Repo('https://github.com/tencent-quantum-lab/tensorcircuit', 'TensorCircuit', '腾讯'),
  Repo('https://github.com/tencent-quantum-lab/TenCirChem', 'TenCirChem', '腾讯'),
  Repo('https://github.com/qiboteam/qibo', 'QiBo', '清华'),
  Repo('https://github.com/QuantumBFS/Yao.jl', 'Yao', '幺'),
  Repo('https://github.com/OriginQ/QPanda-2', 'QPanda-2', '本源量子'),
  Repo('https://pyqpanda-toturial.readthedocs.io/zh/latest/2.pyqpanda简介/index.html#id5', 'pyqpanda', '本源量子'),
  Repo('https://vqnet20-tutorial.readthedocs.io/en/latest/rst/CHANGELOG.html', 'pyvqnet', '本源量子'),
  Repo('https://github.com/OriginQ/pyChemiQ', 'pyChemiQ', '本源量子'),
  Repo('https://gitee.com/arclight_quantum/isq', 'isQ', '弧光量子'),
  Repo('https://github.com/SpinQTech/SpinQit', 'SpinQit', '量旋科技'),
  Repo('https://github.com/qudoor/qutrunk', 'QuTrunk', '启科量子'),

  Repo('https://github.com/tequilahub/tequila', 'Tequila', 'Tequila'),
  Repo('https://github.com/qulacs/qulacs', 'Qulacs', 'Qulacs'),
  Repo('https://github.com/softwareQinc/staq', 'staq', 'staq'),
  Repo('https://github.com/eclipse-qrisp/Qrisp', 'Qrisp', 'Qrisp'),

  # tensor networks
  Repo('https://github.com/google/TensorNetwork', 'TensorNetwork', 'Google'),
  Repo('https://github.com/tenpy/tenpy', 'TeNPy', 'TeNPy'),
]

R_COMMITS = Regex('([,\d]+) Commits')
R_VER = Regex('[vV]?[\d\.]+')
R_DATE = Regex('\d+-\d+-\d+')

def parse_github(html:BeautifulSoup, repo:Repo):
  try:
    resp = R.get(repo.url.replace('github.com/', 'api.github.com/repos/'), headers=HEADERS)
    assert resp.ok
    data: Dict[str, Any] = resp.json()
    repo.stars = data['stargazers_count']
  except: print('>> Error: stars')
  try:
    resp = R.get(repo.url.replace('github.com/', 'api.github.com/repos/') + '/commits', headers=HEADERS, json={'per_page': 1})
    assert resp.ok
    data: Dict[str, Any] = resp.json()
    if len(data):
      repo.last_commit = data[0]['commit']['committer']['date']
  except: print('>> Error: last_commit')
  try:
    spans = html.find_all('span')
    for span in spans:
      if R_COMMITS.match(span.text):
        seg: str = R_COMMITS.findall(span.text)[0]
        repo.total_commits = int(seg.replace(',', ''))
        break
  except: print('>> Error: total_commits')
  try:
    resp = R.get(repo.url.replace('github.com/', 'api.github.com/repos/') + '/releases', headers=HEADERS, json={'per_page': 1})
    assert resp.ok
    data: Dict[str, Any] = resp.json()
    if len(data):
      repo.last_release = data[0]['published_at']
      ver = data[0]['name']
      try: ver = R_VER.findall(data[0]['name'])[0]
      except: pass
      repo.last_version = ver
  except: print('>> Error: last_release & last_version')

def parse_gitee(html:BeautifulSoup, repo:Repo):
  try:
    seg: str = html.find('span', attrs={'class': 'star-container'}).find('a', attrs={'class': 'action-social-count'}).get('title')
    repo.stars = int(seg.strip())
  except: print('>> Error: stars')
  try:
    seg: str = html.find('div', attrs={'class': 'recent-commit'}).find('span', attrs={'class': 'timeago'}).get('datetime')
    repo.last_commit = seg.strip()
  except: print('>> Error: last_commit')
  try:
    seg: str = html.find('div', attrs={'class': 'all-commits'}).text
    repo.total_commits = int(seg.strip().split(' ')[0])
  except: print('>> Error: total_commits')
  try:
    seg: str = html.find('div', attrs={'class': 'release'}).find('span', attrs={'class': 'timeago'}).get('datetime')
    repo.last_release = seg.strip()
  except: print('>> Error: last_release')
  try:
    seg: str = html.find('div', attrs={'class': 'release'}).find('div', attrs={'class': 'desc'}).text
    repo.last_version = seg.strip()
  except: print('>> Error: last_version')

def parse_pyqpanda(html:BeautifulSoup, repo:Repo):
  try:
    seg: str = html.find('section', attrs={'id': 'id5'}).find('section').find('h2').text
    repo.last_version = R_VER.findall(seg)[0]
    repo.last_release = R_DATE.findall(seg)[0]
  except: print('>> Error: last_version & last_release')

def parse_pyvqnet(html:BeautifulSoup, repo:Repo):
  try:
    seg: str = html.find('section', attrs={'id': 'vqnet-changelog'}).find('section').find('h2').text
    repo.last_version = R_VER.findall(seg)[0]
    repo.last_release = R_DATE.findall(seg)[0]
  except: print('>> Error: last_version & last_release')


def load_tmp() -> List[Repo]:
  repos = []
  if TMP_FILE.exists():
    with open(TMP_FILE, 'r', encoding='utf-8') as fh:
      cached = json.load(fh)
    for repo in cached:
      print(f'>> loading {repo["name"]}')
      repos.append(Repo(**repo))
      pp(repo)
  return repos

def save_tmp(repos:List[Repo]):
  data = [vars(r) for r in repos]
  with open(TMP_FILE, 'w', encoding='utf-8') as fh:
    json.dump(data, fh, indent=2, ensure_ascii=False)


def fmt_datetime_str(s:str) -> str:
  if ' ' in s: return s.split(' ')[0]
  if 'T' in s: return s.split('T')[0]
  return s

def fmt_markdown(repos:List[Repo]):
  HEAD_NAMES = [
    'name',
    'organization',
    'github stars',
    'last commit / total commits',
    'last release',
  ]
  HEADER = '|' + ''.join([f' {head} |' for head in HEAD_NAMES])
  SEP = '|' + ' :-: |' * len(HEAD_NAMES) 
  ROWS = []
  for repo in repos:
    segs = [
      f'[{repo.name}]({repo.url})',
      repo.org,
      repo.stars if repo.stars > 0 else '',
      f'{fmt_datetime_str(repo.last_commit)}' + (f' / {repo.total_commits}' if repo.total_commits > 0 else ''),
      f'{fmt_datetime_str(repo.last_release)}' + (f' ({repo.last_version})' if repo.last_version else ''),
    ]
    row = '|' + ''.join([f' {seg} |' for seg in segs])
    ROWS.append(row)

  now_ts = str(datetime.now()).split(' ')[0]
  lines = [f'ℹ update time: {now_ts}', '', HEADER, SEP] + ROWS
  return '\n'.join(lines)


if __name__ == '__main__':
  API_KEY = None
  if KEY_FILE.exists():
    with open(KEY_FILE, 'r', encoding='utf-8') as fh:
      API_KEY = fh.read().strip()
  API_KEY = os.getenv('API_KEY', API_KEY)

  HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
  } if API_KEY is not None else {}

  # load tmp cache
  repos = load_tmp()

  # update info
  for repo in REPO_LIST:
    if repo.url in [r.url for r in repos]: continue
    print(f'>> updating {repo.name}')

    try:
      resp = R.get(repo.url, headers=HEADERS)
      assert resp.ok
      html = BeautifulSoup(resp.content, features='html.parser')
      parser = globals()[f'parse_{repo.src.value}']
      parser(html, repo)
      repos.append(repo)
      pp(vars(repo))
      sleep(0.5)
    except KeyboardInterrupt: break
    except:
      print_exc()
      print(f'>> Failed for {repo.name}: {repo.url}')

  # save tmp cache
  save_tmp(repos)

  # check if all things ok
  if len(repos) < len(REPO_LIST):
    print('>> Error: some repo failed to update, just try again!')
    exit(-1)

  # format markdown
  print()
  print('=' * 72)
  print()
  markdown_str = fmt_markdown(repos)
  print(markdown_str)
