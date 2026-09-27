# Tratamento de dados de atletas olímpicos

Atividade de ingestão, diagnóstico, limpeza e validação do conjunto de dados
**120 years of Olympic history: athletes and results**, disponibilizado no
[Kaggle](https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-history-athletes-and-results).

## Arquivos principais

- `tratamento_atletas_olimpicos.py`: implementação original, organizada em etapas executáveis;
- `tratamento_atletas_olimpicos.ipynb`: versão em Notebook, com o mesmo código e resultados executados;
- `data/athlete_events.csv`: base original utilizada pela atividade;
- `data/athlete_events_clean.csv`: base produzida pelo tratamento;
- `data/noc_regions.csv`: tabela auxiliar incluída no conjunto do Kaggle.

## Execução

```bash
python -m pip install -r requirements.txt
python tratamento_atletas_olimpicos.py
```

O script valida cada transformação com asserções descritas na atividade e
grava o conjunto tratado em `data/athlete_events_clean.csv`.

## Critérios atendidos

O fluxo apresenta evidências antes e depois de cada tratamento exigido pela
rubrica: carrega e lista o dataframe `atletas`, identifica/remove/reverifica
duplicatas, identifica/preenche/reverifica idades ausentes com a média e
identifica/remove/reverifica alturas ausentes. O peso é tratado adicionalmente
por mediana de sexo e modalidade, com contingência por sexo e mediana global.
