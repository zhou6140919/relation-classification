# Relation Classification

Relation Classification on NYT29 Dataset

## Preparation

```bash
conda create -n relation python=3.11 -y
conda activate relation
pip install -r requirements.txt
```

## Pre-Processing

Files:
- relation.txt 29 relation types
- train.sent training sentences
- train.tup training relation tuples from training sentences

Put the downloaded dataset into `datasets` folder and rename it to `datasets/NYT29`

1. Convert the relations into label ids.
    
    {'/location/administrative_division/country': 0, '/location/country/capital': 1, '/location/country/administrative_divisions': 2, '/location/neighborhood/neighborhood_of': 3, '/location/location/contains': 4, '/people/person/nationality': 5, '/people/person/place_lived': 6, '/people/deceased_person/place_of_death': 7, '/business/person/company': 8, '/location/us_state/capital': 9, '/people/person/place_of_birth': 10, '/people/person/children': 11, '/business/company/founders': 12, '/business/company/place_founded': 13, '/sports/sports_team/location': 14, '/people/person/ethnicity': 15, '/people/ethnicity/geographic_distribution': 16, '/people/person/religion': 17, '/business/company/major_shareholders': 18, '/location/province/capital': 19, '/location/br_state/capital': 20, '/business/company/advisors': 21, '/film/film_location/featured_in_films': 22, '/film/film/featured_film_locations': 23, '/location/us_county/county_seat': 24, '/time/event/locations': 25, '/people/deceased_person/place_of_burial': 26, '/people/place_of_interment/interred_here': 27, '/business/company_advisor/companies_advised': 28}
    
    If `add_other` is `True` , the model will be trained on all possible entity pairs, and there will be another class. {’other’: 29}. 
    
    ```bash
    Train set: 341562
    Dev set: 37964
    Test set: 36422
    ```
    
    If `add_other` is `False` , the model will be trained only on the given entity pairs.
    
    ```bash
    Train set: 78973
    Dev set: 8766
    Test set: 5859
    ```
    
2. Construct input text
    
    {'context': 'then terrorism struck again , this time in the <e1>indonesia</e1> capital of <e2>jakarta</e2> . What is the relation between <e1>indonesia</e1> and <e2>jakarta</e2>?', 'label': 2}
    
    I replaced the entity pair with special tokens <e1> and <e2> surrounded. And a sentence to ask what is the relation between entity 1 and entity 2.
    
3. Max sequence length
    ```bash
    python ./pre-processing/statistics.py
    ```
    
    The max length of those input text is about 310 tokens (already tokenized by tokenizer).

```bash
bash ./scripts/preprocess.sh
```
    

## Training

Edit `./config/config.json` to adjust hyperparameters.

```bash
bash train.sh
```

I used `bert-large-uncased` as the base model to do the sequence classification task.

The evaluation metric is `F1` score.

The learning rate is `1e-5` .

The total train epochs is `10`.

## Evaluation

Remember to do the Pre-Processing step before evaluation.

```bash
bash test.sh
```
