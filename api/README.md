# MPqa API

## Prerequisites

* HDT: dbpedia2016-04en.hdt
* ElasticSearch (sudo /etc/init.d/elasticsearch start)
* 2 pre-trained models with weights: 1) for question type classification; 2) for question parsing
* embeddings: glove840B300d

## Setup

```
conda create --name mpqa python=3.6
conda activate mpqa
pip install pybind11
pip install -r requirements.txt
```

## Test QA models

```
python request.py 
```

## Start MPqa API

```
python app.py
```

## Test MPqa API

curl -i http://localhost:5000/ask?question=What%20are%20some%20famous%20works%20of%20the%20writer%20of%20The%20Second%20Coming%3F


## Deploy

TODO


## Resources

* [Flask tutorial](https://blog.miguelgrinberg.com/post/designing-a-restful-api-with-python-and-flask)

