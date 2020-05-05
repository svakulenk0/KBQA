# MPqa API

## Prerequisites

* HDT
* Embeddings
* ElasticSearch (sudo /etc/init.d/elasticsearch start)

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
