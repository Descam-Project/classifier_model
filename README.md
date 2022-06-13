# classifier_model
Machine learning classifier model updates

1. IPYNB is used for data preprocessing and model building
2. Dataset consisted of text advertisement of investment offers, labeled legal/illegal
3. H5 is the exported machine learning model
4. JSON Tokenizer is exported from preprocessing process, containing tokens word words from dataset. Will be used to convert new words based on the tokens to sequence for prediction.
5. predict.py is the flask endpoint, to be used for the app/backend to access and interact with the model for processing incoming predictions
