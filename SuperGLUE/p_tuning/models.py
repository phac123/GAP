from transformers import AutoTokenizer, AutoModelForSequenceClassification
def create_model(args):
    MODEL_CLASS, _ = get_model_and_tokenizer_class(args)
    model = MODEL_CLASS.from_pretrained(args.model_name)

    return model

def get_model_and_tokenizer_class(args):
    if 'gpt2' in args.model_name:
        return AutoModelForSequenceClassification, AutoTokenizer

    return AutoModelForSequenceClassification, AutoTokenizer

def get_embedding_layer(args, model):
    if 'gpt2' in args.model_name:
        embeddings = model.get_input_embeddings()
    else:
        raise NotImplementedError()

    return embeddings