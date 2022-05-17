import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from models.core import ModelFactory, get_bert_base, get_bert_base_uncased, get_roberta_base, get_canine_c, \
    get_canine_s, get_electra_base, get_xlnet_base, get_big_bird, get_xlm, get_transformer_xl, get_bert_large_uncased, \
    get_electra_large, get_roberta_large
from utils.gpu_utils import cleanup


@cleanup
def analyse_model(model_factory: ModelFactory):
    print(f"Analysing {model_factory.model_hub_name}")

    tokenizer = model_factory.load_tokenizer()
    evaluate_tokenizer(tokenizer)

    model = model_factory.load_model()
    evaluate_model(model)

    print()


def evaluate_tokenizer(tokenizer: PreTrainedTokenizerBase):
    phrase = "my texttt i've just made up"
    tokenized = tokenizer([phrase])

    if len(tokenized['input_ids'][0]) - 2 == len(phrase.split(' ')):
        tokenizer_level = "word"
    elif len(tokenized['input_ids'][0]) - 2 == len(phrase):
        tokenizer_level = "character"
    else:
        tokenizer_level = "subword"

    print(f" >Tokenizer is {tokenizer_level}-level and has a vocabulary size of {tokenizer.vocab_size}.")


def evaluate_model(model: PreTrainedModel):
    """
    ref: https://discuss.pytorch.org/t/finding-model-size/130275
    """
    param_size = 0
    num_params = 0
    for param in model.parameters():
        num_params += param.nelement()
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    num_buffers = 0
    for buffer in model.buffers():
        num_buffers += buffer.nelement()
        buffer_size += buffer.nelement() * buffer.element_size()

    param_size_gb = param_size / (1024 ** 3)
    buffer_size_gb = buffer_size / (1024 ** 3)
    total_size_gb = param_size_gb + buffer_size_gb
    print(
        f" >Model size: {total_size_gb:.2f}GB (Param size: {param_size_gb:.2f}GB; Buffer size: {buffer_size_gb:.2f}GB)")

    total_params = num_params + num_buffers
    print(f" >Model size: {num_params} elements ({num_params} parameters and {num_buffers} buffers)")

    if hasattr(model, 'char_embeddings'):
        # vocab_size = model.char_embeddings.char_position_embeddings.num_embeddings
        print(f" >It uses char_embeddings and {model.char_embeddings.position_embedding_type} position embeddings")
    elif hasattr(model, "embeddings"):
        # vocab_size = model.embeddings.word_embeddings.num_embeddings
        print(
            f" >It uses embeddings.word_embeddings and {model.embeddings.position_embedding_type} position embeddings")
    elif hasattr(model, "word_emb"):
        # vocab_size = model.word_emb.n_token
        print(f" >It uses word_emb and <unspecified> position embeddings")
    else:
        # vocab_size = model.word_embedding.num_embeddings
        print(f" >It uses word_embedding and <unspecified> position embeddings")

    if model.config.is_decoder:
        architecture = "a decoder"
    elif model.config.is_encoder_decoder:
        architecture = "an encoder decoder"
    else:
        architecture = "an encoder"

    seq_len = model.config.max_position_embeddings
    print(f" >It's {architecture} with a sequence length of {seq_len}.")

    with torch.no_grad():
        model.eval()
        # Expects input ids to select from vocab, therefore seq_len:
        if seq_len == -1 or seq_len > 512:
            seq_len = 512
        output = model.forward(torch.ones(1, seq_len, requires_grad=False, device="cuda", dtype=int))

    print(f" >Produces outputs of shape {output.last_hidden_state.shape}")
    if hasattr(output, 'pooler_output'):
        print(f" >And pools outputs to {output.pooler_output.shape}")


if __name__ == "__main__":
    # BERT cased
    # Size: 0.4GB
    # Encoder w/ 512 absolute word-level embeddings
    # Vocab: 28996
    # Outputs: [1, 512, 768] & [1, 768]
    analyse_model(get_bert_base())

    # BERT uncased
    # Size: 0.41GB
    # Encoder w/ 512 absolute word-level embeddings
    # Vocab: 30522
    # Outputs: [1, 512, 768] & [1, 768]
    analyse_model(get_bert_base_uncased())

    # ROBERTA cased
    # Size: 0.46GB
    # Encoder w/ 514 absolute word-level embeddings (https://github.com/pytorch/fairseq/issues/1187)
    # Vocab: 50265
    # Outputs: [1, 512, 768] & [1, 768]
    analyse_model(get_roberta_base())

    # CANINE-c
    # Size: 0.49GB
    # Encoder w/ 16384 absolute char-level embeddings
    # Vocab: 1_114_112 (Unicode)
    # Outputs: [1, 512, 768] & [1, 768]
    analyse_model(get_canine_c())  # OOM issues

    # CANINE-s
    # Size: 0.49GB
    # Encoder w/ 16384 absolute char-level embeddings
    # Vocab: 1_114_112 (Unicode)
    # Outputs: [1, 512, 768] & [1, 768]
    analyse_model(get_canine_s())  # OOM issues

    # ELECTRA
    # Size: 0.41GB
    # Encoder w/ 512 absolute word-level embeddings
    # Vocab: 30522
    # Outputs: [1, 512, 768] (no pools)
    analyse_model(get_electra_base())

    # BigBird
    # Size: 0.47GB
    # Encoder w/ 4096 absolute word-level embeddings
    # Vocab: 50358 (SentencePiece, but works like WordPiece)
    # Outputs: [1, 512, 768] & [1, 768]
    analyse_model(get_big_bird())

    # XLNet
    # Size: 0.43GB
    # Encoder w/ -1 relative position word-level embeddings
    # Vocab: 32000
    # Outputs: [1, 512, 768] (no pools)
    analyse_model(get_xlnet_base())

    # XLM
    # Size: 1.04GB
    # Encoder w/ 514 absolute position word-level embeddings
    # Vocab: 250052
    # Outputs: [1, 512, 768] & [1, 768]
    analyse_model(get_xlm())

    # BERT-Large (uncased)
    # Size: 1.25GB
    # Encoder w/ 512 absolute word-level embeddings
    # Vocab: 30522
    # Outputs: [1, 512, 1024] & [1, 1024]
    analyse_model(get_bert_large_uncased())

    # ELECTRA-Large
    # Size: 1.24GB
    # Encoder w/ 512 absolute word-level embeddings
    # Vocab: 30522
    # Outputs: [1, 512, 1024] (no pools)
    analyse_model(get_electra_large())

    # RoBERTa-Large
    # Size: 1.25GB
    # Encoder w/ 514 absolute word-level embeddings (https://github.com/pytorch/fairseq/issues/1187)
    # Vocab: 50265
    # Outputs: [1, 512, 1024] & [1, 1024]
    analyse_model(get_roberta_large())
