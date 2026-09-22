activation patching for encoder only transformers: know the knowledge you're searching for is hidden inside which layer?! works with any hugginface masked language model. current setup is for the PubMedBERT model.

this works by running the model on a correct and corrupt prompt, the activations from the correct run are patched into the corrupt run and recovery is calculated. the layers which recover the correct answers will have the knowledge and will show a spike on the graph.

#### usage (own model and data)

1. get a model (encoder only) from hugging face, and in app.py set: 
  ```python
  MODEL_NAME = "PubMedBERT"
  MODEL_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
  ```
2. you can directly edit the facts in the gradio ui table or edit the `DEFAULT_FACTS` in `app.py`

3. serve as:
    ```bash
    uv sync
    uv run python examples/run.py 
    uv run python app.py           
    ```

#### note
1. in the template, `corrupt` should be an unrealted wrong answer.
2. `target` must be single token, else the 1st subtoken is considered.
