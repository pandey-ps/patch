import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # to run from examples directory(here)

from tracer import CausalTracer
from dataset import build_dataset
FACTS = [
    ("Insulin", "Aspirin",
     "The hormone that regulates blood sugar is {}",
     "Insulin", "sugar", "pain"),
    ("Aspirin", "Insulin",
     "The pill often used to reduce fever is {}",
     "Aspirin", "fever", "sugar"),
    ("Penicillin", "Aspirin",
     "The first antibiotic discovered to fight infection was {}",
     "Penicillin", "infection", "pain"),
    ("Vaccine", "Aspirin",
     "The injection that builds immunity against disease is a {}",
     "Vaccine", "immunity", "pain"),
    ("Dopamine", "Insulin",
     "The brain chemical linked to reward is {}",
     "Dopamine", "brain", "blood"),
    ("Asthma", "Diabetes",
     "The chronic condition that causes wheezing is {}",
     "Asthma", "wheezing", "thirst"),
    ("Malaria", "Asthma",
     "The mosquito-borne disease that causes chills is {}",
     "Malaria", "mosquito", "pollen"),
    ("Calcium", "Iron",
     "The mineral that strengthens bones is {}",
     "Calcium", "bones", "blood"),
    ("Oxygen", "Calcium",
     "The gas carried by red blood cells is {}",
     "Oxygen", "blood", "bone"),
    ("Anemia", "Asthma",
     "The blood disorder caused by low iron is {}",
     "Anemia", "iron", "dust"),
]

MODEL_NAME = "PubMedBERT"
MODEL_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


def warn_multitoken(check_model_id: str) -> None:
    # target has to be a single token; else the 1st subtoken is used.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(check_model_id)
    for clean, _corr, _tpl, target, _cf, _ct in FACTS:
        ids = tok.encode(target, add_special_tokens=False)
        if len(ids) != 1:
            print(f"  [warn] target {target!r} -> {len(ids)} tokens "
                  f"for {check_model_id}; using 1st subtoken only.")


def main() -> None:
    # prompt pairs as dataset
    dataset = build_dataset(FACTS)

    warn_multitoken(MODEL_ID)

    print(f"Loading {MODEL_NAME} ({MODEL_ID}) …")
    tracer = CausalTracer(MODEL_NAME, MODEL_ID)
    result = tracer.trace(dataset)
    tracer.release()
    peak_layer, peak_val = result.peak_layer()
    print(f"  peak L{peak_layer} ({peak_val:.2e}), "
          f"clean−corrupt {result.clean_minus_corrupt:+.2e}")


if __name__ == "__main__":
    main()
